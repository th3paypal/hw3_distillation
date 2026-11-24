from __future__ import annotations

import argparse
import time
from pathlib import Path

from tqdm.auto import tqdm

from .data import prepare_data
from .experiments import (ALL_EXPERIMENTS, TRAIN_EXPERIMENTS, ensure_teacher_cache,
                          evaluate_experiment, prune_experiment, train_experiment)
from .utils import choose_device, read_json


def parser():
    p = argparse.ArgumentParser(description="сжатие моделей распознавания сущностей на Kaggle")
    p.add_argument("--data-dir", type=Path, default=Path("data/conll2003"))
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--config-dir", type=Path, default=Path("configs"))
    p.add_argument("--device", default="auto")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", default="ashaba1in/conll2003")
    prepare.add_argument("--model", default="bert-base-cased")
    prepare.add_argument("--max-length", type=int, default=128)
    prepare.add_argument("--dataset-revision")
    prepare.add_argument("--model-revision")
    train = sub.add_parser("train")
    train.add_argument("experiment", choices=TRAIN_EXPERIMENTS)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--steps", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--accumulation-steps", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--time-budget-hours", type=float)
    sub.add_parser("cache-teacher")
    prune = sub.add_parser("prune")
    prune.add_argument("method", choices=["wanda", "sparsegpt"])
    prune.add_argument("--source", default="student_kd")
    prune.add_argument("--sparsity", type=float, default=0.5)
    prune.add_argument("--samples", type=int, default=256)
    prune.add_argument("--seed", type=int, default=42)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("experiment", choices=ALL_EXPERIMENTS + ["all"])
    evaluation.add_argument("--split", choices=["validation", "test"], default="test")
    pipeline = sub.add_parser("pipeline")
    pipeline.add_argument("--stage", choices=["base", "compression", "all"], default="all")
    pipeline.add_argument("--time-budget-hours", type=float, default=10.5)
    return p


def run_pipeline(args, device):
    if not (args.data_dir / "metadata.json").exists():
        prepare_data(args.data_dir)
    stage = args.stage
    selected = TRAIN_EXPERIMENTS[:3] if stage == "base" else TRAIN_EXPERIMENTS[3:] if stage == "compression" else TRAIN_EXPERIMENTS
    methods = [] if stage == "base" else ["wanda", "sparsegpt"]
    start = time.monotonic()
    with tqdm(total=len(selected) + len(methods), desc="этапы запуска", unit="этап",
              mininterval=1, dynamic_ncols=True) as progress:
        for name in selected:
            progress.set_postfix_str(name)
            status_path = args.runs_dir / name / "status.json"
            # завершённые этапы пропускаем, остальные продолжаем с последнего сохранения
            if status_path.exists() and read_json(status_path)["status"] == "complete":
                tqdm.write(f"уже завершено: {name}")
                progress.update(1)
                continue
            if (time.monotonic() - start) / 3600 >= args.time_budget_hours:
                tqdm.write("время сессии закончилось; повторите ту же команду, чтобы продолжить")
                return
            recipe = read_json(args.config_dir / f"{name}.json")
            if recipe["training"].get("kd_weight", 0):
                ensure_teacher_cache(args.data_dir, args.runs_dir, device)
            # создание кэша учителя тоже расходует общий бюджет времени
            remaining = args.time_budget_hours - (time.monotonic() - start) / 3600
            if remaining <= 0:
                tqdm.write("время сессии закончилось; повторите ту же команду, чтобы продолжить")
                return
            result = train_experiment(name, args.data_dir, args.runs_dir, args.config_dir, device,
                                      resume=(args.runs_dir / name / "last.pt").exists(),
                                      time_budget_hours=remaining)
            if result["status"] != "complete":
                tqdm.write(f"этап {name} приостановлен; повторите ту же команду, чтобы продолжить")
                return
            progress.update(1)
        for method in methods:
            progress.set_postfix_str(method)
            if (time.monotonic() - start) / 3600 >= args.time_budget_hours:
                tqdm.write("время закончилось до прореживания весов; повторите команду, чтобы продолжить")
                return
            prune_experiment(method, args.data_dir, args.runs_dir, device)
            progress.update(1)
    tqdm.write("обучение и сжатие завершены")


def main(argv=None):
    args = parser().parse_args(argv)
    device = choose_device(args.device)
    if args.command == "prepare":
        result = prepare_data(args.data_dir, args.dataset, args.model, args.max_length,
                               args.dataset_revision, args.model_revision)
        print(result["sentence_counts"])
        return
    if args.command == "train":
        overrides = {"max_steps": args.steps, "epochs": args.epochs,
                     "batch_size": args.batch_size, "accumulation_steps": args.accumulation_steps,
                     "learning_rate": args.learning_rate, "seed": args.seed}
        if args.epochs is not None and args.steps is None:
            overrides["max_steps"] = 0
        recipe = read_json(args.config_dir / f"{args.experiment}.json")
        if recipe["training"].get("kd_weight", 0):
            ensure_teacher_cache(args.data_dir, args.runs_dir, device)
        train_experiment(args.experiment, args.data_dir, args.runs_dir, args.config_dir,
                          device, args.resume, overrides, args.time_budget_hours)
    elif args.command == "cache-teacher":
        ensure_teacher_cache(args.data_dir, args.runs_dir, device)
    elif args.command == "prune":
        prune_experiment(args.method, args.data_dir, args.runs_dir, device,
                         args.source, args.sparsity, args.samples, args.seed)
    elif args.command == "evaluate":
        names = ALL_EXPERIMENTS if args.experiment == "all" else [args.experiment]
        with tqdm(names, desc="оценка моделей", unit="модель",
                  mininterval=1, dynamic_ncols=True) as progress:
            for name in progress:
                progress.set_postfix_str(name)
                if args.experiment == "all" and not (args.runs_dir / name / "best.pt").exists():
                    continue
                evaluate_experiment(name, args.data_dir, args.runs_dir, device, args.split)
    elif args.command == "pipeline":
        run_pipeline(args, device)


if __name__ == "__main__":
    main()
