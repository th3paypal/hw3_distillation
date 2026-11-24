from __future__ import annotations

import gc
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .cache import cache_signature, cache_teacher_logits, load_logit_cache
from .data import load_data
from .evaluation import evaluate, save_evaluation
from .models import (assert_student_budget, build_model, factorize_model, load_model,
                     make_teacher, save_model, shared_from_student, student_spec)
from .pruning import (calibration_batches, initialize_by_depth_pruning,
                      layer_importance, prune_transformer)
from .training import TrainConfig, train
from .utils import file_hash, parameter_counts, read_json, seed_everything, write_json

TRAIN_EXPERIMENTS = ["teacher", "factorized", "student_kd", "pruned_init", "shared", "shared_lora"]
ALL_EXPERIMENTS = TRAIN_EXPERIMENTS + ["wanda", "sparsegpt"]


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_teacher_cache(data_dir, runs_dir, device):
    raw, encoded, tokenizer, metadata = load_data(data_dir)
    checkpoint = Path(runs_dir) / "teacher" / "best.pt"
    teacher_status = read_json(Path(runs_dir) / "teacher" / "status.json")
    if teacher_status["status"] != "complete":
        raise RuntimeError("сначала завершите обучение учителя, затем сохраняйте его логиты")
    cache_dir = Path(runs_dir) / "teacher_cache"
    expected = cache_signature(checkpoint, metadata)
    if (cache_dir / "metadata.json").exists():
        load_logit_cache(cache_dir, expected)
        return
    model, _, saved = load_model(checkpoint)
    if saved.get("data_signature") != metadata["signature"]:
        raise ValueError("сохранённые веса учителя получены на других подготовленных данных")
    model.to(device)
    cache_teacher_logits(model, encoded["train"], tokenizer, device, checkpoint,
                         metadata, cache_dir)
    del model, raw, encoded
    release_memory()


def train_experiment(name, data_dir, runs_dir, config_dir, device,
                     resume=False, overrides=None, time_budget_hours=None):
    recipe = read_json(Path(config_dir) / f"{name}.json")
    settings = {**recipe["training"], **{k: v for k, v in (overrides or {}).items() if v is not None}}
    config = TrainConfig(**settings)
    seed_everything(config.seed)
    raw, encoded, tokenizer, metadata = load_data(data_dir)
    output = Path(runs_dir) / name
    if (output / "run.json").exists() and not resume:
        raise FileExistsError(f"запуск {name} уже существует; добавьте --resume для продолжения")
    run_metadata = {"experiment": name, "initialization": recipe["initialization"]}
    if resume and (output / "last.pt").exists():
        # продолжение восстанавливает последний шаг, а итоговая оценка использует лучшие веса
        model, spec, _ = load_model(output / "last.pt")
        run_metadata = read_json(output / "run.json")["metadata"]
    elif name == "teacher":
        request = metadata["request"]
        model, spec = make_teacher(request["model_name"], request["model_revision"])
    elif name == "student_kd":
        spec = student_spec(len(tokenizer), tokenizer.pad_token_id, **recipe.get("student", {}))
        model = build_model(spec)
    else:
        source_name = recipe["source"]
        source_path = Path(runs_dir) / source_name / "best.pt"
        if read_json(source_path.parent / "status.json")["status"] != "complete":
            raise RuntimeError(f"сначала завершите {source_name}, затем создавайте {name}")
        source, source_spec, saved = load_model(source_path)
        if saved.get("data_signature") != metadata["signature"]:
            raise ValueError("исходная модель обучалась на других подготовленных данных")
        run_metadata.update(source=source_name, source_sha256=file_hash(source_path))
        if name == "factorized":
            source.to(device)
            model, spec = factorize_model(source, source_spec, rank=64)
        elif name in {"shared", "shared_lora"}:
            model, spec = shared_from_student(source, source_spec, groups=2,
                                              lora_rank=8 if name == "shared_lora" else 0)
        elif name == "pruned_init":
            batches = calibration_batches(encoded["train"], tokenizer, samples=256, seed=config.seed)
            scores = layer_importance(source, batches, device)
            model, spec, selected = initialize_by_depth_pruning(source, source_spec, scores,
                                                                keep_layers=2, rank=64)
            run_metadata.update(layer_importance=scores, selected_teacher_layers_zero_based=selected)
            del batches
        else:
            raise ValueError(name)
        del source
        release_memory()
    if name not in {"teacher", "factorized"}:
        assert_student_budget(model)
    logits = None
    if config.kd_weight:
        checkpoint = Path(runs_dir) / "teacher" / "best.pt"
        signature = cache_signature(checkpoint, metadata)
        logits = load_logit_cache(Path(runs_dir) / "teacher_cache", signature)
        run_metadata["teacher_sha256"] = signature["checkpoint_sha256"]
    tqdm.write(f"{name}: {parameter_counts(model)['parameters']:,} уникальных параметров")
    result = train(model, spec, encoded, raw, tokenizer, config, output, device,
                   metadata["signature"], logits, resume=resume,
                   time_budget_hours=time_budget_hours, run_metadata=run_metadata)
    del model, logits
    release_memory()
    return result


def prune_experiment(method, data_dir, runs_dir, device, source_name="student_kd",
                     sparsity=0.5, samples=256, seed=42):
    output = Path(runs_dir) / method
    source_path = Path(runs_dir) / source_name / "best.pt"
    raw, encoded, tokenizer, metadata = load_data(data_dir)
    settings = {"source": source_name, "source_sha256": file_hash(source_path),
                "method": method, "sparsity": sparsity, "samples": samples,
                "seed": seed, "data_signature": metadata["signature"]}
    if (output / "run.json").exists():
        old = read_json(output / "run.json")
        if old.get("pruning") != settings:
            raise ValueError("настройки прореживания изменились; укажите другую папку через --runs-dir")
        if (output / "status.json").exists() and read_json(output / "status.json")["status"] == "complete":
            tqdm.write(f"уже завершено: {method}")
            return
        raise RuntimeError(f"в {output} осталось незавершённое прореживание; укажите другую папку через --runs-dir")
    if read_json(source_path.parent / "status.json")["status"] != "complete":
        raise RuntimeError("сначала завершите обучение исходной компактной модели")
    seed_everything(seed)
    model, spec, saved = load_model(source_path)
    if saved.get("data_signature") != metadata["signature"]:
        raise ValueError("исходная модель обучалась на других подготовленных данных")
    assert_student_budget(model)
    model.to(device)
    # сравниваем качество на одной выборке до изменения весов и после него
    before, _, _ = evaluate(model, encoded["validation"], raw["validation"], tokenizer, device,
                             description="качество до прореживания")
    batches = calibration_batches(encoded["train"], tokenizer, samples=samples, seed=seed)
    start = time.monotonic()
    details = prune_transformer(model, batches, device, method, sparsity)
    seconds = time.monotonic() - start
    after, _, _ = evaluate(model, encoded["validation"], raw["validation"], tokenizer, device,
                            description="качество после прореживания")
    output.mkdir(parents=True, exist_ok=True)
    save_model(model, spec, output / "best.pt", pruning=settings, validation=after,
               data_signature=metadata["signature"])
    write_json(output / "run.json", {"pruning": settings, "metadata": {"experiment": method},
                                     **parameter_counts(model)})
    write_json(output / "pruning.json", {"matrices": details, "seconds": seconds,
                                         "validation_before": before, "validation_after": after,
                                         "zero_parameters": sum(int(p.eq(0).sum()) for p in model.parameters())})
    write_json(output / "status.json", {"status": "complete", "elapsed_seconds": seconds,
                                        "best_f1": after["f1"]})
    del model, batches
    release_memory()


def evaluate_experiment(name, data_dir, runs_dir, device, split="test"):
    raw, encoded, tokenizer, metadata = load_data(data_dir)
    output = Path(runs_dir) / name
    checkpoint = output / "best.pt"
    if read_json(output / "status.json")["status"] != "complete":
        raise RuntimeError(f"завершите {name} перед итоговой оценкой")
    model, _, saved = load_model(checkpoint)
    if saved.get("data_signature") != metadata["signature"]:
        raise ValueError("сохранённые веса получены на других подготовленных данных")
    model.to(device)
    metrics = save_evaluation(model, encoded[split], raw[split], tokenizer, device,
                              checkpoint, output, split)
    split_name = {"train": "обучение", "validation": "проверка", "test": "тест"}.get(split, split)
    tqdm.write(f"{name}: F1 ({split_name})={metrics['f1']:.4f}")
    del model
    release_memory()
    return metrics
