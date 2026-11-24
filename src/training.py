from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from .data import NERCollator
from .evaluation import evaluate
from .models import save_model
from .objectives import distillation_loss
from .utils import (atomic_torch_save, autocast_context, model_inputs, parameter_counts,
                    read_json, seed_everything, stable_hash, write_json)


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 5
    max_steps: int = 0
    batch_size: int = 16
    accumulation_steps: int = 2
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    eval_every: int = 400
    log_every: int = 50
    patience: int = 8
    min_delta: float = 0.0001
    temperature: float = 3.0
    kd_weight: float = 0.0
    amp: bool = True


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    name, values, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(values, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def epoch_batches(size, batch_size, seed, epoch):
    # порядок примеров зависит от номера эпохи и воспроизводится после перезапуска
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    return [order[i:i + batch_size] for i in range(0, size, batch_size)]


def train(model, spec, encoded, raw, tokenizer, config: TrainConfig, output: Path,
          device, data_signature: str, teacher_logits=None, resume=False,
          time_budget_hours=None, run_metadata=None):
    output = Path(output)
    if config.batch_size < 1 or config.accumulation_steps < 1:
        raise ValueError("размер батча и число шагов накопления градиента должны быть положительными")
    if config.kd_weight and teacher_logits is None:
        raise ValueError("для дистилляции нужны сохранённые логиты учителя")
    if not 0 <= config.warmup_ratio < 1:
        raise ValueError("доля шагов прогрева warmup_ratio должна быть в диапазоне [0, 1)")
    seed_everything(config.seed)
    output.mkdir(parents=True, exist_ok=True)
    identity = stable_hash({"config": asdict(config), "spec": asdict(spec),
                            "data": data_signature, "metadata": run_metadata})
    last_path = output / "last.pt"
    if (output / "run.json").exists() and not resume:
        raise FileExistsError(f"в {output} уже есть запуск; добавьте --resume или укажите другую папку через --runs-dir")
    model.to(device)
    # не применяем штраф за величину весов к смещениям и коэффициентам нормализации
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim == 1 else decay).append(parameter)
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                    {"params": no_decay, "weight_decay": 0.0}],
                                   lr=config.learning_rate)
    num_batches = math.ceil(len(encoded["train"]) / config.batch_size)
    steps_per_epoch = math.ceil(num_batches / config.accumulation_steps)
    total_steps = config.max_steps or config.epochs * steps_per_epoch
    if total_steps < 1:
        raise ValueError("нужно задать хотя бы один шаг обучения")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, round(total_steps * config.warmup_ratio)),
        num_training_steps=total_steps)
    use_scaler = config.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    state = dict(step=0, epoch=0, next_batch=0, best_f1=-1.0, bad_evals=0,
                 elapsed_seconds=0.0, status="running")
    history = []
    if resume:
        if not last_path.exists():
            raise FileNotFoundError(f"нет файла для продолжения обучения: {last_path}")
        payload = torch.load(last_path, map_location="cpu", weights_only=True)
        if payload["identity"] != identity:
            raise ValueError("настройки, данные или исходная модель отличаются от сохранённого запуска")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        state, history = payload["state"], payload["history"]
        restore_rng(payload["rng"])
        if state["status"] == "complete":
            print(f"запуск уже завершён: {output}")
            return state
    write_json(output / "run.json", {"identity": identity, "config": asdict(config),
                                     "spec": asdict(spec), "data_signature": data_signature,
                                     "metadata": run_metadata or {}, **parameter_counts(model)})
    start = time.monotonic()
    previous_seconds = state["elapsed_seconds"]

    def save_state():
        # для точного продолжения нужны не только веса, но и состояние оптимизатора и случайности
        state["elapsed_seconds"] = previous_seconds + time.monotonic() - start
        atomic_torch_save({"identity": identity, "spec": asdict(spec),
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                           "state": state, "history": history, "rng": rng_state()}, last_path)
        write_json(output / "history.json", history)
        write_json(output / "status.json", state)

    def validate():
        metrics, _, _ = evaluate(model, encoded["validation"], raw["validation"], tokenizer,
                                  device, config.eval_batch_size, config.amp,
                                  description="проверка качества")
        history.append({"step": state["step"], "split": "validation", **metrics})
        # лучшие веса выбираем по проверочной выборке, не обращаясь к тестовой
        if metrics["f1"] > state["best_f1"]:
            if metrics["f1"] > state["best_f1"] + config.min_delta:
                state["bad_evals"] = 0
            else:
                state["bad_evals"] += 1
            state["best_f1"] = metrics["f1"]
            save_model(model, spec, output / "best.pt", step=state["step"],
                       validation=metrics, data_signature=data_signature)
        else:
            state["bad_evals"] += 1
        tqdm.write(f"шаг {state['step']}: F1 на проверке={metrics['f1']:.4f}; лучший={state['best_f1']:.4f}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    description = {
        "teacher": "обучение учителя",
        "factorized": "обучение модели с факторизацией",
        "student_kd": "дистилляция ученика",
        "pruned_init": "обучение ученика после удаления слоёв",
        "shared": "обучение модели с общими весами",
        "shared_lora": "обучение общих весов и адаптеров",
    }.get(output.name, f"обучение: {output.name}")
    progress = tqdm(total=total_steps, initial=state["step"], desc=description,
                    unit="шаг", mininterval=1, dynamic_ncols=True)
    running = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "batches": 0}
    stop = False
    while state["step"] < total_steps and not stop:
        batches = epoch_batches(len(encoded["train"]), config.batch_size,
                                config.seed, state["epoch"])
        offset = state["next_batch"]
        # генератор загрузчика не сдвигает случайные маски модели при продолжении
        loader = DataLoader(encoded["train"], batch_sampler=batches[offset:],
                            collate_fn=NERCollator(tokenizer, teacher_logits), num_workers=0,
                            generator=torch.Generator().manual_seed(config.seed))
        for batch_index, batch in enumerate(loader, start=offset):
            # последняя группа накопления может быть короче, её масштабируем по фактическому размеру
            group_start = (batch_index // config.accumulation_steps) * config.accumulation_steps
            group_size = min(config.accumulation_steps, len(batches) - group_start)
            labels = batch["labels"].to(device)
            teacher = batch.get("teacher_logits")
            if teacher is not None:
                teacher = teacher.to(device)
            with autocast_context(device, config.amp):
                logits = model(**model_inputs(batch, device)).logits
                loss, parts = distillation_loss(logits, labels, teacher,
                                                config.temperature, config.kd_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"бесконечное или неопределённое значение потерь на шаге {state['step']}")
            scaler.scale(loss / group_size).backward()
            running["loss"] += float(loss.detach())
            for key in ("ce", "kd"):
                running[key] += float(parts[key])
            running["batches"] += 1
            boundary = (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(batches)
            if not boundary:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            updated = scaler.get_scale() >= old_scale
            # при переполнении градиентов шаг пропускается вместе с изменением скорости обучения
            if updated:
                scheduler.step()
                state["step"] += 1
                progress.update(1)
            state["next_batch"] = batch_index + 1
            if state["next_batch"] == len(batches):
                state["epoch"] += 1
                state["next_batch"] = 0
            if updated and state["step"] % config.log_every == 0:
                record = {key: running[key] / running["batches"] for key in ("loss", "ce", "kd")}
                record.update(step=state["step"], split="train", lr=scheduler.get_last_lr()[0])
                history.append(record)
                progress.set_postfix({"потери": f"{record['loss']:.3f}"}, refresh=False)
                running = dict(loss=0.0, ce=0.0, kd=0.0, batches=0)
            should_eval = updated and (state["step"] % config.eval_every == 0 or state["step"] >= total_steps)
            if should_eval:
                validate()
                if config.patience and state["bad_evals"] >= config.patience:
                    state["stop_reason"] = "early_stopping_on_validation"
                    stop = True
                save_state()
            if time_budget_hours is not None and time.monotonic() - start >= time_budget_hours * 3600:
                # по лимиту времени сохраняем прогресс и приостанавливаем обучение
                state["status"] = "paused"
                if not (output / "best.pt").exists():
                    validate()
                save_state()
                progress.close()
                return state
            if state["step"] >= total_steps or stop:
                stop = True
                break
    progress.close()
    state["status"] = "complete"
    state.setdefault("stop_reason", "training_budget_completed")
    save_state()
    return state
