from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import LABELS, NERCollator
from .utils import autocast_context, model_inputs, parameter_counts, write_json


@torch.inference_mode()
def predict_word_labels(model, encoded, raw, tokenizer, device, batch_size=32, amp=True,
                        description="оценка качества"):
    loader = DataLoader(encoded, batch_size=batch_size, shuffle=False,
                        collate_fn=NERCollator(tokenizer), num_workers=0,
                        generator=torch.Generator().manual_seed(0))
    was_training = model.training
    model.eval()
    predictions = [[None] * len(row["tokens"]) for row in raw]
    for batch in tqdm(loader, desc=description, unit="батч", leave=False,
                      mininterval=1, dynamic_ncols=True):
        with autocast_context(device, amp):
            logits = model(**model_inputs(batch, device)).logits
        labels = logits.argmax(-1).cpu()
        # метрика работает со словами, поэтому используем только их первые части
        for row, sentence_id in enumerate(batch["example_id"].tolist()):
            for position in batch["first_subword"][row].nonzero().flatten().tolist():
                word_id = int(batch["word_ids"][row, position])
                if predictions[sentence_id][word_id] is not None:
                    raise AssertionError("для одного слова предсказание получено дважды")
                predictions[sentence_id][word_id] = LABELS[int(labels[row, position])]
    if any(label is None for row in predictions for label in row):
        raise AssertionError("при токенизации или оценке потерялись слова")
    references = [[LABELS[i] for i in row["ner_tags"]] for row in raw]
    # после промежуточной проверки возвращаем режим обучения с отключением нейронов
    model.train(was_training)
    return references, predictions


def evaluate(model, encoded, raw, tokenizer, device, batch_size=32,
             amp=True, detailed=False, description="оценка качества"):
    true, predicted = predict_word_labels(model, encoded, raw, tokenizer, device,
                                          batch_size, amp, description)
    metrics = {"f1": float(f1_score(true, predicted)),
               "precision": float(precision_score(true, predicted, zero_division=0)),
               "recall": float(recall_score(true, predicted, zero_division=0)),
               "sentences": len(true), "words": sum(map(len, true))}
    if detailed:
        metrics["per_entity"] = classification_report(
            true, predicted, output_dict=True, zero_division=0)
    return metrics, true, predicted


@torch.inference_mode()
def benchmark(model, encoded, tokenizer, device, batch_size=16, repetitions=30, amp=True):
    # замеряем прямой проход после прогрева на одинаковом батче для всех моделей
    batch = NERCollator(tokenizer)([encoded[i] for i in range(min(batch_size, len(encoded)))])
    inputs = model_inputs(batch, device)
    was_training = model.training
    model.eval()
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(5):
        with autocast_context(device, amp):
            model(**inputs)
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in tqdm(range(repetitions), desc="замер скорости", unit="повтор",
                  leave=False, mininterval=1, dynamic_ncols=True):
        sync()
        start = time.perf_counter()
        with autocast_context(device, amp):
            model(**inputs)
        sync()
        # время обновления индикатора не входит в измеренный прямой проход
        durations.append(time.perf_counter() - start)
    model.train(was_training)
    median = float(np.median(durations))
    return {"latency_ms": median * 1000,
            "sentences_per_second": len(batch["example_id"]) / median,
            "batch_size": len(batch["example_id"]),
            "padded_sequence_length": int(inputs["input_ids"].shape[1]),
            "amp": bool(amp and device.type == "cuda"),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU/MPS",
            "peak_cuda_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}


def save_evaluation(model, encoded, raw, tokenizer, device, checkpoint: Path,
                    output: Path, split="test", batch_size=32):
    split_name = {"validation": "проверочной выборке", "test": "тестовой выборке"}.get(split, split)
    metrics, true, predicted = evaluate(model, encoded, raw, tokenizer, device, batch_size,
                                        detailed=True, description=f"оценка на {split_name}")
    metrics.update(parameter_counts(model))
    metrics["checkpoint_mb"] = Path(checkpoint).stat().st_size / 2**20
    metrics["benchmark"] = benchmark(model, encoded, tokenizer, device)
    write_json(Path(output) / f"{split}_metrics.json", metrics)
    errors = []
    for i, (gold, pred) in enumerate(zip(true, predicted)):
        if gold != pred:
            errors.append({"example_id": i, "tokens": raw[i]["tokens"],
                           "gold": gold, "predicted": pred})
    write_json(Path(output) / f"{split}_errors.json", errors[:100])
    return metrics

