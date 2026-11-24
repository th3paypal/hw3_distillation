from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import LABELS, NERCollator
from .utils import autocast_context, file_hash, model_inputs, read_json, write_json


def cache_signature(checkpoint, data_metadata):
    # связываем кэш с конкретными весами учителя и подготовленными данными
    return {"checkpoint_sha256": file_hash(Path(checkpoint)),
            "data_signature": data_metadata["signature"],
            "labels": LABELS, "dtype": "float16", "split": "train"}


@torch.inference_mode()
def cache_teacher_logits(model, encoded, tokenizer, device, checkpoint, data_metadata,
                         output: Path, batch_size=32):
    output = Path(output)
    signature = cache_signature(checkpoint, data_metadata)
    if (output / "metadata.json").exists():
        load_logit_cache(output, signature)
        return
    output.mkdir(parents=True, exist_ok=True)
    length = ((data_metadata["request"]["max_length"] + 7) // 8) * 8
    temporary = output / "logits.tmp.npy"
    # записываем логиты на диск по батчам, не удерживая весь массив в памяти
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16,
                                    shape=(len(encoded), length, len(LABELS)))
    array[:] = 0
    loader = DataLoader(encoded, batch_size=batch_size, shuffle=False,
                        collate_fn=NERCollator(tokenizer), num_workers=0)
    model.eval()
    model.requires_grad_(False)
    for batch in tqdm(loader, desc="логиты учителя на обучающей выборке",
                      unit="батч", mininterval=1, dynamic_ncols=True):
        with autocast_context(device):
            logits = model(**model_inputs(batch, device)).logits
        # индекс фрагмента сохраняет соответствие логитов обучающим примерам
        array[batch["chunk_id"].numpy(), :logits.shape[1]] = logits.float().cpu().numpy().astype(np.float16)
    array.flush()
    del array
    # помечаем кэш готовым только после записи всех батчей
    temporary.replace(output / "logits.npy")
    write_json(output / "metadata.json", {**signature, "shape": [len(encoded), length, len(LABELS)]})


def load_logit_cache(path: Path, expected: dict):
    path = Path(path)
    metadata = read_json(path / "metadata.json")
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("кэш учителя устарел: изменились веса, метки или токенизация")
    logits = np.load(path / "logits.npy", mmap_mode="r")
    if list(logits.shape) != metadata["shape"] or logits.dtype != np.float16:
        raise ValueError("неверная форма или тип данных в кэше логитов учителя")
    return logits

