from __future__ import annotations

import hashlib
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def choose_device(name: str = "auto") -> torch.device:
    return torch.device("cuda:0" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def autocast_context(device: torch.device, enabled: bool = True):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    def numpy_scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"не удалось записать {type(item).__name__} в JSON")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    default=numpy_scalar, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stable_hash(value) -> str:
    # одинаковые настройки дают одну подпись независимо от порядка ключей
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    # готовый файл заменяет предыдущий только после успешной записи
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def parameter_counts(model) -> dict:
    # общие параметры учитываются только один раз
    parameters = list(model.parameters())
    return {"parameters": sum(p.numel() for p in parameters),
            "trainable_parameters": sum(p.numel() for p in parameters if p.requires_grad)}


def model_inputs(batch: dict, device: torch.device) -> dict:
    return {name: batch[name].to(device) for name in
            ("input_ids", "attention_mask", "token_type_ids") if name in batch}

