from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import BertConfig, BertForTokenClassification

from .data import LABELS
from .utils import atomic_torch_save, parameter_counts


class FactorizedEmbedding(nn.Module):
    # храним эмбеддинги как W[V,H] = A[V,E] B[E,H], без полной матрицы V x H

    def __init__(self, vocab_size, hidden_size, rank, padding_idx=0):
        super().__init__()
        if not 0 < rank <= min(vocab_size, hidden_size):
            raise ValueError("недопустимый ранг факторизации эмбеддингов")
        self.num_embeddings, self.embedding_dim = vocab_size, hidden_size
        self.padding_idx, self.rank = padding_idx, rank
        self.table = nn.Embedding(vocab_size, rank, padding_idx=padding_idx)
        self.projection = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.table.weight, std=0.02)
        nn.init.normal_(self.projection.weight, std=1 / math.sqrt(rank))
        if padding_idx is not None:
            with torch.no_grad():
                self.table.weight[padding_idx].zero_()

    def forward(self, input_ids):
        return self.projection(self.table(input_ids))

    @classmethod
    @torch.no_grad()
    def from_embedding(cls, original: nn.Embedding, rank: int):
        # приближённое разложение заменяет полное разложение большой матрицы
        weight = original.weight.detach().float()
        q = min(rank + 16, min(weight.shape))
        u, s, v = torch.svd_lowrank(weight, q=q, niter=4)
        # распределяем сингулярные значения между двумя матрицами через квадратный корень
        root_s = s[:rank].clamp_min(0).sqrt()
        result = cls(*weight.shape, rank, original.padding_idx).to(weight.device)
        result.table.weight.copy_(u[:, :rank] * root_s)
        result.projection.weight.copy_(v[:, :rank] * root_s)
        # сохраняем исходный вектор добивки: лишние позиции скрывает маска
        return result.to(dtype=original.weight.dtype)


class LoRALinear(nn.Module):
    # общая линейная проекция с отдельной низкоранговой поправкой для каждого слоя

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.in_features, self.out_features = base.in_features, base.out_features
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = (alpha if alpha is not None else rank) / rank
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        # нулевая вторая матрица делает начальную поправку адаптера равной нулю
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.b(self.a(x))


@dataclass
class ModelSpec:
    bert: dict
    embedding_rank: int | None = None
    shared_groups: int | None = None
    lora_rank: int = 0


def tie_encoder_layers(model, groups: int, lora_rank: int = 0):
    layers = model.bert.encoder.layer
    if not 1 <= groups <= len(layers):
        raise ValueError("недопустимое число групп слоёв с общими весами")
    # слои остаются разными модулями, но ссылаются на общие параметры
    bases = list(layers[:groups])
    copies = []
    for depth in range(len(layers)):
        base = bases[depth % groups]
        layer = copy.deepcopy(base)
        for name, parameter in base.named_parameters():
            path, leaf = name.rsplit(".", 1)
            setattr(layer.get_submodule(path), leaf, parameter)
        if lora_rank:
            # для каждого слоя создаём собственные адаптеры запросов и значений
            for name in ("query", "value"):
                attention = layer.attention.self
                setattr(attention, name, LoRALinear(getattr(attention, name), lora_rank))
        copies.append(layer)
    model.bert.encoder.layer = nn.ModuleList(copies)


def build_model(spec: ModelSpec):
    config = BertConfig.from_dict(spec.bert)
    config._attn_implementation = "sdpa"
    model = BertForTokenClassification(config)
    if spec.embedding_rank:
        model.bert.embeddings.word_embeddings = FactorizedEmbedding(
            config.vocab_size, config.hidden_size, spec.embedding_rank, config.pad_token_id)
    if spec.shared_groups:
        tie_encoder_layers(model, spec.shared_groups, spec.lora_rank)
    elif spec.lora_rank:
        raise ValueError("в этом проекте адаптеры LoRA по глубине требуют слоёв с общими весами")
    return model


def student_spec(vocab_size=28996, pad_token_id=0, hidden_size=384,
                 layers=6, heads=6, intermediate_size=1536, rank=64):
    config = BertConfig(vocab_size=vocab_size, hidden_size=hidden_size,
                        num_hidden_layers=layers, num_attention_heads=heads,
                        intermediate_size=intermediate_size, max_position_embeddings=512,
                        num_labels=len(LABELS), pad_token_id=pad_token_id,
                        id2label=dict(enumerate(LABELS)),
                        label2id={name: i for i, name in enumerate(LABELS)})
    return ModelSpec(config.to_dict(), embedding_rank=rank)


def make_teacher(model_name="bert-base-cased", revision=None):
    model = BertForTokenClassification.from_pretrained(
        model_name, revision=revision, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={n: i for i, n in enumerate(LABELS)},
        attn_implementation="sdpa")
    return model, ModelSpec(model.config.to_dict())


def factorize_model(model, spec, rank=64):
    result = copy.deepcopy(model)
    result.bert.embeddings.word_embeddings = FactorizedEmbedding.from_embedding(
        model.bert.embeddings.word_embeddings, rank)
    new_spec = copy.deepcopy(spec)
    new_spec.embedding_rank = rank
    return result, new_spec


@torch.no_grad()
def shared_from_student(source, spec, groups=2, lora_rank=0):
    # усредняем слои с одинаковым шагом по глубине и связываем веса внутри групп
    new_spec = copy.deepcopy(spec)
    new_spec.shared_groups, new_spec.lora_rank = groups, lora_rank
    target = build_model(new_spec)
    target.bert.embeddings.load_state_dict(source.bert.embeddings.state_dict())
    target.classifier.load_state_dict(source.classifier.state_dict())
    source_layers = source.bert.encoder.layer
    for depth, layer in enumerate(target.bert.encoder.layer):
        indices = list(range(depth % groups, len(source_layers), groups))
        states = [source_layers[i].state_dict() for i in indices]
        averaged = {name: torch.stack([state[name].float() for state in states]).mean(0)
                    for name in states[0]}
        if lora_rank:
            # загружаем усреднённые веса в основу адаптера, оставляя его поправку нулевой
            averaged = {name.replace("attention.self.query.", "attention.self.query.base.")
                        .replace("attention.self.value.", "attention.self.value.base."): tensor
                        for name, tensor in averaged.items()}
            missing, unexpected = layer.load_state_dict(averaged, strict=False)
            assert not unexpected and all(".a." in n or ".b." in n for n in missing)
        else:
            layer.load_state_dict(averaged)
    return target, new_spec


def assert_student_budget(model, limit=20_000_000):
    count = parameter_counts(model)["parameters"]
    if count > limit:
        raise ValueError(f"у ученика {count:,} параметров при лимите {limit:,}")
    return count


def save_model(model, spec, path: Path, **metadata):
    atomic_torch_save({"spec": asdict(spec), "model": model.state_dict(),
                       "metadata": metadata}, path)


def load_model(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    spec = ModelSpec(**payload["spec"])
    model = build_model(spec)
    model.load_state_dict(payload["model"], strict=True)
    return model, spec, payload.get("metadata", {})
