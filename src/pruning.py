from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import NERCollator
from .models import FactorizedEmbedding, build_model
from .utils import model_inputs


def calibration_batches(encoded_train, tokenizer, samples=256, batch_size=8, seed=42):
    # берём фиксированную часть обучающей выборки для калибровки без меток
    indices = torch.randperm(len(encoded_train), generator=torch.Generator().manual_seed(seed))[:samples].tolist()
    subset = encoded_train.select(indices)
    return list(DataLoader(subset, batch_size=batch_size, shuffle=False,
                           collate_fn=NERCollator(tokenizer), num_workers=0))


@torch.inference_mode()
def layer_importance(model, batches, device):
    # слой важнее, если сильнее меняет направление векторов слов
    model.eval().to(device)
    totals = torch.zeros(len(model.bert.encoder.layer), dtype=torch.float64)
    count = 0
    for batch in tqdm(batches, desc="оценка важности слоёв", unit="батч",
                      mininterval=1, dynamic_ncols=True):
        hidden = model(**model_inputs(batch, device), output_hidden_states=True).hidden_states
        valid = batch["labels"].ne(-100).to(device)
        count += int(valid.sum())
        for layer in range(len(totals)):
            distance = 1 - F.cosine_similarity(hidden[layer].float(), hidden[layer + 1].float(), dim=-1)
            totals[layer] += distance[valid].double().sum().cpu()
    if not count:
        raise ValueError("в калибровочной выборке нет токенов слов")
    return (totals / count).tolist()


@torch.no_grad()
def initialize_by_depth_pruning(teacher, teacher_spec, scores, keep_layers=2, rank=64):
    # удаляем лишние слои и факторизуем эмбеддинги, сохраняя порядок слоёв
    if not 1 <= keep_layers <= len(scores):
        raise ValueError("недопустимое число оставляемых слоёв")
    selected = sorted(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep_layers])
    spec = copy.deepcopy(teacher_spec)
    spec.bert["num_hidden_layers"] = keep_layers
    spec.embedding_rank, spec.shared_groups, spec.lora_rank = rank, None, 0
    target = build_model(spec)
    for name in ("position_embeddings", "token_type_embeddings", "LayerNorm"):
        getattr(target.bert.embeddings, name).load_state_dict(
            getattr(teacher.bert.embeddings, name).state_dict())
    target.bert.embeddings.word_embeddings = FactorizedEmbedding.from_embedding(
        teacher.bert.embeddings.word_embeddings, rank).cpu()
    for new_layer, original in zip(target.bert.encoder.layer, selected):
        new_layer.load_state_dict(teacher.bert.encoder.layer[original].state_dict())
    target.classifier.load_state_dict(teacher.classifier.state_dict())
    return target, spec, selected


@torch.no_grad()
def wanda_prune_weight(weight, squared_inputs, sparsity=0.5):
    if not 0 <= sparsity < 1:
        raise ValueError("доля нулевых весов sparsity должна быть в диапазоне [0, 1)")
    result = weight.detach().float().clone()
    # вес важен, если он велик сам или умножается на часто активный вход
    score = result.abs() * squared_inputs.clamp_min(0).sqrt().unsqueeze(0)
    count = int(result.shape[1] * sparsity)
    if count:
        indices = score.argsort(dim=1, stable=True)[:, :count]
        result.scatter_(1, indices, 0.0)
    return result.to(weight.dtype)


@torch.no_grad()
def sparsegpt_prune_weight(weight, hessian, sparsity=0.5, block_size=128, damp_ratio=0.01):
    # матрица скалярных произведений входов приближает гессиан ошибки слоя
    # её обратная матрица помогает выбрать веса и компенсировать ошибку обнуления
    if not 0 <= sparsity < 1 or block_size < 1 or damp_ratio <= 0:
        raise ValueError("недопустимая доля нулей, размер блока или коэффициент регуляризации")
    w = weight.detach().float().clone()
    if sparsity == 0:
        return w.to(weight.dtype)
    h = hessian.detach().float().clone()
    if h.shape != (w.shape[1], w.shape[1]):
        raise ValueError("размер гессиана не соответствует числу входов слоя")
    dead = h.diag() <= 1e-12
    # неактивные входы не влияют на выход, а их нули мешают обращению матрицы
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    damping = max(float(h.diag().mean()) * damp_ratio, 1e-6)
    identity = torch.eye(h.shape[0], device=h.device)
    # усиливаем регуляризацию, если матрица плохо обусловлена
    for _ in range(6):
        chol, info = torch.linalg.cholesky_ex(h + damping * identity)
        if int(info.max()) == 0:
            inverse = torch.cholesky_inverse(chol)
            inverse_chol, inverse_info = torch.linalg.cholesky_ex(inverse, upper=True)
            if int(inverse_info.max()) == 0:
                break
        damping *= 10
    else:
        raise RuntimeError("гессиан не стал положительно определённым после регуляризации")
    for start in range(0, w.shape[1], block_size):
        end = min(start + block_size, w.shape[1])
        block = w[:, start:end].clone()
        upper = inverse_chol[start:end, start:end]
        scores = block.square() / upper.diag().square().unsqueeze(0)
        n_prune = int(scores.numel() * sparsity)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if n_prune:
            selected = torch.argsort(scores.flatten(), stable=True)[:n_prune]
            mask.view(-1)[selected] = True
        output = torch.zeros_like(block)
        errors = torch.zeros_like(block)
        for column in range(end - start):
            current = block[:, column].clone()
            quantized = current.masked_fill(mask[:, column], 0.0)
            output[:, column] = quantized
            error = (current - quantized) / upper[column, column]
            # переносим ошибку обнулённого веса на ещё не обработанные столбцы
            block[:, column:] -= error.unsqueeze(1) * upper[column, column:].unsqueeze(0)
            errors[:, column] = error
        w[:, start:end] = output
        w[:, end:] -= errors @ inverse_chol[start:end, end:]
    if not torch.isfinite(w).all():
        raise FloatingPointError("после SparseGPT появились бесконечные или неопределённые веса")
    return w.to(weight.dtype)


@torch.inference_mode()
def prune_transformer(model, batches, device, method="wanda", sparsity=0.5,
                      block_size=128, damp_ratio=0.01):
    # эмбеддинги, нормализацию, смещения и классификатор не меняем и не дообучаем
    if method not in {"wanda", "sparsegpt"}:
        raise ValueError(method)
    layers = model.bert.encoder.layer
    # требуем раздельные веса слоёв, чтобы разные маски не применялись к одним весам
    if any(not isinstance(layer.attention.self.query, nn.Linear) for layer in layers):
        raise ValueError("для этого эксперимента нужен ученик без общих весов и адаптеров LoRA")
    pointers = [layer.attention.self.query.weight.data_ptr() for layer in layers]
    if len(set(pointers)) != len(pointers):
        raise ValueError("для этого эксперимента нужен ученик без общих весов между слоями")
    model.eval().to(device)
    details = []
    for layer_index, layer in enumerate(tqdm(layers, desc=f"прореживание весов ({method})",
                                             unit="слой", mininterval=1, dynamic_ncols=True)):
        modules = {name: module for name, module in layer.named_modules() if isinstance(module, nn.Linear)}
        statistics = {name: torch.zeros((m.in_features, m.in_features) if method == "sparsegpt"
                                        else (m.in_features,), device=device, dtype=torch.float32)
                      for name, m in modules.items()}
        token_count = {name: 0 for name in modules}
        context = {}

        def hook(name):
            def collect(module, args, output):
                x = args[0].detach().float()[context["valid"]]
                token_count[name] += len(x)
                # wanda нужны масштабы входов, а sparsegpt также учитывает связи между ними
                statistics[name].add_(x.T @ x if method == "sparsegpt" else x.square().sum(0))
            return collect

        handles = [module.register_forward_hook(hook(name)) for name, module in modules.items()]
        try:
            # следующий слой калибруем уже с учётом прореженных предыдущих слоёв
            for batch in tqdm(batches, desc=f"калибровка слоя {layer_index + 1}",
                              unit="батч", leave=False, mininterval=1, dynamic_ncols=True):
                inputs = model_inputs(batch, device)
                context["valid"] = inputs["attention_mask"].bool()
                model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        for name, module in tqdm(modules.items(), desc="прореживание матриц", unit="матрица",
                                  leave=False, mininterval=1, dynamic_ncols=True):
            if not token_count[name]:
                raise ValueError("калибровочная выборка пуста")
            statistic = statistics[name] / token_count[name]
            old_weight = module.weight.detach().clone()
            if method == "wanda":
                pruned = wanda_prune_weight(old_weight, statistic, sparsity)
            else:
                pruned = sparsegpt_prune_weight(old_weight, statistic, sparsity, block_size, damp_ratio)
            module.weight.copy_(pruned)
            details.append({"layer": layer_index, "matrix": name,
                            "weights": module.weight.numel(),
                            "zeros": int(module.weight.eq(0).sum()),
                            "calibration_tokens": token_count[name]})
        del statistics
    return details
