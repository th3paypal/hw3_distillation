from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from .utils import read_json, stable_hash, write_json

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
IGNORE_INDEX = -100


def align_labels_with_tokens(labels, tokenized, label_names=LABELS, batch_index=0):
    word_ids = (tokenized.word_ids(batch_index=batch_index)
                if hasattr(tokenized, "word_ids") else tokenized)
    name_to_id = {name: i for i, name in enumerate(label_names)}
    previous = None
    aligned = []
    for word_id in word_ids:
        if word_id is None:
            # служебные токены не участвуют в подсчёте ошибки
            aligned.append(IGNORE_INDEX)
        else:
            label = int(labels[word_id])
            name = label_names[label]
            if word_id == previous and name.startswith("B-"):
                # продолжение слова сохраняет тип сущности, но теряет метку начала
                label = name_to_id["I-" + name[2:]]
            aligned.append(label)
        previous = word_id
    return aligned


def encode_sentence(words, labels, tokenizer, max_length: int, example_id: int):
    # делим длинное предложение по границам слов, сохраняя все слова
    if not words or len(words) != len(labels):
        raise ValueError("списки слов и меток должны быть непустыми и одинаковой длины")
    pieces = tokenizer(words, is_split_into_words=True, add_special_tokens=False,
                       truncation=False)
    lengths = Counter(pieces.word_ids())
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 1:
        raise ValueError("max_length должен оставлять место для слов после служебных токенов")
    if any(lengths[i] == 0 or lengths[i] > capacity for i in range(len(words))):
        raise ValueError("после токенизации слово пусто или не помещается в max_length")
    rows = []
    start = 0
    while start < len(words):
        end, length = start, 0
        while end < len(words) and length + lengths[end] <= capacity:
            length += lengths[end]
            end += 1
        encoded = tokenizer(words[start:end], is_split_into_words=True, truncation=False)
        ids = encoded.word_ids()
        first = [i is not None and (j == 0 or ids[j - 1] != i)
                 for j, i in enumerate(ids)]
        row = dict(encoded)
        # сохраняем номера слов из исходного предложения для сборки предсказаний
        row.update(labels=align_labels_with_tokens(labels[start:end], ids),
                   word_ids=[-1 if i is None else i + start for i in ids],
                   first_subword=first, example_id=example_id)
        assert len(row["input_ids"]) <= max_length
        rows.append(row)
        start = end
    return rows


def prepare_data(output: Path, dataset_name="ashaba1in/conll2003",
                 model_name="bert-base-cased", max_length=128,
                 dataset_revision=None, model_revision=None):
    output = Path(output)
    requested = dict(dataset_name=dataset_name, model_name=model_name, max_length=max_length,
                     dataset_revision=dataset_revision, model_revision=model_revision,
                     preprocessing_version=1)
    if (output / "metadata.json").exists():
        metadata = read_json(output / "metadata.json")
        if metadata["request"] != requested:
            raise ValueError("настройки данных изменились; укажите другую папку через --data-dir")
        return metadata
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"в папке {output} остались незавершённые данные; укажите другую через --data-dir")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, revision=model_revision)
    if not tokenizer.is_fast:
        raise ValueError("для word_ids нужен быстрый токенизатор")
    raw = load_dataset(dataset_name, revision=dataset_revision)
    required = {"train", "validation", "test"}
    if not required.issubset(raw):
        raise ValueError(f"нужны выборки {required}; доступны {list(raw)}")
    source_names = getattr(getattr(raw["train"].features["ner_tags"], "feature", None), "names", None)
    if source_names is None:
        source_names = LABELS
    if set(source_names) != set(LABELS):
        raise ValueError(f"неожиданный набор меток: {source_names}")
    # переводим номера меток датасета в единый порядок классов модели
    remap = {i: LABELS.index(name) for i, name in enumerate(source_names)}
    normalized, encoded = {}, {}
    for split in ("train", "validation", "test"):
        sentences, chunks = [], []
        description = {"train": "токенизация обучающей выборки",
                       "validation": "токенизация проверочной выборки",
                       "test": "токенизация тестовой выборки"}[split]
        for example_id, item in enumerate(tqdm(raw[split], desc=description,
                                               unit="текст", mininterval=1, dynamic_ncols=True)):
            tags = [remap[int(tag)] for tag in item["ner_tags"]]
            sentences.append({"tokens": item["tokens"], "ner_tags": tags})
            for row in encode_sentence(item["tokens"], tags, tokenizer, max_length, example_id):
                row["chunk_id"] = len(chunks)
                chunks.append(row)
        normalized[split] = Dataset.from_list(sentences)
        encoded[split] = Dataset.from_list(chunks)
    output.mkdir(parents=True, exist_ok=True)
    # исходные слова нужны для оценки, а токенизированные фрагменты для обучения
    DatasetDict(normalized).save_to_disk(str(output / "raw"))
    DatasetDict(encoded).save_to_disk(str(output / "encoded"))
    tokenizer.save_pretrained(output / "tokenizer")
    metadata = {"request": requested, "labels": LABELS,
                "source_label_names": source_names,
                "raw_fingerprints": {s: raw[s]._fingerprint for s in required},
                "encoded_fingerprints": {s: encoded[s]._fingerprint for s in required},
                "sentence_counts": {s: len(normalized[s]) for s in required},
                "chunk_counts": {s: len(encoded[s]) for s in required}}
    metadata["signature"] = stable_hash(metadata)
    write_json(output / "metadata.json", metadata)
    return metadata


def load_data(path: Path):
    path = Path(path)
    return (load_from_disk(str(path / "raw")), load_from_disk(str(path / "encoded")),
            AutoTokenizer.from_pretrained(path / "tokenizer", use_fast=True),
            read_json(path / "metadata.json"))


@dataclass
class NERCollator:
    tokenizer: object
    teacher_logits: object = None

    def __call__(self, rows):
        inputs = [{key: row[key] for key in ("input_ids", "attention_mask", "token_type_ids")
                   if key in row} for row in rows]
        batch = dict(self.tokenizer.pad(inputs, padding=True, pad_to_multiple_of=8,
                                        return_tensors="pt"))
        length = batch["input_ids"].shape[1]
        # метки и номера слов дополняем до той же длины, что и входные токены
        for key, fill, dtype in [("labels", IGNORE_INDEX, torch.long),
                                  ("word_ids", -1, torch.long),
                                  ("first_subword", False, torch.bool)]:
            batch[key] = torch.tensor([list(row[key]) + [fill] * (length - len(row[key]))
                                       for row in rows], dtype=dtype)
        for key in ("example_id", "chunk_id"):
            batch[key] = torch.tensor([row[key] for row in rows], dtype=torch.long)
        if self.teacher_logits is not None:
            # логиты учителя выбираем для тех же фрагментов, что попали в батч
            indices = [row["chunk_id"] for row in rows]
            batch["teacher_logits"] = torch.from_numpy(
                np.array(self.teacher_logits[indices, :length], copy=True))
        return batch

