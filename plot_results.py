import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


MODELS = [
    "teacher", "factorized", "student_kd", "pruned_init",
    "shared", "shared_lora", "wanda", "sparsegpt",
]
TRAINED = MODELS[:6]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_results(runs):
    results = {}
    for name in MODELS:
        folder = runs / name
        result = {
            "metrics": read_json(folder / "test_metrics.json"),
            "run": read_json(folder / "run.json"),
            "status": read_json(folder / "status.json"),
        }
        if result["status"]["status"] != "complete":
            raise ValueError(f"эксперимент {name} ещё не завершён")
        extra = "history" if name in TRAINED else "pruning"
        result[extra] = read_json(folder / f"{extra}.json")
        results[name] = result
    return results


def save(fig, output, name, title, note=""):
    fig.suptitle(title)
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06 if note else 0, 1, 0.96))
    fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"сохранён {name}.png")


def bars(ax, names, values, title, xlabel):
    columns = ax.barh(names, values)
    ax.invert_yaxis()
    ax.set(title=title, xlabel=xlabel)
    ax.margins(x=0.2)
    ax.bar_label(columns, fmt="%.2f", padding=3, fontsize=9)


def plot_learning(results, output):
    for split, metric, title, filename in (
        ("train", "loss", "потери на обучении", "01_training_loss"),
        ("validation", "f1", "F1 на валидации", "02_validation_f1"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, name in zip(axes.flat, TRAINED):
            rows = [r for r in results[name]["history"] if r["split"] == split]
            steps = [r["step"] for r in rows]
            ax.plot(steps, [r[metric] for r in rows], label=metric)
            ax.set(title=name, xlabel="шаг", ylabel=metric)
            ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
            if split == "train":
                ax.set_yscale("log")
                weight = results[name]["run"]["config"].get("kd_weight", 0)
                if weight:
                    # вклад дистилляции учитывает её вес в общей функции потерь
                    ax.plot(steps, [r["ce"] for r in rows], label="кросс-энтропия")
                    ax.plot(steps, [weight * r["kd"] for r in rows], label="дистилляция")
                    ax.legend(fontsize=8)
            else:
                ax.set_ylim(0, 1)
        save(fig, output, filename, title)


def plot_quality(results, output):
    fig, ax = plt.subplots(figsize=(11, 7))
    for index, metric in enumerate(("precision", "recall", "f1")):
        positions = [i + (index - 1) * 0.25 for i in range(len(MODELS))]
        values = [results[name]["metrics"][metric] for name in MODELS]
        columns = ax.barh(positions, values, height=0.24, label=metric)
        ax.bar_label(columns, fmt="%.3f", padding=3, fontsize=8)
    ax.set_yticks(range(len(MODELS)), labels=MODELS)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.1), xlabel="значение метрики")
    ax.legend(loc="lower right")
    save(fig, output, "03_test_quality", "качество на тестовой выборке")


def plot_resources(results, output, benchmark_note):
    metrics = [results[name]["metrics"] for name in MODELS]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ([m["parameters"] / 1e6 for m in metrics], "число параметров", "млн"),
        ([m["checkpoint_mb"] for m in metrics], "размер весов", "МиБ"),
        ([m["benchmark"]["latency_ms"] for m in metrics], "медианное время прохода", "мс на батч"),
        ([m["benchmark"]["peak_cuda_mb"] for m in metrics], "пиковая память PyTorch", "МиБ"),
    ]
    for ax, (values, title, unit) in zip(axes.flat, panels):
        bars(ax, MODELS, values, title, unit)
    save(fig, output, "04_resources", "размер и вычислительные затраты", benchmark_note)


def plot_tradeoffs(results, output, benchmark_note):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, name in enumerate(TRAINED):
        metric = results[name]["metrics"]
        xs = (metric["parameters"] / 1e6, metric["benchmark"]["latency_ms"])
        for ax, x in zip(axes, xs):
            ax.scatter(x, metric["f1"], color=f"C{index}", label=name)
    for ax, xlabel in zip(axes, ("млн параметров", "медианное время батча, мс")):
        ax.set(xlabel=xlabel, ylabel="F1 на test", ylim=(0, 1), xlim=(0, None))
        ax.legend(fontsize=8, loc="lower right")
    axes[0].axvline(20, color="gray", linestyle="--", label="лимит ученика")
    axes[0].legend(fontsize=8, loc="lower right")
    save(fig, output, "05_quality_tradeoffs", "качество и затраты", benchmark_note)


def plot_entities(results, output):
    entities = ["PER", "ORG", "LOC", "MISC"]
    values = [[results[name]["metrics"]["per_entity"][e]["f1-score"]
               for e in entities] for name in MODELS]
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(entities)), labels=entities)
    ax.set_yticks(range(len(MODELS)), labels=MODELS)
    for row, scores in enumerate(values):
        for col, value in enumerate(scores):
            ax.text(col, row, f"{value:.3f}", ha="center", va="center",
                    color="white" if value > 0.6 else "black")
    fig.colorbar(heatmap, ax=ax, label="F1")
    save(fig, output, "06_entity_f1", "F1 по типам сущностей на test")


def plot_pruning(results, output):
    names = ["wanda", "sparsegpt"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, name in enumerate(names):
        result = results[name]
        pruning = result["pruning"]
        source = result["run"]["pruning"]["source"]
        deltas = [
            pruning["validation_after"]["f1"] - pruning["validation_before"]["f1"],
            result["metrics"]["f1"] - results[source]["metrics"]["f1"],
        ]
        matrices = pruning["matrices"]
        # в обработанных матрицах и во всей модели разные доли нулей
        zeros = [
            sum(m["zeros"] for m in matrices) / sum(m["weights"] for m in matrices),
            pruning["zero_parameters"] / result["metrics"]["parameters"],
        ]
        for ax, values, labels in (
            (axes[0], deltas, ("validation", "test")),
            (axes[1], zeros, ("обработанные матрицы", "вся модель")),
        ):
            for j, (value, label) in enumerate(zip(values, labels)):
                column = ax.bar(i + (j - 0.5) * 0.35, 100 * value, width=0.35,
                                color=f"C{j}", label=label if i == 0 else None)
                ax.bar_label(column, fmt="%.3f" if ax is axes[0] else "%.1f", padding=3)
        column = axes[2].bar(i, pruning["seconds"])
        axes[2].bar_label(column, fmt="%.2f", padding=3)
    for ax, title, unit in zip(axes, ("изменение F1", "доля нулевых весов", "время прореживания"),
                               ("процентные пункты", "%", "секунды")):
        ax.set_xticks(range(len(names)), labels=names)
        ax.set(title=title, ylabel=unit)
        ax.margins(y=0.25)
    axes[0].axhline(0, color="gray", linewidth=1)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].legend(loc="upper center", fontsize=8)
    save(fig, output, "07_pruning", "Wanda и SparseGPT",
         "изменение качества относительно ученика; время без оценки качества")


def plot_layers(results, output):
    metadata = results["pruned_init"]["run"]["metadata"]
    scores = metadata["layer_importance"]
    selected = metadata["selected_teacher_layers_zero_based"]
    fig, ax = plt.subplots(figsize=(10, 4))
    # на графике нумерация слоёв начинается с единицы
    ax.bar(range(1, len(scores) + 1), scores,
           color=["C0" if i in selected else "lightgray" for i in range(len(scores))])
    ax.set_xticks(range(1, len(scores) + 1))
    ax.set(xlabel="номер слоя учителя", ylabel="среднее косинусное расстояние")
    save(fig, output, "08_layer_importance", "важность слоёв учителя",
         "синим отмечены слои, сохранённые в ученике")


def main():
    parser = argparse.ArgumentParser(description="графики по результатам hw3")
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.results_dir.expanduser().resolve()
    runs = root if root.name == "runs" else root / "runs"
    output = args.output_dir.expanduser().resolve() if args.output_dir else runs.parent / "figures"
    results = load_results(runs)
    # скорость сравнивается при одинаковом батче, длине и режиме вычислений
    keys = ("device_name", "batch_size", "padded_sequence_length", "amp")
    settings = {tuple(r["metrics"]["benchmark"][key] for key in keys) for r in results.values()}
    if len(settings) != 1:
        raise ValueError("условия замеров скорости различаются")
    device, batch, length, amp = settings.pop()
    note = f"{device}; батч {batch}; длина {length}; AMP {'включён' if amp else 'выключен'}"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    plot_learning(results, output)
    plot_quality(results, output)
    plot_resources(results, output, note)
    plot_tradeoffs(results, output, note)
    plot_entities(results, output)
    plot_pruning(results, output)
    plot_layers(results, output)
    fig, ax = plt.subplots(figsize=(10, 5))
    minutes = [results[name]["status"]["elapsed_seconds"] / 60 for name in TRAINED]
    bars(ax, TRAINED, minutes, "", "минуты")
    save(fig, output, "09_training_time", "время обучения с валидацией",
         "без инициализации и кэша; этапы шли параллельно, сумма времён не равна времени прогона")
    print(f"готово: 9 графиков png в {output}")


if __name__ == "__main__":
    main()
