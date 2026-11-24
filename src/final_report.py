import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plots import COLORS, NEURAL_LABELS, _finish, plot_neural_comparison, plot_training_curves

NEURAL_DIRS = [
    "full_cnn",
    "full_mlp",
    "full_gru",
    "full_gru_long",
    "full_transformer",
    "full_dnabert2",
    "full_dnabert2_scratch",
    "full_dnabert2_2ep",
    "full_dnabert2_scratch_warmup_only",
]
CONDITIONS = [
    ("test_pooled", "250 bp test"),
    ("test_1_len150", "150 bp"),
    ("test_1_len100", "100 bp"),
    ("test_1_letters_shuffled", "letters shuffled"),
    ("bacpacs_test_1", "BacPaCS"),
]
LABELS = {**NEURAL_LABELS, "dnabert2_2ep": "DNABERT-2, 2-epoch cap (first run)", "dnabert2_scratch_warmup_only": "DNABERT-2 random init, warmup only", "gru_long": "BiGRU, longer training", "simple": "Best simple model"}


def _selected_tag(summary: pd.DataFrame) -> str:
    row = summary[summary["group"] == "model"].sort_values("val_auc_mean").iloc[-1]
    k = "None" if pd.isna(row["k"]) else str(int(row["k"]))
    return "model_" + row["model"] + "_" + row["representation"] + "_" + k


def collect(results: Path) -> pd.DataFrame:
    """One row per model and evaluation set: read AUC and genome-level AUC (seed means)."""
    rows = []
    sources = {name[5:]: results / name for name in NEURAL_DIRS if (results / name / "results_summary.csv").exists() and (results / name / "genome").exists()}
    sources["simple"] = results / "simple_models"
    for name, folder in sources.items():
        summary = pd.read_csv(folder / "results_summary.csv")
        tag = _selected_tag(summary)
        genome = pd.read_csv(folder / "genome" / "genome_summary.csv")
        genome = genome[genome["tag"] == tag]
        mean = genome.groupby("split")[
            ["read_auc", "genome_weighted_auc", "genome_auc", "genome_balanced_accuracy"]
        ].mean()
        for split, values in mean.iterrows():
            rows.append({"model": name, "tag": tag, "split": split, **values.to_dict()})
    return pd.DataFrame(rows)


def baselines(results: Path) -> pd.DataFrame:
    summary = pd.read_csv(results / "simple_models" / "results_summary.csv")
    return summary[summary["group"] == "baseline"][
        ["representation", "val_auc_mean", "test_1_auc_mean"]
    ]


def figure_comparison(table: pd.DataFrame, base: pd.DataFrame, path: Path):
    """Read AUC and genome AUC on the pooled test set for every model, with baselines."""
    part = table[table["split"] == "test_pooled"].sort_values("read_auc")
    figure, axes = plt.subplots(1, 2, figsize=(10, 0.5 * len(part) + 1.8), sharey=True, layout="constrained")
    y = np.arange(len(part))
    for axis, column, title in zip(axes, ["read_auc", "genome_auc"], ["Read-level AUC", "Genome-level AUC (mean read score)"]):
        axis.barh(y, part[column] - 0.5, left=0.5, color=COLORS[0], height=0.6)
        for position, value in zip(y, part[column]):
            axis.text(0.995, position, format(value, ".3f"), ha="right", va="center", fontsize=8)
        axis.axvline(0.5, color="0.6", linestyle="--", linewidth=1)
        axis.set(title=title, xlim=(0.5, 1.0))
    for _, row in base.iterrows():
        if row["representation"] in ("composition_k2", "gc_fraction"):
            axes[0].axvline(row["test_1_auc_mean"], color=COLORS[1], linestyle=":", linewidth=1)
    axes[0].set_yticks(y, [LABELS.get(m, m) for m in part["model"]])
    axes[0].text(0.505, len(part) - 0.45, "orange dotted: GC / letter-pair baselines", fontsize=7, color=COLORS[1])
    _finish(figure, axes, path, categorical=True)


def figure_robustness(table: pd.DataFrame, path: Path):
    """AUC of each model on the test set, shorter reads, shuffled letters and BacPaCS."""
    models = [m for m in table["model"].unique() if m != "dnabert2_scratch" or True]
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, layout="constrained")
    for axis, column, title in zip(axes, ["read_auc", "genome_auc"], ["Read-level AUC", "Genome-level AUC"]):
        for number, model in enumerate(models):
            part = table[table["model"] == model].set_index("split")
            values = [part.loc[s, column] if s in part.index else np.nan for s, _ in CONDITIONS]
            axis.plot(range(len(CONDITIONS)), values, marker="o", markersize=4, linewidth=1.3,
                      color=plt.cm.tab10(number), label=LABELS.get(model, model))
        axis.axhline(0.5, color="0.6", linestyle="--", linewidth=1)
        axis.set_xticks(range(len(CONDITIONS)), [c for _, c in CONDITIONS], fontsize=8)
        axis.set(title=title, ylim=(0.45, 1.0))
    axes[0].set_ylabel("AUC")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    _finish(figure, axes, path, categorical=True)


def figure_aggregation(results: Path, table: pd.DataFrame, path: Path):
    """Genome AUC against the number of reads averaged per genome (pooled test_1)."""
    figure, axis = plt.subplots(figsize=(6, 3.8), layout="constrained")
    for number, model in enumerate(table["model"].unique()):
        folder = results / ("simple_models" if model == "simple" else "full_" + model)
        curves = pd.read_csv(folder / "genome" / "aggregation_curves.csv")
        curves = curves[(curves["split"] == "test_1") & (curves["tag"] == table[table["model"] == model]["tag"].iloc[0])]
        if curves.empty:
            continue
        axis.errorbar(curves["reads_per_genome"], curves["genome_auc_mean"], curves["genome_auc_std"],
                      marker="o", markersize=3, linewidth=1.2, color=plt.cm.tab10(number),
                      label=LABELS.get(model, model))
    axis.set(xscale="log", xlabel="Reads averaged per genome", ylabel="Genome-level AUC", ylim=(0.45, 1.0))
    axis.legend(frameon=False, fontsize=7)
    _finish(figure, axis, path, categorical=True)


def figure_overfit(results: Path, path: Path):
    """Training loss against validation AUC per epoch, showing early peaks."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for number, name in enumerate(NEURAL_DIRS):
        if not (results / name / "history.csv").exists() or not (results / name / "results_summary.csv").exists():
            continue
        history = pd.read_csv(results / name / "history.csv")
        history = history[(history["group"] == "model") & (history["seed"] == history["seed"].min())]
        best = history.groupby("params")["val_auc"].max().idxmax()
        part = history[history["params"] == best]
        label = LABELS.get(name[5:], name[5:])
        axes[0].plot(part["epoch"], part["train_loss"], marker="o", markersize=3, color=plt.cm.tab10(number), label=label)
        axes[1].plot(part["epoch"], part["val_auc"], marker="o", markersize=3, color=plt.cm.tab10(number), label=label)
    axes[0].set(title="Training loss", xlabel="Epoch")
    axes[1].set(title="Validation AUC (noisy: 45 genomes)", xlabel="Epoch")
    axes[1].legend(frameon=False, fontsize=7)
    _finish(figure, axes, path, categorical=True)


def figure_task5(task5: Path, path_dir: Path):
    """Word-effect replication, top words, GC and reverse-complement tests."""
    summary = json.loads((task5 / "interpretation_summary.json").read_text())
    effects = pd.read_csv(task5 / "word_effects_k6.csv")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    axes[0].scatter(effects["genome_auc"], effects["genome_auc_held_out"], s=3, alpha=0.3, color=COLORS[0])
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--")
    axes[0].axhline(0.5, color="0.8"); axes[0].axvline(0.5, color="0.8")
    axes[0].set(title="6-mer genome AUC: train vs held-out genomes", xlabel="train genomes", ylabel="held-out genomes")
    top = pd.concat([effects.nsmallest(10, "genome_auc"), effects.nlargest(10, "genome_auc")]).sort_values("log2_ratio")
    axes[1].barh(top["word"], top["log2_ratio"], color=[COLORS[1] if v < 0 else COLORS[0] for v in top["log2_ratio"]])
    axes[1].set(title="Strongest 6-mers (log2 pathogenic / nonpathogenic)")
    axes[1].tick_params(axis="y", labelsize=7)
    tests = summary["embedding_rc_tests_k6"]
    names = list(tests)
    x = np.arange(len(names))
    axes[2].bar(x - 0.2, [tests[n]["rc_cosine_mean"] for n in names], 0.4, color=COLORS[0], label="word vs its reverse complement")
    axes[2].bar(x + 0.2, [tests[n]["random_pair_cosine_mean"] for n in names], 0.4, color="0.7", label="random word pair")
    axes[2].set_xticks(x, ["Word2Vec\noverlapping", "Word2Vec\nstrided", "count SVD"], fontsize=8)
    axes[2].set(title="Do embeddings know strand symmetry?", ylabel="mean cosine")
    axes[2].legend(frameon=False, fontsize=7)
    _finish(figure, axes, path_dir / "task5_summary.png", categorical=True)
    gc = pd.read_csv(task5 / "genome_gc.csv")
    figure, axis = plt.subplots(figsize=(6, 3.6), layout="constrained")
    for number, (label, value) in enumerate(gc.groupby("label")):
        axis.hist(value["gc"], bins=30, alpha=0.6, color=COLORS[number], label=["nonpathogenic", "pathogenic"][int(label)] if str(label).isdigit() else str(label))
    axis.set(title="GC fraction per genome (all splits)", xlabel="GC fraction", ylabel="genomes")
    axis.legend(frameon=False)
    _finish(figure, axis, path_dir / "task5_gc_per_genome.png", categorical=True)


def figure_task6(path: Path, replies: Path, truth: Dict[str, float]):
    """Cost and answer length per model, and Q3 predictions against the truth."""
    rows = [json.loads(line) for line in replies.read_text().splitlines()]
    frame = pd.DataFrame(rows)
    frame["chars"] = frame["text"].str.len()
    frame["empty"] = frame["chars"] == 0
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    tiers = {
        "very_cheap": "Gemini 3.1\nFlash Lite",
        "cheap_open_weight": "GLM-5.2",
        "medium": "Claude\nSonnet 5.5",
    }
    colors = dict(zip(tiers, COLORS))
    cost = frame.groupby("tier")["cost"].sum().reindex(list(tiers))
    empty = frame.groupby("tier")["empty"].sum().reindex(list(tiers))
    axes[0].bar(list(tiers.values()), cost.values, color=[colors[t] for t in tiers])
    axes[0].set(title="Total cost in USD (24 calls each)")
    axes[1].bar(list(tiers.values()), empty.values, color=[colors[t] for t in tiers])
    axes[1].set(title="Empty replies (of 24 calls)")
    midpoints = [0.85, 0.85, 0.76]
    axes[2].bar(list(tiers.values()), midpoints, color=[colors[t] for t in tiers])
    axes[2].axhline(truth["bacpacs_genome_auc"], color="k", linestyle="--")
    axes[2].text(-0.4, truth["bacpacs_genome_auc"] - 0.035, "measured BacPaCS genome AUC (mean of models)", fontsize=7)
    axes[2].set(title="Q3 prediction: BacPaCS genome AUC (range midpoint)", ylim=(0.5, 0.95))
    for axis in axes:
        axis.tick_params(axis="x", labelsize=7)
    _finish(figure, axes, path, categorical=True)


def markdown_table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(frame.columns) + " |\n| " + " | ".join("---" for _ in frame.columns) + " |\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False))


def build(results: Path, outputs: Path, task5: Path):
    """Rebuild every figure, per-model plots and the consolidated task 4 report."""
    simple = pd.read_csv(results / "simple_models" / "results_summary.csv").astype({"k": "Int64"})
    for name in NEURAL_DIRS:
        folder = results / name
        if not (folder / "results_summary.csv").exists():
            continue
        target = outputs / "task4" / name / "figures"
        target.mkdir(parents=True, exist_ok=True)
        summary = pd.read_csv(folder / "results_summary.csv").astype({"k": "Int64"})
        history = pd.read_csv(folder / "history.csv")
        plot_neural_comparison(summary, simple, target / "neural_comparison.png")
        plot_training_curves(history, target / "training_curves.png")
        for item in ("report.md", "results.csv", "results_summary.csv", "tuning.csv", "history.csv", "summary.json"):
            if (folder / item).exists():
                data = (folder / item).read_bytes()
                if item == "report.md":
                    data = data.replace(
                        b"Networks are small on purpose (see the parameter counts). ", b""
                    ) + (
                        "\n## Added after the run\n\n"
                        "- Genome-level metrics and the extra test sets (BacPaCS, 150/100 bp reads, shuffled letters) "
                        "are in `genome/genome_summary.csv`; cross-model figures are in `../summary/`.\n"
                        "- Validation AUC is noisy (45 genomes) and early epochs are often best, so the stopping epoch and "
                        "setting choice carry little information.\n"
                    ).encode()
                (outputs / "task4" / name / item).write_bytes(data)
        genome_target = outputs / "task4" / name / "genome"
        genome_target.mkdir(exist_ok=True)
        for item in (folder / "genome").glob("*.csv"):
            (genome_target / item.name).write_bytes(item.read_bytes())
    table = collect(results)
    base = baselines(results)
    folder = outputs / "task4" / "summary"
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder / "all_models_by_split.csv", index=False)
    figure_comparison(table, base, folder / "comparison.png")
    figure_robustness(table, folder / "robustness.png")
    figure_aggregation(results, table, folder / "reads_per_genome.png")
    figure_overfit(results, folder / "training_overfit.png")
    figure_task5(task5, outputs / "task5")
    truth = {"bacpacs_genome_auc": float(table[(table["split"] == "bacpacs_test_1") & (table["model"] != "dnabert2_scratch")]["genome_auc"].mean())}
    replies = outputs / "task6" / "llm_replies.jsonl"
    if replies.exists():
        figure_task6(outputs / "task6" / "llm_overview.png", replies, truth)
    wide = table.pivot(index="model", columns="split", values="read_auc").round(3)
    wide_g = table.pivot(index="model", columns="split", values="genome_auc").round(3)
    cols = [s for s, _ in CONDITIONS]
    text = (
        "# Task 4: all models, summary\n\n"
        "Read-level AUC (best setting by validation AUC, mean over seeds). `simple` is the best simple model.\n\n"
        + markdown_table(wide[cols].reset_index()) + "\n\n"
        "Genome-level AUC (each genome's mean read score; pooled test = both mates):\n\n"
        + markdown_table(wide_g[cols].reset_index()) + "\n\n"
        "![Comparison](comparison.png)\n\n![Robustness](robustness.png)\n\n"
        "![Reads per genome](reads_per_genome.png)\n\n![Overfitting](training_overfit.png)\n\n"
        "Notes: validation AUC is noisy (45 genomes, three nonpathogenic ones hold about 60% of that class), "
        "so selection among settings is close to random. Most networks peak after 1-2 epochs while training "
        "loss keeps falling, i.e. they learn the training genomes rather than a more general rule. "
        "Random-initialised DNABERT-2 learns once the learning rate is warmed up (a first run without warmup stayed at chance; kept in `full_dnabert2_scratch_nowarmup`); "
        "its best epoch was the first (val AUC 0.771, pretrained 0.781) and it diverged in epoch 4, so on validation pretraining helps little, while on test and BacPaCS the pretrained model is clearly better. "
        "`gru_long` repeats the BiGRU with up to 14 epochs: no better validation AUC, and test AUC differs from the first BiGRU by 0.02, which is the size of single-seed noise here. "
        "Gradient boosting is skipped for tables wider than 1024 columns (k = 6 counts/TF-IDF).\n\n"
        "Models were trained on 1M reads (GRU, transformer, DNABERT-2) or 2M reads (CNN, MLP); simple models used 400k.\n"
    )
    (folder / "report.md").write_text(text, encoding="utf-8")
    print("Saved summary figures and report: " + str(folder), flush=True)
