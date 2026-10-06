from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


COLORS = ["#3569b8", "#d97939", "#1c8c83", "#8665b4"]
HATCHES = ["///", "\\\\", "||", "--"]
GC_COLORS = matplotlib.colors.LinearSegmentedColormap.from_list(
    "composition", ["#2f5ca6", "#239589", "#e9b44c"]
)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "text.color": "#233044",
        "axes.labelcolor": "#334155",
        "axes.titleweight": "semibold",
        "legend.labelcolor": "#334155",
        "hatch.linewidth": 0.75,
    }
)
TITLES = {
    "counts": "Word counts",
    "tfidf": "Weighted word counts",
    "mean_vectors": "Average word vectors",
    "weighted_vectors": "Weighted average word vectors",
}


def _finish(figure, axes, path: Path, categorical: bool = False):
    for axis in np.asarray(axes, dtype=object).ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#cbd5e1")
        axis.spines[["left", "bottom"]].set_linewidth(0.8)
        axis.tick_params(labelsize=9, colors="#475569", length=3)
        if not categorical:
            axis.yaxis.set_major_locator(MaxNLocator(4))
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_sequence_statistics(stats: pd.DataFrame, alphabet: str, path: Path):
    """Plot letter proportions and GC fractions using two simple panels.

    Args:
        stats: Sequence statistics with a label column.
        alphabet: Either dna or rna.
        path: Destination PNG.
    """
    letters = "ACGT" if alphabet == "dna" else "ACGU"
    labels = sorted(stats["label"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    width = 0.75 / len(labels)
    for number, label in enumerate(labels):
        group = stats.loc[stats["label"] == label]
        color = COLORS[number % len(COLORS)]
        axes[0].bar(
            np.arange(4) - 0.375 + width / 2 + number * width,
            [group[letter + "_fraction"].mean() for letter in letters],
            width=width,
            label=label,
            facecolor=matplotlib.colors.to_rgba(color, 0.07),
            edgecolor=color,
            linewidth=0.9,
            hatch=HATCHES[number % len(HATCHES)],
        )
        axes[1].hist(
            group["gc_fraction"],
            bins=np.linspace(0, 1, 26),
            histtype="stepfilled",
            facecolor=matplotlib.colors.to_rgba(color, 0.05),
            edgecolor=color,
            linewidth=1.3,
            hatch=HATCHES[number % len(HATCHES)],
        )
    axes[0].set(
        title="Letter proportions",
        ylabel="Average fraction",
        xticks=range(4),
        xticklabels=list(letters),
    )
    axes[1].set(
        title="G and C proportions", xlabel="Fraction of G and C", ylabel="Sequences"
    )
    axes[0].legend(frameon=False, fontsize=8)
    _finish(figure, axes, path)


def plot_word_frequencies(frequencies: pd.DataFrame, k: int, path: Path):
    """Plot the ten most common words.

    Args:
        frequencies: Word and count columns, sorted by descending count.
        k: Word size.
        path: Destination PNG.
    """
    common = frequencies.head(10).iloc[::-1]
    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    axis.barh(
        common["word"],
        common["count"],
        facecolor=matplotlib.colors.to_rgba(COLORS[0], 0.07),
        edgecolor=COLORS[0],
        linewidth=0.8,
        hatch=HATCHES[0],
    )
    axis.set(title="Common " + str(k) + "-letter words", xlabel="Occurrences")
    axis.xaxis.set_major_locator(MaxNLocator(4))
    _finish(figure, [axis], path, categorical=True)


def plot_sequence_spaces(
    projections: dict, labels: list, information: dict, k: int, path: Path
):
    """Compare the four sequence representations in a compact panel.

    Args:
        projections: Representation names mapped to two-coordinate arrays.
        labels: Class label for each sequence in row order.
        information: Projection methods and captured variation.
        k: Word size.
        path: Destination PNG.
    """
    labels = np.asarray(labels)
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for axis, (name, points) in zip(axes.flat, projections.items()):
        for number, label in enumerate(sorted(set(labels))):
            group = points[labels == label]
            axis.scatter(
                group[:, 0],
                group[:, 1],
                s=5,
                alpha=0.25,
                color=COLORS[number % len(COLORS)],
                linewidths=0,
                label=label,
                rasterized=True,
            )
        shown = format(information[name]["captured_variation"], ".0%")
        axis.set(
            title=TITLES[name] + " · " + shown + " shown",
            xlabel="Coordinate 1",
            ylabel="Coordinate 2",
        )
        axis.xaxis.set_major_locator(MaxNLocator(3))
    axes[0, 0].legend(frameon=False, markerscale=2, fontsize=8)
    figure.suptitle(str(k) + "-letter words", fontsize=13)
    _finish(figure, axes, path)


def plot_k_comparison(summary: pd.DataFrame, path: Path):
    """Plot how much of the possible vocabulary each sequence contains.

    Args:
        summary: Completed word-size summaries.
        path: Destination PNG.
    """
    figure, axis = plt.subplots(figsize=(6, 3.8), layout="constrained")
    axis.plot(
        summary["k"],
        summary["mean_sequence_coverage"] * 100,
        "o-",
        color=COLORS[0],
        linewidth=2,
        markersize=5,
    )
    axis.fill_between(
        summary["k"],
        summary["mean_sequence_coverage"] * 100,
        0,
        facecolor=matplotlib.colors.to_rgba(COLORS[0], 0.03),
        edgecolor=matplotlib.colors.to_rgba(COLORS[0], 0.3),
        hatch=HATCHES[0],
        linewidth=0,
    )
    axis.set(
        title="Vocabulary within a sequence",
        xlabel="Letters per word",
        ylabel="Average percentage of possible words",
        xticks=summary["k"],
        ylim=(0, 100),
    )
    _finish(figure, [axis], path)


def plot_frequency_concentration(tables: dict, path: Path):
    """Show how much of the word count comes from the most frequent words.

    Args:
        tables: Word sizes mapped to ranked frequencies and cumulative fractions.
        path: Destination PNG.
    """
    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    for number, (k, table) in enumerate(sorted(tables.items())):
        axis.plot(
            np.r_[0, table["rank"].to_numpy() / len(table) * 100],
            np.r_[0, table["cumulative_fraction"].to_numpy() * 100],
            color=COLORS[number % len(COLORS)],
            linestyle=["-", "--", "-.", ":"][number % 4],
            linewidth=1.8,
            label="k = " + str(k),
        )
    axis.plot(
        [0, 100], [0, 100], "--", color="0.6", linewidth=1, label="Equal frequencies"
    )
    axis.set(
        title="Concentration of word frequencies",
        xlabel="Percentage of words, most frequent first",
        ylabel="Percentage of word occurrences",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    axis.legend(frameon=False, fontsize=8)
    _finish(figure, [axis], path)


def plot_composition_spaces(points_by_k: dict, path: Path):
    """Colour saved mean-vector projections by their G and C proportions.

    Args:
        points_by_k: Word sizes mapped to coordinates and GC fractions.
        path: Destination PNG.
    """
    columns = min(3, len(points_by_k))
    rows = int(np.ceil(len(points_by_k) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4 * columns, 3.8 * rows),
        squeeze=False,
        layout="constrained",
    )
    for axis, (k, points) in zip(axes.flat, sorted(points_by_k.items())):
        axis.scatter(
            points["coordinate_1"],
            points["coordinate_2"],
            c=points["gc_fraction"],
            cmap=GC_COLORS,
            vmin=0,
            vmax=1,
            s=5,
            alpha=0.4,
            linewidths=0,
            rasterized=True,
        )
        axis.set(title="k = " + str(k), xlabel="Coordinate 1", ylabel="Coordinate 2")
        axis.xaxis.set_major_locator(MaxNLocator(3))
    for axis in list(axes.flat)[len(points_by_k) :]:
        axis.set_visible(False)
    figure.colorbar(
        plt.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 1), cmap=GC_COLORS),
        ax=list(axes.flat),
        label="Fraction of G and C",
        shrink=0.85,
    )
    figure.suptitle("Average word vectors coloured by letter composition", fontsize=13)
    _finish(figure, axes, path)


def plot_word_ratios(table: pd.DataFrame, title: str, comparison: str, path: Path):
    """Plot selected word ratios around an equal-frequency reference.

    Args:
        table: Selected words with smoothed log2 ratios.
        title: Short title identifying the comparison.
        comparison: Numerator and denominator labels.
        path: Destination PNG.
    """
    figure, axis = plt.subplots(figsize=(7, 4.6), layout="constrained")
    for word, value in table[["word", "log2_ratio"]].itertuples(index=False, name=None):
        group = 0 if value >= 0 else 1
        color = COLORS[group]
        axis.barh(
            word,
            value,
            facecolor=matplotlib.colors.to_rgba(color, 0.07),
            edgecolor=color,
            linewidth=0.8,
            hatch=HATCHES[group],
        )
    axis.axvline(0, color="0.5", linewidth=1)
    axis.xaxis.set_major_locator(MaxNLocator(5))
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, position: format(2**value, ".2g") + "×")
    )
    axis.set(title=title, xlabel="Ratio: " + comparison)
    _finish(figure, [axis], path, categorical=True)


REPRESENTATION_LABELS = {
    "counts": "Counts",
    "tfidf": "TF-IDF",
    "mean_vectors": "Mean\nvectors",
    "weighted_vectors": "Weighted\nvectors",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "decision_tree": "Decision tree",
    "random_forest": "Random forest",
    "naive_bayes": "Naive Bayes",
    "gradient_boosting": "Gradient boosting",
}
SPLIT_LABELS = {"val": "Validation", "test_1": "Test 1", "test_2": "Test 2"}


def _best_per_representation(models: pd.DataFrame) -> pd.DataFrame:
    """Keep each representation and word size's model with the best validation AUC."""
    ordered = models.sort_values("val_auc_mean", ascending=False)
    return ordered.drop_duplicates(["representation", "k"]).sort_values(
        ["representation", "k"]
    )


def plot_model_heatmap(summary: pd.DataFrame, path: Path):
    """Show AUC for every model, representation, and word size on each split.

    Args:
        summary: Seed-averaged results with group, model, representation, k,
            and {split}_auc_mean columns.
        path: Destination PNG.
    """
    models = summary[summary["group"] == "model"]
    rows = [name for name in MODEL_LABELS if name in set(models["model"])]
    columns = [
        (representation, int(k))
        for representation in REPRESENTATION_LABELS
        for k in sorted(
            models.loc[models["representation"] == representation, "k"].unique()
        )
    ]
    splits = list(SPLIT_LABELS)
    top = max(
        0.6, float(np.nanmax(models[[s + "_auc_mean" for s in splits]].to_numpy()))
    )
    figure, axes = plt.subplots(
        len(splits),
        1,
        figsize=(
            max(7, 0.9 * len(columns) + 2.5),
            1.0 * len(rows) * len(splits) + 1.5,
        ),
        squeeze=False,
        layout="constrained",
    )
    for axis, split in zip(axes.ravel(), splits):
        grid = np.full((len(rows), len(columns)), np.nan)
        for _, row in models.iterrows():
            grid[
                rows.index(row["model"]),
                columns.index((row["representation"], int(row["k"]))),
            ] = row[split + "_auc_mean"]
        image = axis.imshow(grid, cmap=GC_COLORS, vmin=0.5, vmax=top, aspect="auto")
        for y, x in np.argwhere(~np.isnan(grid)):
            axis.text(
                x,
                y,
                format(grid[y, x], ".2f"),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if grid[y, x] < 0.5 + (top - 0.5) * 0.3 else "#233044",
            )
        axis.set(
            title=SPLIT_LABELS[split] + " AUC",
            xticks=range(len(columns)),
            xticklabels=[
                REPRESENTATION_LABELS[name] + "\nk=" + str(k) for name, k in columns
            ],
            yticks=range(len(rows)),
            yticklabels=[MODEL_LABELS[name] for name in rows],
        )
        axis.tick_params(labelsize=8)
    figure.colorbar(image, ax=list(axes.ravel()), label="AUC", shrink=0.8)
    _finish(figure, axes, path, categorical=True)


def plot_baseline_comparison(summary: pd.DataFrame, path: Path):
    """Compare each representation's best model with baselines and the control.

    The best model for each representation and word size is chosen on
    validation AUC, then shown on every split. Error bars are genome-level
    bootstrap intervals averaged over seeds.

    Args:
        summary: Seed-averaged results with group, model, representation, k,
            {split}_auc_mean, and {split}_auc_low/high columns.
        path: Destination PNG.
    """
    best = _best_per_representation(summary[summary["group"] == "model"])
    ks = sorted(best["k"].unique())
    references = [
        ("Letter composition, k=2", "baseline", "composition_k2", COLORS[1], "--"),
        ("G and C fraction only", "baseline", "gc_fraction", COLORS[2], "-."),
        ("Majority class / chance", "baseline", "none", "0.5", ":"),
    ]
    control = summary[summary["group"] == "control"]
    figure, axes = plt.subplots(
        1, len(SPLIT_LABELS), figsize=(14, 4.2), sharey=True, layout="constrained"
    )
    width = 0.8 / max(len(ks), 1)
    for axis, split in zip(axes, SPLIT_LABELS):
        for number, k in enumerate(ks):
            part = best[best["k"] == k].set_index("representation")
            positions = []
            heights, lows, highs = [], [], []
            for index, representation in enumerate(REPRESENTATION_LABELS):
                if representation in part.index:
                    row = part.loc[representation]
                    positions.append(index - 0.4 + width / 2 + number * width)
                    heights.append(row[split + "_auc_mean"])
                    lows.append(row[split + "_auc_mean"] - row[split + "_auc_low"])
                    highs.append(row[split + "_auc_high"] - row[split + "_auc_mean"])
            color = COLORS[number % len(COLORS)]
            axis.bar(
                positions,
                heights,
                width=width,
                yerr=np.nan_to_num([lows, highs]),
                error_kw={"linewidth": 0.8, "capsize": 2, "ecolor": "0.4"},
                label="k = " + str(int(k)),
                facecolor=matplotlib.colors.to_rgba(color, 0.07),
                edgecolor=color,
                linewidth=0.9,
                hatch=HATCHES[number % len(HATCHES)],
            )
        for label, group, name, color, style in references:
            value = summary.loc[
                (summary["group"] == group) & (summary["representation"] == name),
                split + "_auc_mean",
            ]
            if len(value):
                axis.axhline(
                    float(value.iloc[0]),
                    color=color,
                    linestyle=style,
                    linewidth=1.2,
                    label=label,
                )
        if len(control):
            axis.axhline(
                float(control[split + "_auc_mean"].mean()),
                color="0.3",
                linestyle=(0, (1, 3)),
                linewidth=1.2,
                label="Shuffled-label control",
            )
        axis.set(
            title=SPLIT_LABELS[split],
            xticks=range(len(REPRESENTATION_LABELS)),
            xticklabels=list(REPRESENTATION_LABELS.values()),
            ylim=(0.4, 1.0),
        )
    axes[0].set_ylabel("AUC of best model by validation")
    axes[-1].legend(
        frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1)
    )
    _finish(figure, axes, path, categorical=True)


def plot_generalisation(summary: pd.DataFrame, path: Path):
    """Plot validation against test AUC for every configuration.

    Args:
        summary: Seed-averaged results with val_auc_mean and test AUC columns.
        path: Destination PNG.
    """
    figure, axes = plt.subplots(
        1, 2, figsize=(10, 4.2), sharey=True, layout="constrained"
    )
    for axis, split in zip(axes, ["test_1", "test_2"]):
        for number, representation in enumerate(REPRESENTATION_LABELS):
            part = summary[
                (summary["group"] == "model")
                & (summary["representation"] == representation)
            ]
            axis.scatter(
                part["val_auc_mean"],
                part[split + "_auc_mean"],
                s=22,
                color=COLORS[number % len(COLORS)],
                alpha=0.8,
                linewidths=0,
                label=REPRESENTATION_LABELS[representation],
            )
        baselines = summary[summary["group"] == "baseline"]
        axis.scatter(
            baselines["val_auc_mean"],
            baselines[split + "_auc_mean"],
            marker="x",
            s=26,
            color="0.3",
            label="Baselines",
        )
        axis.plot([0.4, 1], [0.4, 1], "--", color="0.6", linewidth=1)
        axis.set(
            title=SPLIT_LABELS[split] + " against validation",
            xlabel="Validation AUC",
            xlim=(0.45, 1),
            ylim=(0.45, 1),
        )
    axes[0].set_ylabel("Test AUC")
    axes[0].legend(frameon=False, fontsize=8)
    _finish(figure, axes, path, categorical=True)
