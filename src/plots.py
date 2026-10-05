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
