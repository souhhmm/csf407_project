from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


COLORS = ["#3d6f9e", "#bf8655", "#718c46", "#76569a"]
TITLES = {
    "counts": "Word counts",
    "tfidf": "Weighted word counts",
    "mean_vectors": "Average word vectors",
    "weighted_vectors": "Weighted average word vectors",
}


def _finish(figure, axes, path: Path):
    for axis in np.asarray(axes, dtype=object).ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)
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
            color=color,
        )
        axes[1].hist(
            group["gc_fraction"], bins=np.linspace(0, 1, 26), color=color, alpha=0.5
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
    axis.barh(common["word"], common["count"], color=COLORS[0])
    axis.set(title="Common " + str(k) + "-letter words", xlabel="Occurrences")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=9)
    axis.xaxis.set_major_locator(MaxNLocator(4))
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


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
        linewidth=1.5,
        markersize=4,
    )
    axis.set(
        title="Vocabulary within a sequence",
        xlabel="Letters per word",
        ylabel="Average percentage of possible words",
        xticks=summary["k"],
        ylim=(0, 100),
    )
    _finish(figure, [axis], path)
