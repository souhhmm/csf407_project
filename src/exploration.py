import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.plots import (
    TITLES,
    plot_composition_spaces,
    plot_frequency_concentration,
    plot_word_ratios,
)
from src.sequences import ALPHABETS


def compare_class_frequencies(frequencies: pd.DataFrame) -> pd.DataFrame:
    """Compare word rates for two classes without training a predictor.

    Args:
        frequencies: Word counts for each class, including zero-count words.

    Returns:
        Counts, rates and smoothed log2 ratios for the second class over the first.
        An empty table is returned unless exactly two classes contain words.
    """
    labels = sorted(frequencies["label"].unique())
    if len(labels) != 2:
        return pd.DataFrame()
    counts = frequencies.pivot(index="word", columns="label", values="count")
    counts = counts.fillna(0).sort_index()
    totals = counts.sum()
    if (totals == 0).any():
        return pd.DataFrame()
    denominator, numerator = labels
    rates = counts.div(totals)
    smoothed = (counts + 0.5).div(totals + 0.5 * len(counts))
    return pd.DataFrame(
        {
            "word": counts.index,
            "label_numerator": numerator,
            "label_denominator": denominator,
            "count_numerator": counts[numerator].to_numpy(),
            "count_denominator": counts[denominator].to_numpy(),
            "fraction_numerator": rates[numerator].to_numpy(),
            "fraction_denominator": rates[denominator].to_numpy(),
            "log2_ratio": np.log2(
                smoothed[numerator] / smoothed[denominator]
            ).to_numpy(),
        }
    )


def compare_shuffled_letters(
    records: list, k: int, alphabet: str = "dna", repeats: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Compare word counts with repeated shuffles inside known-letter runs.

    Args:
        records: Checked records containing a sequence field.
        k: Number of letters in each word.
        alphabet: Either dna or rna.
        repeats: Number of independently shuffled versions of the full dataset.
        seed: Fixed random setting.

    Returns:
        Observed counts, shuffled count summaries and smoothed log2 count ratios.
        Runs retain their lengths and letter counts, and unknown gaps stay fixed.

    Raises:
        ValueError: If a word size or repeat count is not positive.
    """
    if k < 1 or repeats < 1:
        raise ValueError("Word size and shuffle repeats must be at least 1.")
    pattern = "[" + ALPHABETS[alphabet] + "]+"
    runs = [
        run
        for record in records
        for run in re.findall(pattern, record["sequence"])
        if len(run) >= k
    ]
    observed = Counter()
    for run in runs:
        observed.update(run[start : start + k] for start in range(len(run) - k + 1))
    encoded = [np.frombuffer(run.encode("ascii"), dtype=np.uint8) for run in runs]
    generator = np.random.default_rng(seed)
    shuffled_counts = []
    for _ in range(repeats):
        counts = Counter()
        for run in encoded:
            shuffled = generator.permutation(run).tobytes().decode("ascii")
            counts.update(
                shuffled[start : start + k] for start in range(len(shuffled) - k + 1)
            )
        shuffled_counts.append(counts)
    words = sorted(set(observed).union(*(set(counts) for counts in shuffled_counts)))
    samples = np.asarray(
        [[counts[word] for word in words] for counts in shuffled_counts], dtype=float
    )
    actual = np.asarray([observed[word] for word in words], dtype=int)
    mean = samples.mean(axis=0)
    return pd.DataFrame(
        {
            "word": words,
            "count": actual,
            "shuffled_mean": mean,
            "shuffled_sd": samples.std(axis=0, ddof=1 if repeats > 1 else 0),
            "shuffled_min": samples.min(axis=0),
            "shuffled_max": samples.max(axis=0),
            "log2_ratio": np.log2((actual + 0.5) / (mean + 0.5)),
        }
    )


def _rank_correlation(left: np.ndarray, right: np.ndarray):
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    if len(left) < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    return float(spearmanr(left, right).statistic)


def _select_ratios(table: pd.DataFrame) -> pd.DataFrame:
    negative = table.loc[table["log2_ratio"] < 0].nsmallest(6, "log2_ratio")
    positive = table.loc[table["log2_ratio"] > 0].nlargest(6, "log2_ratio")
    return pd.concat([negative, positive]).sort_values("log2_ratio")


def create_extra_analysis(records: list, summary: dict, output_dir: Path) -> dict:
    """Add descriptive analyses using the saved representations and original data.

    Args:
        records: Checked records in the original analysis order.
        summary: Existing settings and word-size results.
        output_dir: Directory containing the saved analysis.

    Returns:
        Additional settings, results and explanations of skipped comparisons.

    Raises:
        ValueError: If saved sequence rows do not match the input records.
    """
    started = time.perf_counter()
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    for filename in (
        "frequency_concentration.png",
        "sequence_composition.png",
        "class_word_differences.png",
        "word_order_effect.png",
    ):
        (figures / filename).unlink(missing_ok=True)
    completed = [
        result for result in summary["word_sizes"] if result["status"] == "complete"
    ]
    if not completed:
        return {"status": "skipped", "reason": "No word size has usable words."}
    stats = pd.read_csv(
        output_dir / "sequence_statistics.csv", dtype={"id": str, "label": str}
    )
    expected_rows = [
        {"id": record["id"], "label": record["label"]} for record in records
    ]
    if stats[["id", "label"]].to_dict("records") != expected_rows:
        raise ValueError("Saved sequence statistics do not match the input rows.")
    frequency_tables, composition_points, composition_rows, concentration = (
        {},
        {},
        [],
        [],
    )
    for result in completed:
        k = result["k"]
        directory = output_dir / ("k_" + str(k))
        frequencies = pd.read_csv(directory / "word_frequencies.csv")
        frequencies = frequencies.sort_values(
            ["count", "word"], ascending=[False, True]
        )
        frequencies["rank"] = np.arange(1, len(frequencies) + 1)
        frequencies["cumulative_fraction"] = frequencies["fraction"].cumsum()
        frequencies.to_csv(directory / "frequency_ranks.csv", index=False)
        frequency_tables[k] = frequencies
        top_count = max(1, int(np.ceil(len(frequencies) * 0.1)))
        concentration.append(
            {
                "k": k,
                "words_in_top_group": top_count,
                "observed_words": len(frequencies),
                "top_group_occurrence_fraction": float(
                    frequencies.head(top_count)["fraction"].sum()
                ),
            }
        )
        usable = pd.read_csv(directory / "word_statistics.csv", dtype={"id": str})[
            ["id", "words"]
        ]
        for name in TITLES:
            points = pd.read_csv(
                directory / ("projection_" + name + ".csv"),
                dtype={"id": str, "label": str},
            )
            if points[["id", "label"]].to_dict("records") != expected_rows:
                raise ValueError("Saved projection rows do not match the input rows.")
            points = points.merge(
                stats[["id", "gc_fraction"]], on="id", validate="one_to_one"
            )
            points = points.merge(usable, on="id", validate="one_to_one")
            points = points.loc[(points["words"] > 0) & points["gc_fraction"].notna()]
            for coordinate in ("coordinate_1", "coordinate_2"):
                composition_rows.append(
                    {
                        "k": k,
                        "representation": name,
                        "coordinate": coordinate,
                        "sequences": len(points),
                        "gc_rank_correlation": _rank_correlation(
                            points[coordinate].to_numpy(),
                            points["gc_fraction"].to_numpy(),
                        ),
                    }
                )
            if name == "mean_vectors":
                composition_points[k] = points
    pd.DataFrame(concentration).to_csv(
        output_dir / "frequency_concentration.csv", index=False
    )
    pd.DataFrame(composition_rows).to_csv(
        output_dir / "composition_correlations.csv", index=False
    )
    plot_frequency_concentration(
        frequency_tables, figures / "frequency_concentration.png"
    )
    plot_composition_spaces(composition_points, figures / "sequence_composition.png")
    focus = min((result["k"] for result in completed), key=lambda k: (abs(k - 4), k))
    directory = output_dir / ("k_" + str(focus))
    contrasts = compare_class_frequencies(
        pd.read_csv(directory / "word_frequencies_by_class.csv", dtype={"label": str})
    )
    class_comparison = {
        "k": focus,
        "status": "skipped",
        "reason": "Exactly two classes with usable words are required.",
    }
    if not contrasts.empty:
        contrasts.to_csv(directory / "class_word_comparison.csv", index=False)
        selected = _select_ratios(
            contrasts.loc[
                (contrasts["count_numerator"] >= 20)
                & (contrasts["count_denominator"] >= 20)
            ]
        )
        if not selected.empty:
            numerator, denominator = contrasts.iloc[0][
                ["label_numerator", "label_denominator"]
            ]
            plot_word_ratios(
                selected,
                "Source-class word frequencies (k = " + str(focus) + ")",
                numerator + " / " + denominator,
                figures / "class_word_differences.png",
            )
            selected.to_csv(directory / "class_words_shown.csv", index=False)
            class_comparison = {
                "k": focus,
                "status": "complete",
                "numerator": numerator,
                "denominator": denominator,
                "minimum_count_per_class": 20,
            }
        else:
            class_comparison["reason"] = (
                "No differing words have at least 20 occurrences in each class."
            )
    eligible = [result["k"] for result in completed if result["k"] <= 6]
    word_order = {
        "status": "skipped",
        "reason": "Word-order comparison requires a completed word size of at most six.",
    }
    if eligible:
        k = min(eligible, key=lambda k: (abs(k - 4), k))
        print("Comparing original words with 10 letter shuffles...", flush=True)
        shuffled = compare_shuffled_letters(
            records, k, summary["alphabet"], seed=summary["settings"]["seed"]
        )
        shuffled.to_csv(
            output_dir / ("k_" + str(k)) / "shuffled_word_comparison.csv", index=False
        )
        selected = _select_ratios(
            shuffled.loc[(shuffled["count"] >= 20) & (shuffled["shuffled_mean"] >= 20)]
        )
        word_order = {
            "k": k,
            "status": "complete",
            "repeats": 10,
            "seed": summary["settings"]["seed"],
            "minimum_count": 20,
            "plot_created": not selected.empty,
        }
        if not selected.empty:
            plot_word_ratios(
                selected,
                "Effect of rearranging letters (k = " + str(k) + ")",
                "original / average shuffled",
                figures / "word_order_effect.png",
            )
            selected.to_csv(
                output_dir / ("k_" + str(k)) / "shuffled_words_shown.csv", index=False
            )
    return {
        "status": "complete",
        "frequency_concentration": concentration,
        "composition_correlations": composition_rows,
        "class_comparison": class_comparison,
        "word_order": word_order,
        "elapsed_seconds": time.perf_counter() - started,
    }
