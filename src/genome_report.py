import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.classifiers import fast_auc
from src.splits import EVALUATION_SPLITS


POSITIVE_LABEL = "pathogenic"
AGGREGATION_SIZES = (1, 2, 5, 10, 20, 50, 100, 200)
AGGREGATION_DRAWS = 30


def load_truth(splits_dir: Path, names: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """Read the label, genome and header of every read of the named splits.

    Args:
        splits_dir: Directory written by prepare_splits.
        names: Split names to read.

    Returns:
        One table per split with label (1 = pathogenic), genome and header.
    """
    truth = {}
    for name in names:
        table = pd.read_csv(
            splits_dir / (name + ".csv"),
            usecols=["label", "genome", "source_header", "sequence"],
            dtype=str,
        )
        # The loaders drop repeated sequences (keeping the first), so do the same.
        table = table[~table["sequence"].str.upper().duplicated()].reset_index(
            drop=True
        ).drop(columns="sequence")
        table["label"] = (table["label"] == POSITIVE_LABEL).astype(int)
        truth[name] = table
    return truth


def genome_weighted_auc(
    labels: np.ndarray, scores: np.ndarray, genomes: np.ndarray
) -> float:
    """Read-level AUC in which every genome carries equal weight within its class.

    Args:
        labels: 1 for pathogenic reads, 0 otherwise.
        scores: Estimated probability of pathogenic for each read.
        genomes: Source genome of each read.

    Returns:
        The weighted AUC, or NaN when only one class is present.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    _, inverse, counts = np.unique(genomes, return_inverse=True, return_counts=True)
    return float(roc_auc_score(labels, scores, sample_weight=1.0 / counts[inverse]))


def genome_table(
    labels: np.ndarray, scores: np.ndarray, genomes: np.ndarray
) -> pd.DataFrame:
    """Summarise reads per genome.

    Args:
        labels: 1 for pathogenic reads, 0 otherwise.
        scores: Estimated probability of pathogenic for each read.
        genomes: Source genome of each read.

    Returns:
        One row per genome with its class, read count, mean read score, and the
        fraction of reads called correctly at the 0.5 threshold.
    """
    frame = pd.DataFrame({"genome": genomes, "label": labels, "score": scores})
    frame["correct"] = ((frame["score"] >= 0.5).astype(int) == frame["label"]).astype(
        float
    )
    return (
        frame.groupby("genome")
        .agg(
            label=("label", "first"),
            reads=("score", "size"),
            mean_score=("score", "mean"),
            read_accuracy=("correct", "mean"),
        )
        .reset_index()
    )


def genome_level_metrics(table: pd.DataFrame) -> dict:
    """Score the genome-level call made from each genome's mean read score.

    Args:
        table: Result of genome_table.

    Returns:
        AUC over genomes, accuracy and balanced accuracy at a 0.5 mean score,
        and the number of genomes of each class.
    """
    labels = table["label"].to_numpy()
    called = (table["mean_score"].to_numpy() >= 0.5).astype(int)
    recalls = [
        float((called[labels == value] == value).mean())
        for value in (0, 1)
        if (labels == value).any()
    ]
    return {
        "genome_auc": fast_auc(labels, table["mean_score"].to_numpy()),
        "genome_accuracy": float((called == labels).mean()),
        "genome_balanced_accuracy": float(np.mean(recalls)),
        "genomes_nonpathogenic": int((labels == 0).sum()),
        "genomes_pathogenic": int((labels == 1).sum()),
    }


def aggregation_curve(
    labels: np.ndarray, scores: np.ndarray, genomes: np.ndarray, seed: int = 42
) -> List[dict]:
    """Measure genome-level AUC when only a few reads per genome are averaged.

    Args:
        labels: 1 for pathogenic reads, 0 otherwise.
        scores: Estimated probability of pathogenic for each read.
        genomes: Source genome of each read.
        seed: Fixed random setting.

    Returns:
        For each number of reads per genome, the mean and spread of genome-level
        AUC over repeated random draws.
    """
    rng = np.random.default_rng(seed)
    groups = pd.Series(np.arange(len(scores))).groupby(genomes).apply(np.asarray)
    classes = np.asarray([labels[rows[0]] for rows in groups])
    rows = []
    for size in AGGREGATION_SIZES:
        values = []
        for _ in range(AGGREGATION_DRAWS):
            means = np.asarray(
                [
                    scores[rng.choice(r, size=min(size, len(r)), replace=False)].mean()
                    for r in groups
                ]
            )
            values.append(fast_auc(classes, means))
        rows.append(
            {
                "reads_per_genome": size,
                "genome_auc_mean": float(np.nanmean(values)),
                "genome_auc_std": float(np.nanstd(values)),
            }
        )
    return rows


def run_genome_report(
    splits_dir: Path, results_dir: Path, output_dir: Path, curve_splits: Sequence[str]
) -> pd.DataFrame:
    """Add genome-level statistics to saved per-read scores of one results folder.

    Args:
        splits_dir: Directory written by prepare_splits.
        results_dir: Results folder holding a scores/ directory of .npz files.
        output_dir: Destination for genome_summary.csv, genome_tables.csv and
            aggregation_curves.csv.
        curve_splits: Splits for which to compute the reads-per-genome curve.

    Returns:
        One row per configuration, seed and split, with read-level AUC,
        genome-weighted AUC and genome-level metrics. Mates (test_1 and test_2)
        are also pooled into a "test_pooled" split by averaging each fragment's
        two reads' genome means via one combined table.
    """
    files = sorted((results_dir / "scores").glob("*.npz"))
    manifest = json.loads((splits_dir / "manifest.json").read_text(encoding="utf-8"))
    names = tuple(EVALUATION_SPLITS) + tuple(manifest.get("extra_sets", {}))
    truth = load_truth(splits_dir, names)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, tables, curves = [], [], []
    for path in files:
        tag = path.stem
        found = re.match(r"(.*)_s(\d+)$", tag)
        key, seed = found.group(1), int(found.group(2))
        saved = np.load(path)
        group, model, representation, k = (key.split("_", 3) + [""] * 4)[:4]
        pooled = {"labels": [], "scores": [], "genomes": []}
        for split in saved.files:
            labels = truth[split]["label"].to_numpy()
            genomes = truth[split]["genome"].to_numpy()
            scores = saved[split].astype(np.float64)
            table = genome_table(labels, scores, genomes)
            row = {
                "tag": key,
                "seed": seed,
                "split": split,
                "read_auc": fast_auc(labels, scores),
                "genome_weighted_auc": genome_weighted_auc(labels, scores, genomes),
                **genome_level_metrics(table),
            }
            summary.append(row)
            tables.append(table.assign(tag=key, seed=seed, split=split))
            if split in ("test_1", "test_2"):
                for field, value in (
                    ("labels", labels),
                    ("scores", scores),
                    ("genomes", genomes),
                ):
                    pooled[field].append(value)
            if split in curve_splits and seed == min(
                int(re.search(r"_s(\d+)$", f.stem).group(1)) for f in files
            ):
                curves.extend(
                    {**item, "tag": key, "split": split}
                    for item in aggregation_curve(labels, scores, genomes)
                )
        if pooled["labels"]:
            labels = np.concatenate(pooled["labels"])
            genomes = np.concatenate(pooled["genomes"])
            scores = np.concatenate(pooled["scores"])
            table = genome_table(labels, scores, genomes)
            summary.append(
                {
                    "tag": key,
                    "seed": seed,
                    "split": "test_pooled",
                    "read_auc": fast_auc(labels, scores),
                    "genome_weighted_auc": genome_weighted_auc(
                        labels, scores, genomes
                    ),
                    **genome_level_metrics(table),
                }
            )
            tables.append(table.assign(tag=key, seed=seed, split="test_pooled"))
    result = pd.DataFrame(summary)
    result.to_csv(output_dir / "genome_summary.csv", index=False)
    pd.concat(tables).to_csv(output_dir / "genome_tables.csv", index=False)
    pd.DataFrame(curves).to_csv(output_dir / "aggregation_curves.csv", index=False)
    print("Saved genome-level results: " + str(output_dir), flush=True)
    return result
