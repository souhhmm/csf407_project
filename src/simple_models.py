import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from src.classifiers import (
    METRICS,
    MODELS,
    STOCHASTIC,
    fit_estimator,
    genome_bootstrap_interval,
    is_compatible,
    make_estimator,
    predict_scores,
    score_metrics,
    select_configuration,
)
from src.features import (
    REPRESENTATIONS,
    build_representations,
    count_words,
    load_representations,
    save_representations,
)
from src.plots import (
    plot_baseline_comparison,
    plot_generalisation,
    plot_model_heatmap,
)
from src.splits import EVALUATION_SPLITS, load_splits


POSITIVE_LABEL = "pathogenic"
BASELINES = ("gc_fraction", "ambiguous_letters", "composition_k1", "composition_k2")
CONTROL_PARAMS = {"C": 0.01}
CONTROL_REPEATS = 5
GROUP_COLUMNS = ["group", "model", "representation", "k"]


def _baseline_tables(sequences_by_split: Dict[str, List[str]]) -> dict:
    tables = {name: {} for name in BASELINES}
    for split, sequences in sequences_by_split.items():
        letters = count_words(sequences, 1).toarray()
        known = letters.sum(axis=1)
        safe = np.where(known > 0, known, 1)
        tables["gc_fraction"][split] = ((letters[:, 1] + letters[:, 2]) / safe)[:, None]
        tables["ambiguous_letters"][split] = (
            np.asarray([len(sequence) for sequence in sequences]) - known
        )[:, None]
        tables["composition_k1"][split] = normalize(letters, norm="l1")
        tables["composition_k2"][split] = normalize(
            count_words(sequences, 2).toarray(), norm="l1"
        )
    return tables


def _evaluate(estimator, model: str, tables: dict, context: dict) -> dict:
    row = {}
    for split in ("train",) + tuple(EVALUATION_SPLITS):
        scores = predict_scores(estimator, model, tables[split])
        labels = context["labels"][split]
        for name, value in score_metrics(labels, scores).items():
            row[split + "_" + name] = value
        if split != "train":
            low, high = genome_bootstrap_interval(
                labels,
                scores,
                context["genomes"][split],
                context["bootstrap"],
                context["seeds"][0],
            )
            row[split + "_auc_low"] = low
            row[split + "_auc_high"] = high
    return row


def _run_configuration(
    model: str,
    representation: str,
    k: Optional[int],
    group: str,
    tables: dict,
    context: dict,
) -> tuple:
    """Tune one model on validation data, then evaluate it for every seed.

    Selection sees only the training and validation tables; test tables are
    used afterwards, only for the setting that validation chose.
    """
    meta = {"group": group, "model": model, "representation": representation, "k": k}
    seeds = context["seeds"]
    n_jobs, trees = context["n_jobs"], context["trees"]
    train_labels = context["labels"]["train"]
    started = time.perf_counter()
    params, estimator, tuning = select_configuration(
        model,
        tables["train"],
        train_labels,
        tables["val"],
        context["labels"]["val"],
        seeds[0],
        n_jobs,
        trees,
    )
    final = []
    for number, seed in enumerate(seeds):
        if number > 0:
            if model not in STOCHASTIC:
                break
            estimator = make_estimator(
                model, params, seed, n_jobs, trees, sparse.issparse(tables["train"])
            )
            fit_estimator(estimator, model, tables["train"], train_labels)
        final.append(
            {
                **meta,
                "seed": seed,
                "params": json.dumps(params),
                **_evaluate(estimator, model, tables, context),
            }
        )
    print(
        "  "
        + group
        + " | "
        + representation
        + " | "
        + str(k)
        + " | "
        + model
        + ": val AUC "
        + format(final[0]["val_auc"], ".3f")
        + ", "
        + format(time.perf_counter() - started, ".1f")
        + " s",
        flush=True,
    )
    return [{**meta, **row} for row in tuning], final


def _run_control(representation: str, k: int, tables: dict, context: dict) -> list:
    """Train logistic regression on shuffled training labels as a leak alarm.

    Each repeat uses a different permutation; one permutation alone can land far
    from 0.5 when a single strong direction (such as G/C content) separates the
    classes, so the average and spread over repeats are what to read.
    """
    meta = {
        "group": "control",
        "model": "logistic_regression",
        "representation": representation,
        "k": k,
    }
    rows = []
    for number in range(CONTROL_REPEATS):
        seed = context["seeds"][0] + number
        shuffled = np.random.default_rng(seed).permutation(context["labels"]["train"])
        estimator = make_estimator(
            "logistic_regression",
            CONTROL_PARAMS,
            seed,
            context["n_jobs"],
            context["trees"],
            sparse.issparse(tables["train"]),
        )
        fit_estimator(estimator, "logistic_regression", tables["train"], shuffled)
        rows.append(
            {
                **meta,
                "seed": seed,
                "params": json.dumps(CONTROL_PARAMS),
                **_evaluate(estimator, "logistic_regression", tables, context),
            }
        )
    print(
        "  control | "
        + representation
        + " | "
        + str(k)
        + ": shuffled-label val AUC "
        + format(float(np.mean([row["val_auc"] for row in rows])), ".3f"),
        flush=True,
    )
    return rows


def summarize_results(final: pd.DataFrame) -> pd.DataFrame:
    """Average results over seeds.

    Args:
        final: One row per configuration and seed, with metric columns.

    Returns:
        One row per group, model, representation, and word size, with the mean
        and standard deviation of every metric across seeds, and the bootstrap
        interval bounds averaged across seeds.
    """
    intervals = [column for column in final if column.endswith(("_low", "_high"))]
    metrics = [
        column
        for column in final
        if column.endswith(tuple("_" + name for name in METRICS))
    ]
    grouped = final.groupby(GROUP_COLUMNS, dropna=False, sort=False)
    means = grouped[metrics].mean().add_suffix("_mean")
    deviations = grouped[metrics].std(ddof=0).add_suffix("_std")
    bounds = grouped[intervals].mean()
    params = grouped["params"].first()
    seeds = grouped["seed"].count().rename("seeds")
    return pd.concat([seeds, means, deviations, bounds, params], axis=1).reset_index()


def _table(rows: pd.DataFrame, columns: Sequence[str], headers: Sequence[str]) -> list:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in rows.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("n/a")
            elif isinstance(value, float):
                cells.append(format(value, ".3f"))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _report(summary: pd.DataFrame, manifest: dict, settings: dict) -> str:
    audit = manifest["split_audit"]
    lines = [
        "# Task 4: simple models on DeePaC train, validation, and test reads",
        "",
        "Representations are learned from the training reads only. Hyperparameters are chosen on the validation reads. The two test sets are scored only for the validation-chosen setting of each model. All splits come from different genomes.",
        "",
        "## Data",
        "",
        "- Reads per class: "
        + ", ".join(
            name + " = " + str(count) for name, count in manifest["per_class"].items()
        ),
        "- Genomes per split: "
        + ", ".join(
            name + " = " + str(count)
            for name, count in audit["genomes_per_split"].items()
        ),
        "- Genome overlap between splits (test_1 and test_2 are mates of the same fragments, so they share genomes by design): "
        + json.dumps(audit["genome_overlaps"]),
        "- Test read pairing: " + json.dumps(audit.get("paired_test_reads")),
        "- Exact repeated reads across splits: "
        + json.dumps(audit["exact_sequence_overlaps"]),
        "- Seeds: " + ", ".join(str(seed) for seed in settings["seeds"]),
        "",
        "## Baselines and controls",
        "",
        "Baselines use no word information (majority class), or only letter content. The shuffled-label control trains on randomly reassigned training labels, repeated with five different shuffles. Its average should sit near 0.5 AUC. A single shuffle can land away from 0.5 when one strong direction such as G/C content separates the classes, so read the average and the spread; a consistently high value would signal a leak.",
        "",
    ]
    reference = summary[summary["group"].isin(["baseline", "control"])].copy()
    reference["name"] = [
        (
            "majority class"
            if row.model == "majority"
            else (
                row.representation
                if row.group == "baseline"
                else "shuffled labels, " + row.representation + ", k=" + str(row.k)
            )
        )
        for row in reference.itertuples()
    ]
    lines.extend(
        _table(
            reference,
            [
                "group",
                "name",
                "val_auc_mean",
                "test_1_auc_mean",
                "test_2_auc_mean",
                "test_1_accuracy_mean",
            ],
            ["Group", "Name", "Val AUC", "Test 1 AUC", "Test 2 AUC", "Test 1 accuracy"],
        )
    )
    models = summary[summary["group"] == "model"].sort_values(
        "val_auc_mean", ascending=False
    )
    lines.extend(
        [
            "",
            "## Models ranked by validation AUC",
            "",
            "AUC is the chance that a random pathogenic read scores above a random nonpathogenic one (0.5 is chance). The intervals resample whole genomes within each class, because reads from one genome are not independent; their bounds are averaged over seeds.",
            "",
        ]
    )
    models = models.assign(
        test_1_interval=[
            format(low, ".3f") + "–" + format(high, ".3f")
            for low, high in zip(models["test_1_auc_low"], models["test_1_auc_high"])
        ]
    )
    lines.extend(
        _table(
            models.head(15),
            [
                "model",
                "representation",
                "k",
                "val_auc_mean",
                "test_1_auc_mean",
                "test_1_interval",
                "test_2_auc_mean",
                "train_auc_mean",
            ],
            [
                "Model",
                "Representation",
                "k",
                "Val AUC",
                "Test 1 AUC",
                "Test 1 95% interval",
                "Test 2 AUC",
                "Train AUC",
            ],
        )
    )
    lines.extend(
        [
            "",
            "The complete tables are `results_summary.csv` (averaged over seeds), `results.csv` (every seed), and `tuning.csv` (every setting tried, validation only).",
            "",
            "## Figures",
            "",
            "![Model and representation comparison](figures/model_heatmap.png)",
            "",
            "![Best model per representation against baselines](figures/baseline_comparison.png)",
            "",
            "![Validation versus test performance](figures/generalisation.png)",
            "",
            "## How to read these results",
            "",
            "- Labels describe the source bacteria, not whether a read causes disease.",
            "- test_1 and test_2 are the two mate reads of the same fragments (identical read ids), so they are two views of one test, not independent replicates.",
            "- Compare every model with the letter-composition baselines: performance close to them suggests composition, not word grammar, drives the result.",
            "- Selecting the best row by test score would be a form of tuning on the test set; use the validation ranking.",
            "- Thresholded metrics (accuracy, F1) use a 0.5 cutoff chosen without regard to species shift; AUC does not depend on it.",
            "",
        ]
    )
    return "\n".join(lines)


def run_simple_models(
    splits_dir: Path,
    output_dir: Path,
    feature_dir: Path,
    ks: Sequence[int] = (3, 4, 6),
    models: Sequence[str] = MODELS,
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    seeds: Sequence[int] = (42, 43, 44),
    workers: int = 1,
    n_jobs: int = 1,
    trees: int = 200,
    bootstrap: int = 100,
) -> dict:
    """Run baselines, controls, and simple models over every representation.

    Args:
        splits_dir: Directory written by prepare_splits.
        output_dir: Destination for tables, figures, and the report.
        feature_dir: Directory for cached learned representations.
        ks: Word sizes to compare.
        models: Models to run, from classifiers.MODELS.
        vector_size: Number of values in a word vector.
        window: Maximum number of neighbouring words on either side.
        epochs: Word2Vec passes over the training reads.
        seeds: Random settings; the first drives tuning and Word2Vec.
        workers: Word2Vec CPU workers. Values above 1 lose exact repeatability.
        n_jobs: CPU workers for random forests.
        trees: Number of trees per random forest.
        bootstrap: Genome-level bootstrap resamples per interval; 0 disables.

    Returns:
        The saved settings and runtime.

    Raises:
        ValueError: If a model name, seed list, or split is invalid.
    """
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError("Unknown models: " + ", ".join(sorted(unknown)))
    if not seeds:
        raise ValueError("Give at least one seed.")
    started = time.perf_counter()
    records, manifest = load_splits(splits_dir)
    names = list(records)
    sequences = {name: [row["sequence"] for row in records[name]] for name in names}
    context = {
        "labels": {
            name: np.asarray(
                [int(row["label"] == POSITIVE_LABEL) for row in records[name]]
            )
            for name in names
        },
        "genomes": {
            name: np.asarray([row["genome"] for row in records[name]]) for name in names
        },
        "seeds": list(seeds),
        "n_jobs": n_jobs,
        "trees": trees,
        "bootstrap": bootstrap,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_rows, final_rows = [], []

    def record(tuning: list, final: list):
        tuning_rows.extend(tuning)
        final_rows.extend(final)

    print("Baselines", flush=True)
    for baseline, tables in _baseline_tables(sequences).items():
        record(
            *_run_configuration(
                "logistic_regression", baseline, None, "baseline", tables, context
            )
        )
    majority = {name: np.zeros((len(sequences[name]), 1)) for name in names}
    record(*_run_configuration("majority", "none", None, "baseline", majority, context))
    for k in ks:
        print("Word size " + str(k), flush=True)
        settings = {
            "k": k,
            "vector_size": vector_size,
            "window": window,
            "epochs": epochs,
            "seed": seeds[0],
            "splits": manifest["csv_sha256"],
        }
        directory = feature_dir / ("k_" + str(k))
        representations = load_representations(directory, settings, k, names)
        if representations is None:
            representations = build_representations(
                sequences, k, vector_size, window, epochs, seeds[0], workers
            )
            save_representations(representations, directory, settings)
        else:
            print("  using saved representations: " + str(directory), flush=True)
        for representation in REPRESENTATIONS:
            tables = {
                name: representations["splits"][name][representation] for name in names
            }
            for model in models:
                if is_compatible(model, representation, tables["train"].shape[1]):
                    record(
                        *_run_configuration(
                            model, representation, k, "model", tables, context
                        )
                    )
            final_rows.extend(_run_control(representation, k, tables, context))
        del representations
    tuning = pd.DataFrame(tuning_rows)
    final = pd.DataFrame(final_rows)
    for table in (tuning, final):
        table["k"] = table["k"].astype("Int64")
    summary = summarize_results(final)
    tuning.to_csv(output_dir / "tuning.csv", index=False)
    final.to_csv(output_dir / "results.csv", index=False)
    summary.to_csv(output_dir / "results_summary.csv", index=False)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_model_heatmap(summary, figures / "model_heatmap.png")
    plot_baseline_comparison(summary, figures / "baseline_comparison.png")
    plot_generalisation(summary, figures / "generalisation.png")
    run_settings = {
        "scope": "task 4, simple models",
        "ks": list(ks),
        "models": list(models),
        "vector_size": vector_size,
        "window": window,
        "epochs": epochs,
        "seeds": list(seeds),
        "workers": workers,
        "n_jobs": n_jobs,
        "trees": trees,
        "bootstrap": bootstrap,
        "splits_dir": str(splits_dir),
        "splits_manifest": manifest,
        "software": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "gensim",
                "matplotlib",
            )
        },
        "python": platform.python_version(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(run_settings, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        _report(summary, manifest, run_settings), encoding="utf-8"
    )
    print("Saved results: " + str(output_dir), flush=True)
    return run_settings
