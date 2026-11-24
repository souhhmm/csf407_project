import itertools
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from src.classifiers import (
    METRICS,
    genome_bootstrap_interval,
    save_scores,
    score_metrics,
)
from src.features import encode_sequences, load_or_build_representations
from src.networks import (
    DEFAULT_MODELS,
    FIXED,
    GRIDS,
    NEURAL_MODELS,
    PRETRAINED_MODELS,
    DNABERT_PATH,
    count_parameters,
    make_network,
)
from src.neural_training import (
    ReadInputs,
    TableInputs,
    predict_scores,
    seed_everything,
    select_device,
    tokenize_reads,
    train_network,
)
from src.plots import plot_neural_comparison, plot_training_curves
from src.simple_models import POSITIVE_LABEL, _table, summarize_results
from src.splits import evaluation_splits, load_splits


CONTROL_REPEATS = 2
# How each model is described in the tables, in place of a representation.
REPRESENTATION_NAMES = {
    "mlp": "tfidf",
    "cnn": "one_hot_reads",
    "gru": "word2vec_tokens",
    "transformer": "word_tokens",
    "dnabert2": "bpe_tokens_pretrained",
    "dnabert2_scratch": "bpe_tokens_random_init",
}
USES_WORDS = {"mlp", "gru", "transformer"}


def _tag(meta: dict, seed: int) -> str:
    """Name a configuration and seed for its saved scores and weights."""
    return (
        "_".join(str(meta[key]) for key in ("group", "model", "representation", "k"))
        + "_s"
        + str(seed)
    )


def _evaluate(
    network: torch.nn.Module, inputs: dict, context: dict, labels: dict, tag: str
) -> dict:
    """Score every split with a trained network and add genome-level intervals."""
    row = {}
    kept = {}
    for split in ("train",) + tuple(context["eval_splits"]):
        scores = predict_scores(network, inputs[split], context["device"])
        if split != "train":
            kept[split] = scores
        for name, value in score_metrics(labels[split], scores).items():
            row[split + "_" + name] = value
        if split != "train":
            low, high = genome_bootstrap_interval(
                labels[split],
                scores,
                context["genomes"][split],
                context["bootstrap"],
                context["seeds"][0],
            )
            row[split + "_auc_low"] = low
            row[split + "_auc_high"] = high
    save_scores(context["score_dir"], tag, kept)
    if context["score_dir"] is not None:
        folder = context["score_dir"].parent / "weights"
        folder.mkdir(exist_ok=True)
        torch.save(network.state_dict(), folder / (tag + ".pt"))
    return row


def _train(
    model: str,
    params: dict,
    k: Optional[int],
    seed: int,
    setup: dict,
    context: dict,
    labels: np.ndarray,
) -> tuple:
    """Build and train one network from scratch for a seed."""
    seed_everything(seed)
    network = make_network(
        model,
        params,
        k,
        setup["features"],
        context["length"],
        setup.get("word_vectors"),
    ).to(context["device"])
    started = time.perf_counter()
    network, history = train_network(
        network,
        setup["inputs"]["train"],
        labels,
        setup["inputs"]["val"],
        context["labels"]["val"],
        FIXED[model]["lr"],
        seed,
        context["device"],
        context["max_epochs"],
        context["patience"],
        context["batch_size"],
        FIXED[model].get("warmup", 0),
    )
    return network, history, time.perf_counter() - started


def _run_configuration(
    model: str, k: Optional[int], setup: dict, context: dict, group: str = "model"
) -> tuple:
    """Tune one network on validation data, then evaluate it for every seed.

    Every grid setting is trained with the first seed and ranked by validation
    AUC. Only the winning setting is scored on test data, and only the winner is
    retrained with the remaining seeds. Early stopping also looks at validation
    data only.
    """
    meta = {
        "group": group,
        "model": model,
        "representation": REPRESENTATION_NAMES[model],
        "k": k,
    }
    seeds = context["seeds"]
    grid = GRIDS[model]
    names = list(grid)
    candidates = (
        [dict(zip(names, values)) for values in itertools.product(*grid.values())][
            : context["max_grid"]
        ]
        if group == "model"
        else [{name: grid[name][0] for name in names}]
    )
    labels = context["labels"]
    train_labels = (
        labels["train"]
        if group == "model"
        else np.random.default_rng(seeds[0]).permutation(labels["train"])
    )
    started = time.perf_counter()
    tuning, histories, best = [], [], None
    for params in candidates:
        network, history, seconds = _train(
            model, params, k, seeds[0], setup, context, train_labels
        )
        scores = predict_scores(network, setup["inputs"]["val"], context["device"])
        metrics = score_metrics(labels["val"], scores)
        tuning.append(
            {
                **meta,
                "params": json.dumps(params),
                "seed": seeds[0],
                "fit_seconds": seconds,
                "best_epoch": int(np.argmax([row["val_auc"] for row in history])) + 1,
                "epochs_run": len(history),
                "n_parameters": count_parameters(network),
                **{"val_" + name: value for name, value in metrics.items()},
            }
        )
        histories.extend(
            {**meta, "params": json.dumps(params), "seed": seeds[0], **row}
            for row in history
        )
        if best is None or metrics["auc"] > best[0] or np.isnan(best[0]):
            best = (metrics["auc"], params, network)
        else:
            del network
    _, params, network = best
    final = []
    repeats = seeds if group == "model" else seeds[:CONTROL_REPEATS]
    for number, seed in enumerate(repeats):
        if number > 0:
            shuffled = (
                train_labels
                if group == "model"
                else np.random.default_rng(seed).permutation(labels["train"])
            )
            network, history, _ = _train(
                model, params, k, seed, setup, context, shuffled
            )
            histories.extend(
                {**meta, "params": json.dumps(params), "seed": seed, **row}
                for row in history
            )
        final.append(
            {
                **meta,
                "seed": seed,
                "params": json.dumps(params),
                "n_parameters": count_parameters(network),
                **_evaluate(
                    network, setup["inputs"], context, labels, _tag(meta, seed)
                ),
            }
        )
    print(
        "  "
        + group
        + " | "
        + model
        + " | "
        + str(k)
        + ": val AUC "
        + format(final[0]["val_auc"], ".3f")
        + ", "
        + format(time.perf_counter() - started, ".1f")
        + " s",
        flush=True,
    )
    return tuning, final, histories


def _report(
    summary: pd.DataFrame,
    simple: Optional[pd.DataFrame],
    manifest: dict,
    settings: dict,
) -> str:
    audit = manifest["split_audit"]
    lines = [
        "# Task 4: neural models on DeePaC train, validation, and test reads",
        "",
        "The grid setting and the stopping epoch are chosen on validation reads only, and the test sets are scored once, for the validation-chosen setting. All splits come from different genomes.",
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
        "- Test read pairing: " + json.dumps(audit.get("paired_test_reads")),
        "- Seeds: " + ", ".join(str(seed) for seed in settings["seeds"]),
        "- Device: " + settings["device"],
        "",
        "## Models ranked by validation AUC",
        "",
        "The intervals resample whole genomes within each class; their bounds are averaged over seeds. The control rows train the CNN on shuffled training labels and should sit near 0.5.",
        "",
    ]
    models = summary[summary["group"] == "model"].sort_values(
        "val_auc_mean", ascending=False
    )
    models = models.assign(
        test_1_interval=[
            format(low, ".3f") + "–" + format(high, ".3f")
            for low, high in zip(models["test_1_auc_low"], models["test_1_auc_high"])
        ]
    )
    columns = [
        "model",
        "representation",
        "k",
        "n_parameters",
        "val_auc_mean",
        "test_1_auc_mean",
        "test_1_interval",
        "test_2_auc_mean",
        "train_auc_mean",
    ]
    headers = [
        "Model",
        "Representation",
        "k",
        "Parameters",
        "Val AUC",
        "Test 1 AUC",
        "Test 1 95% interval",
        "Test 2 AUC",
        "Train AUC",
    ]
    lines.extend(_table(models, columns, headers))
    controls = summary[summary["group"] == "control"]
    if len(controls):
        lines.extend(["", "### Shuffled-label control", ""])
        lines.extend(
            _table(controls, columns[:3] + columns[4:6], headers[:3] + headers[4:6])
        )
    if simple is not None:
        best = (
            simple[simple["group"] == "model"]
            .sort_values("val_auc_mean", ascending=False)
            .head(3)
        )
        lines.extend(
            [
                "",
                "## Best simple models, for comparison",
                "",
                "From `simple_models/results_summary.csv`, ranked the same way.",
                "",
            ]
        )
        lines.extend(
            _table(
                best.assign(n_parameters=np.nan),
                columns[:3] + columns[4:6] + columns[7:8],
                headers[:3] + headers[4:6] + headers[7:8],
            )
        )
    lines.extend(
        [
            "",
            "The complete tables are `results_summary.csv` (averaged over seeds), `results.csv` (every seed), `tuning.csv` (every setting tried, validation only) and `history.csv` (every epoch).",
            "",
            "## Figures",
            "",
            "![Neural models against simple models](figures/neural_comparison.png)",
            "",
            "![Validation AUC while training](figures/training_curves.png)",
            "",
            "## How to read these results",
            "",
            "- Labels describe the source bacteria, not whether a read causes disease.",
            "- test_1 and test_2 are the two mate reads of the same fragments, so they are two views of one test, not independent replicates.",
            "- The CNN, GRU and transformer see each read on both strands: strands are flipped at random while training and the two scores are averaged when predicting.",
            "- The mean-pooled word-vector rows of the simple models and the GRU share the same Word2Vec vectors, learned from training reads only.",
            "- Selecting the best row by test score would be tuning on the test set; use the validation ranking.",
            "- GPU training is not bit-for-bit repeatable, so reruns of the same seed can differ slightly. Seeds are for spread, not exact reproduction.",
            "- A gap between train and validation AUC means the network is memorising species-specific reads; the test sets are species the networks never saw.",
            "",
        ]
    )
    return "\n".join(lines)


def run_neural_models(
    splits_dir: Path,
    output_dir: Path,
    feature_dir: Path,
    simple_summary: Optional[Path] = None,
    ks: Sequence[int] = (3, 6),
    models: Sequence[str] = DEFAULT_MODELS,
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    seeds: Sequence[int] = (42, 43, 44),
    workers: int = 1,
    n_jobs: int = 1,
    bootstrap: int = 100,
    max_epochs: int = 15,
    patience: int = 3,
    batch_size: int = 256,
    device: str = "auto",
    train_limit: int = 0,
    fit_limit: int = 0,
    max_grid: int = 99,
    save_per_read: bool = True,
) -> dict:
    """Train small neural networks on the published splits and compare them.

    Args:
        splits_dir: Directory written by prepare_splits.
        output_dir: Destination for tables, figures, and the report.
        feature_dir: Directory for cached learned representations, shared with
            the simple models.
        simple_summary: results_summary.csv of the simple models, used for
            comparison when it exists.
        ks: Word sizes for the word-based networks.
        models: Networks to run, from networks.NEURAL_MODELS.
        vector_size: Number of values in a word vector.
        window: Maximum number of neighbouring words on either side.
        epochs: Word2Vec passes over the training reads.
        seeds: Random settings; the first drives tuning and Word2Vec.
        workers: Word2Vec CPU workers.
        n_jobs: CPU threads for torch.
        bootstrap: Genome-level bootstrap resamples per interval; 0 disables.
        max_epochs: Upper limit on passes over the training reads.
        patience: Epochs without validation improvement before stopping.
        batch_size: Reads per update.
        device: auto, cuda, mps, or cpu.
        train_limit: Train on only the first this-many (shuffled) reads; 0 keeps all.
        fit_limit: Learn weights and word vectors from only this many training
            reads; 0 uses every training read.
        max_grid: Largest number of grid settings to try per model.
        save_per_read: Whether to save per-read scores and final weights.

    Returns:
        The saved settings and runtime.

    Raises:
        ValueError: If a model name, seed list, or split is invalid.
    """
    unknown = set(models) - set(NEURAL_MODELS)
    if unknown:
        raise ValueError("Unknown models: " + ", ".join(sorted(unknown)))
    if not seeds:
        raise ValueError("Give at least one seed.")
    started = time.perf_counter()
    chosen = select_device(device)
    seed_everything(seeds[0], n_jobs)
    records, manifest = load_splits(splits_dir)
    if train_limit:
        records["train"] = records["train"][:train_limit]
    names = list(records)
    sequences = {name: [row["sequence"] for row in records[name]] for name in names}
    codes = {name: encode_sequences(sequences[name]) for name in names}
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
        "bootstrap": bootstrap,
        "eval_splits": evaluation_splits(manifest),
        "score_dir": output_dir / "scores" if save_per_read else None,
        "max_grid": max_grid,
        "device": chosen,
        "length": codes["train"].shape[1],
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
    }
    print("Training on " + str(chosen), flush=True)
    reads = {name: ReadInputs(codes[name], chosen) for name in names}
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_rows, final_rows, history_rows = [], [], []

    def record(result: tuple):
        tuning_rows.extend(result[0])
        final_rows.extend(result[1])
        history_rows.extend(result[2])
        pd.DataFrame(final_rows).to_csv(output_dir / "results_partial.csv", index=False)

    if "cnn" in models:
        setup = {"inputs": reads, "features": 0}
        record(_run_configuration("cnn", None, setup, context))
        record(_run_configuration("cnn", None, setup, context, group="control"))
    pretrained = [name for name in models if name in PRETRAINED_MODELS]
    if pretrained:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(DNABERT_PATH, trust_remote_code=True)
        started_tokens = time.perf_counter()
        tokens = {
            name: tokenize_reads(sequences[name], tokenizer, chosen) for name in names
        }
        print(
            "Tokenized reads in "
            + format(time.perf_counter() - started_tokens, ".0f")
            + " s",
            flush=True,
        )
        for name in pretrained:
            setup = {"inputs": tokens, "features": 0}
            record(_run_configuration(name, None, setup, context))
        del tokens
    for k in ks:
        if not USES_WORDS & set(models):
            break
        print("Word size " + str(k), flush=True)
        representations = None
        if {"mlp", "gru"} & set(models):
            representations = load_or_build_representations(
                sequences,
                k,
                feature_dir,
                manifest["csv_sha256"],
                vector_size,
                window,
                epochs,
                seeds[0],
                workers,
                fit_limit,
                train_limit,
            )
        if "mlp" in models:
            tables = {
                name: TableInputs(representations["splits"][name]["tfidf"], chosen)
                for name in names
            }
            setup = {"inputs": tables, "features": 4**k}
            record(_run_configuration("mlp", k, setup, context))
        if "gru" in models:
            setup = {
                "inputs": reads,
                "features": 0,
                "word_vectors": representations["word_vectors"],
            }
            record(_run_configuration("gru", k, setup, context))
        if "transformer" in models:
            record(
                _run_configuration(
                    "transformer", k, {"inputs": reads, "features": 0}, context
                )
            )
        del representations
    tuning = pd.DataFrame(tuning_rows)
    final = pd.DataFrame(final_rows)
    history = pd.DataFrame(history_rows)
    for table in (tuning, final, history):
        table["k"] = table["k"].astype("Int64")
    summary = summarize_results(final)
    counts = final.groupby(["group", "model", "representation", "k"], dropna=False)[
        "n_parameters"
    ].first()
    summary = summary.merge(
        counts.reset_index(), on=["group", "model", "representation", "k"], how="left"
    )
    simple = (
        pd.read_csv(simple_summary).astype({"k": "Int64"})
        if simple_summary is not None and Path(simple_summary).exists()
        else None
    )
    tuning.to_csv(output_dir / "tuning.csv", index=False)
    final.to_csv(output_dir / "results.csv", index=False)
    summary.to_csv(output_dir / "results_summary.csv", index=False)
    history.to_csv(output_dir / "history.csv", index=False)
    (output_dir / "results_partial.csv").unlink(missing_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plot_neural_comparison(summary, simple, figures / "neural_comparison.png")
    plot_training_curves(history, figures / "training_curves.png")
    run_settings = {
        "scope": "task 4, neural models",
        "ks": list(ks),
        "models": list(models),
        "vector_size": vector_size,
        "window": window,
        "word2vec_epochs": epochs,
        "seeds": list(seeds),
        "workers": workers,
        "n_jobs": n_jobs,
        "bootstrap": bootstrap,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "train_limit": train_limit,
        "fit_limit": fit_limit,
        "max_grid": max_grid,
        "device": str(chosen),
        "fixed_settings": FIXED,
        "grids": GRIDS,
        "splits_dir": str(splits_dir),
        "splits_manifest": manifest,
        "software": {
            name: version(name)
            for name in ("numpy", "pandas", "torch", "gensim", "matplotlib")
        },
        "python": platform.python_version(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(run_settings, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        _report(summary, simple, manifest, run_settings), encoding="utf-8"
    )
    print("Saved results: " + str(output_dir), flush=True)
    return run_settings
