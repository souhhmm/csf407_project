import json
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.data import file_digest, load_dataset
from src.plots import (
    TITLES,
    plot_k_comparison,
    plot_sequence_spaces,
    plot_sequence_statistics,
    plot_word_frequencies,
)
from src.representations import (
    compare_contexts,
    fit_representations,
    nearest_words,
    project_to_two_dimensions,
)
from src.sequences import sequence_statistics, word_segments


def _word_tables(representations: dict, records: list) -> tuple:
    counts = representations["counts"]
    words = representations["words"]
    frequencies = np.asarray(counts.sum(axis=0)).ravel().astype(int)
    table = pd.DataFrame(
        {
            "word": words,
            "count": frequencies,
            "fraction": frequencies / frequencies.sum(),
            "sequences_containing_word": counts.getnnz(axis=0),
        }
    ).sort_values(["count", "word"], ascending=[False, True])
    by_class = []
    labels = np.asarray([record["label"] for record in records])
    for label in sorted(set(labels)):
        group = counts[labels == label]
        total = np.asarray(group.sum(axis=0)).ravel().astype(int)
        by_class.append(
            pd.DataFrame(
                {
                    "label": label,
                    "word": words,
                    "count": total,
                    "fraction": (
                        total / total.sum() if total.sum() else np.zeros(len(total))
                    ),
                    "sequences_containing_word": group.getnnz(axis=0),
                }
            )
        )
    return (
        table,
        pd.concat(by_class, ignore_index=True),
        Counter(dict(zip(words, frequencies.tolist()))),
    )


def _save_representations(
    representations: dict, directory: Path, row_ids: pd.DataFrame
):
    row_ids.to_csv(directory / "row_ids.csv", index=False)
    for name in ("counts", "tfidf"):
        sparse.save_npz(directory / (name + ".npz"), representations[name])
    for name in ("idf", "word_vectors", "mean_vectors", "weighted_vectors"):
        np.save(directory / (name + ".npy"), representations[name], allow_pickle=False)
    (directory / "vocabulary.json").write_text(
        json.dumps(representations["words"], indent=2) + "\n", encoding="utf-8"
    )
    representations["model"].save(str(directory / "word2vec.model"))


def _number(value):
    return "not available" if value is None else format(value, ".3f")


def _write_report(summary: dict, output_dir: Path):
    rows = [
        "# DNA/RNA as a language: steps 1–3",
        "",
        "This report explores sequences, words, and their numeric representations. Class labels are used for descriptive tables and chart colours only. No disease-class prediction model is trained.",
        "",
        "## The input",
        "",
        "- Sequences analysed: " + str(summary["sequence_count"]),
        "- Alphabet: " + summary["alphabet"].upper(),
        "- Class counts: "
        + ", ".join(
            label + " = " + str(count)
            for label, count in summary["class_counts"].items()
        ),
        "- Length range: "
        + str(summary["minimum_length"])
        + " to "
        + str(summary["maximum_length"])
        + " letters",
        "- Sequences with unknown letters: "
        + str(summary["sequences_with_ambiguous_letters"]),
        "- Duplicate input rows removed: "
        + str(summary["input_audit"]["duplicate_rows_removed"]),
        "",
        "![Letter proportions](figures/sequence_statistics.png)",
        "",
        "GC fraction means the proportion of known letters that are G or C. Unknown letters remain in the sequence, but words containing them are skipped. Words on opposite sides of an unknown letter do not become neighbours during learning.",
        "",
        "## Comparing word sizes",
        "",
        "Words overlap: ATGCA produces ATG, TGC, and GCA for size 3. Longer words describe more specific patterns, but each individual sequence contains a smaller fraction of the possible vocabulary.",
        "",
        "| Word size | Words observed | Possible words | Total occurrences | Single-occurrence words | Empty sequences |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in summary["word_sizes"]:
        if result["status"] == "complete":
            rows.append(
                "| "
                + " | ".join(
                    str(result[field])
                    for field in (
                        "k",
                        "observed_words",
                        "possible_words",
                        "total_occurrences",
                        "single_occurrence_words",
                        "empty_sequences",
                    )
                )
                + " |"
            )
        else:
            rows.append(
                "| " + str(result["k"]) + " | No usable words | — | — | — | — |"
            )
    if any(result["status"] == "complete" for result in summary["word_sizes"]):
        rows.extend(["", "![Vocabulary comparison](figures/k_comparison.png)"])
    rows.extend(
        [
            "",
            "## How sequences become numbers",
            "",
            "1. **Bag-of-Words** counts occurrences of each word. Word order is discarded.",
            "2. **TF-IDF** gives larger weight to words occurring in fewer sequences, then scales each sequence's weights to comparable size. It also discards word order.",
            "3. **Word2Vec** learns word vectors by trying to distinguish observed neighbours from randomly chosen words. Class labels are never given to this learning process.",
            "4. **Whole-sequence vectors** average the word vectors, either counting every occurrence equally or weighting distinct words by TF-IDF. Both averages discard the final sequence's word order.",
            "",
            "The charts reduce these representations to two coordinates. Nearby points have similar projected representations. The percentage in each chart describes how much variation is shown; the remaining variation is not visible. Apparent groups are exploratory observations, not evidence that a classifier works.",
        ]
    )
    for result in summary["word_sizes"]:
        k = result["k"]
        prefix = "k_" + str(k)
        rows.extend(["", "## " + str(k) + "-letter words", ""])
        if result["status"] != "complete":
            rows.append(result["reason"])
            continue
        rows.extend(
            [
                "- Most common word: "
                + result["most_common_word"]
                + " ("
                + str(result["most_common_count"])
                + " occurrences)",
                "- Least common observed word: "
                + result["least_common_word"]
                + " ("
                + str(result["least_common_count"])
                + " occurrences)",
                "- Windows skipped because they contain unknown letters: "
                + str(result["skipped_windows"]),
                "- Average distinct words per sequence: "
                + format(result["mean_distinct_words"], ".1f"),
                "",
                "![Word frequencies](" + prefix + "/word_frequencies.png)",
                "",
                "![Sequence representations](" + prefix + "/sequence_spaces.png)",
                "",
                "### Do similar surroundings produce similar word vectors?",
                "",
                "We count neighbouring words around a random sample of sequences, compare these surroundings for word pairs, and compare their learned vectors. The rank correlation ranges from -1 to 1: positive values mean more similar surroundings tend to accompany more similar learned vectors. A shuffled comparison assigns learned vectors to random words as a reference.",
                "",
                "| Surroundings examined | Word pairs | Rank correlation | Shuffled reference |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name, caption in (
            ("standard", "Nearby words"),
            ("nonoverlapping", "Words sharing no sequence positions"),
        ):
            context = result["context_comparison"][name]
            rows.append(
                "| "
                + caption
                + " | "
                + str(context["pairs"])
                + " | "
                + _number(context["correlation"])
                + " | "
                + _number(context["shuffled_correlation"])
                + " |"
            )
        rows.extend(
            [
                "",
                "The second comparison excludes overlapping neighbours and uses a wider neighbourhood. It helps inspect the effect of shared letters; it is a diagnostic, not a proof of biological meaning. See [nearest words]("
                + prefix
                + "/nearest_words.csv), [common words]("
                + prefix
                + "/common_words.csv), and [rare observed words]("
                + prefix
                + "/rare_words.csv).",
            ]
        )
        if not result["has_training_context"]:
            rows.extend(
                [
                    "",
                    "**No segment contained neighbouring words. The word vectors retain their initial random values and should not be interpreted as learned context.**",
                ]
            )
    rows.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Overlapping words share letters by construction. Similar word vectors alone do not demonstrate that the computer has learned biological grammar.",
            "- Class differences can reflect species, letter composition, or sequence length. These charts do not establish the cause of a difference.",
            "- Zero vectors indicate sequences with no usable words for that word size. They have no learned representation and may coincide in a chart.",
            "- Rare-word tables include words actually observed. Words absent from the sample are counted in vocabulary coverage, not listed as observed rare words.",
            "- The context comparisons use at most 200 randomly selected sequences and 256 frequent words. One shuffled reference is an exploratory comparison, not a statistical significance test.",
        ]
    )
    if summary["source_manifest"] is not None:
        rows.extend(
            [
                "- DeePaC supplies simulated 250-letter reads from bacterial genomes. The source organisms carry the labels; individual fragments do not necessarily cause disease.",
                "- The prepared sample uses the published validation files for this exploratory work. The published test files remain unused. If the project later grows to step 4, use the published training data for learning and reserve untouched test data for the final evaluation.",
                "- The sample is balanced by class, not by bacterial species. Repeated sequences are checked within sampled candidates, not across every source genome.",
                "",
                "Source: [DeePaC dataset](https://zenodo.org/records/3678563), [research paper](https://doi.org/10.1093/bioinformatics/btz541).",
            ]
        )
    rows.extend(
        [
            "",
            "## Saved files",
            "",
            "Each word-size directory contains the full frequency tables, per-sequence word statistics, four numeric sequence representations, word vectors, word neighbours, context comparisons, and chart coordinates. `row_ids.csv` gives the sequence order; `vocabulary.json` gives the column or word-vector order. `summary.json` records settings, fingerprints, software versions, and projection information.",
            "",
            "Runtime on this machine: "
            + format(summary["elapsed_seconds"], ".1f")
            + " seconds, excluding dataset preparation.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(rows), encoding="utf-8")


def analyze_dataset(
    input_path: Path,
    output_dir: Path,
    ks: tuple = (3, 4, 6),
    alphabet: str = "dna",
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    seed: int = 42,
) -> dict:
    """Run the sequence-language exploration through project step 3.

    Args:
        input_path: CSV with sequence IDs, sequences, and labels.
        output_dir: Directory for the report, charts, and numeric representations.
        ks: Word sizes to compare.
        alphabet: Either dna or rna.
        vector_size: Number of values in each learned word vector.
        window: Maximum neighbouring-word distance used during learning.
        epochs: Number of passes used to learn word vectors.
        seed: Fixed random setting.

    Returns:
        A JSON-compatible summary of the completed exploration.

    Raises:
        ValueError: If settings or the input data are invalid.
    """
    if not ks or any(k < 1 for k in ks) or min(vector_size, window, epochs) < 1:
        raise ValueError(
            "Word sizes, vector size, window, and passes must all be positive."
        )
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("The random setting must be between 0 and 4294967295.")
    started = time.perf_counter()
    records, audit = load_dataset(input_path, alphabet)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    row_ids = pd.DataFrame(
        [{"id": record["id"], "label": record["label"]} for record in records]
    )
    stats = pd.DataFrame(
        [
            {
                "id": record["id"],
                "label": record["label"],
                **sequence_statistics(record["sequence"], alphabet),
            }
            for record in records
        ]
    )
    stats.to_csv(output_dir / "sequence_statistics.csv", index=False)
    stats.groupby("label").agg(
        sequences=("id", "count"),
        minimum_length=("length", "min"),
        maximum_length=("length", "max"),
        mean_length=("length", "mean"),
        mean_gc_fraction=("gc_fraction", "mean"),
        ambiguous_letters=("ambiguous_letters", "sum"),
    ).to_csv(output_dir / "class_statistics.csv")
    plot_sequence_statistics(stats, alphabet, figures / "sequence_statistics.png")
    manifest_path = input_path.parent / "manifest.json"
    source_manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("csv_sha256") == file_digest(input_path):
            source_manifest = manifest
    summary = {
        "scope": "project steps 1 through 3 only",
        "input_path": str(input_path.resolve()),
        "input_sha256": file_digest(input_path),
        "input_audit": audit,
        "source_manifest": source_manifest,
        "alphabet": alphabet,
        "sequence_count": len(records),
        "class_counts": dict(Counter(record["label"] for record in records)),
        "minimum_length": int(stats["length"].min()),
        "maximum_length": int(stats["length"].max()),
        "sequences_with_ambiguous_letters": int((stats["ambiguous_letters"] > 0).sum()),
        "settings": {
            "ks": list(dict.fromkeys(ks)),
            "vector_size": vector_size,
            "window": window,
            "epochs": epochs,
            "seed": seed,
            "workers": 1,
        },
        "software": {
            package: version(package)
            for package in (
                "numpy",
                "scipy",
                "pandas",
                "scikit-learn",
                "gensim",
                "matplotlib",
            )
        },
        "python_version": platform.python_version(),
        "word_sizes": [],
    }
    for k in dict.fromkeys(ks):
        print("Analysing " + str(k) + "-letter words...", flush=True)
        segments_by_sequence = [
            word_segments(record["sequence"], k, alphabet) for record in records
        ]
        documents = [
            [word for segment in segments for word in segment]
            for segments in segments_by_sequence
        ]
        if not any(documents):
            summary["word_sizes"].append(
                {
                    "k": k,
                    "status": "skipped",
                    "reason": "No usable words of this size can be formed.",
                }
            )
            continue
        directory = output_dir / ("k_" + str(k))
        directory.mkdir(exist_ok=True)
        segments = [segment for group in segments_by_sequence for segment in group]
        print("  learning word vectors on the CPU", flush=True)
        representations = fit_representations(
            documents, segments, vector_size, window, epochs, seed
        )
        _save_representations(representations, directory, row_ids)
        frequencies, class_frequencies, counts = _word_tables(representations, records)
        frequencies.to_csv(directory / "word_frequencies.csv", index=False)
        class_frequencies.to_csv(
            directory / "word_frequencies_by_class.csv", index=False
        )
        frequencies.head(20).to_csv(directory / "common_words.csv", index=False)
        rare = frequencies.sort_values(["count", "word"])
        rare.head(20).to_csv(directory / "rare_words.csv", index=False)
        word_stats = pd.DataFrame(
            [
                {
                    "id": record["id"],
                    "label": record["label"],
                    "words": len(document),
                    "distinct_words": len(set(document)),
                    "skipped_windows": max(0, len(record["sequence"]) - k + 1)
                    - len(document),
                }
                for record, document in zip(records, documents)
            ]
        )
        word_stats.to_csv(directory / "word_statistics.csv", index=False)
        nearest_words(representations["model"], counts).to_csv(
            directory / "nearest_words.csv", index=False
        )
        projections = {}
        projection_info = {}
        for name in TITLES:
            projections[name], projection_info[name] = project_to_two_dimensions(
                representations[name], seed
            )
            pd.DataFrame(
                {
                    "id": row_ids["id"],
                    "label": row_ids["label"],
                    "coordinate_1": projections[name][:, 0],
                    "coordinate_2": projections[name][:, 1],
                }
            ).to_csv(directory / ("projection_" + name + ".csv"), index=False)
        plot_sequence_spaces(
            projections,
            row_ids["label"].tolist(),
            projection_info,
            k,
            directory / "sequence_spaces.png",
        )
        plot_word_frequencies(frequencies, k, directory / "word_frequencies.png")
        context_summaries = {}
        for nonoverlapping, name in ((False, "standard"), (True, "nonoverlapping")):
            table, context_summaries[name] = compare_contexts(
                segments_by_sequence,
                representations["words"],
                representations["word_vectors"],
                counts,
                k,
                window,
                seed,
                nonoverlapping=nonoverlapping,
            )
            table.to_csv(directory / ("context_pairs_" + name + ".csv"), index=False)
        probabilities = frequencies["fraction"].to_numpy()
        summary["word_sizes"].append(
            {
                "k": k,
                "status": "complete",
                "observed_words": len(frequencies),
                "possible_words": 4**k,
                "vocabulary_coverage": len(frequencies) / (4**k),
                "mean_sequence_coverage": float(
                    word_stats["distinct_words"].mean() / (4**k)
                ),
                "mean_distinct_words": float(word_stats["distinct_words"].mean()),
                "total_occurrences": int(word_stats["words"].sum()),
                "skipped_windows": int(word_stats["skipped_windows"].sum()),
                "empty_sequences": int((word_stats["words"] == 0).sum()),
                "single_occurrence_words": int((frequencies["count"] == 1).sum()),
                "word_entropy_bits": float(
                    -(probabilities * np.log2(probabilities)).sum()
                ),
                "most_common_word": str(frequencies.iloc[0]["word"]),
                "most_common_count": int(frequencies.iloc[0]["count"]),
                "least_common_word": str(rare.iloc[0]["word"]),
                "least_common_count": int(rare.iloc[0]["count"]),
                "has_training_context": any(len(segment) > 1 for segment in segments),
                "projections": projection_info,
                "context_comparison": context_summaries,
            }
        )
        print(
            "  saved representations, tables, and charts for size " + str(k), flush=True
        )
    completed = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in ("projections", "context_comparison")
            }
            for result in summary["word_sizes"]
            if result["status"] == "complete"
        ]
    )
    if len(completed):
        completed.to_csv(output_dir / "k_comparison.csv", index=False)
        plot_k_comparison(completed, figures / "k_comparison.png")
    summary["elapsed_seconds"] = time.perf_counter() - started
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_report(summary, output_dir)
    print("Report: " + str(output_dir / "report.md"), flush=True)
    return summary
