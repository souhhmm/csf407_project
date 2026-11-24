import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.classifiers import fast_auc
from src.features import KNOWN_RUN, all_words, count_words, fit_word_vectors
from src.splits import load_splits


POSITIVE_LABEL = "pathogenic"
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement_word(word: str) -> str:
    """Return the opposite-strand spelling of a word."""
    return word.translate(COMPLEMENT)[::-1]


def genome_frequencies(
    sequences: List[str], genomes: np.ndarray, labels: np.ndarray, k: int
) -> tuple:
    """Relative word frequency of every genome, from all its sampled reads.

    Args:
        sequences: Reads.
        genomes: Source genome of each read.
        labels: 1 for pathogenic reads, 0 otherwise.
        k: Word size.

    Returns:
        Genome names, genome labels, and a genomes-by-words table of frequencies
        (each row sums to one).
    """
    names, inverse = np.unique(genomes, return_inverse=True)
    counts = count_words(sequences, k)
    grouping = sparse.csr_matrix(
        (np.ones(len(inverse)), (inverse, np.arange(len(inverse)))),
        shape=(len(names), len(inverse)),
    )
    table = np.asarray((grouping @ counts).todense(), dtype=np.float64)
    table /= table.sum(axis=1, keepdims=True)
    genome_labels = np.zeros(len(names), dtype=int)
    genome_labels[inverse] = labels
    return names, genome_labels, table


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust p-values to control the false-discovery rate."""
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / (np.arange(len(p_values)) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def genome_word_effects(
    table: np.ndarray, genome_labels: np.ndarray, words: Sequence[str]
) -> pd.DataFrame:
    """Compare every word's frequency between pathogenic and nonpathogenic genomes.

    The genome, not the read, is the unit, so a word counts as different only if
    the genomes of one class tend to use it more than the genomes of the other.

    Args:
        table: Genomes-by-words frequencies.
        genome_labels: 1 for pathogenic genomes, 0 otherwise.
        words: Column names.

    Returns:
        One row per word with the genome-level AUC (above 0.5 means more common
        in pathogenic genomes), log2 ratio of class means (each genome weighted
        equally), Mann-Whitney p-value and its false-discovery-adjusted value.
    """
    positive, negative = table[genome_labels == 1], table[genome_labels == 0]
    test = mannwhitneyu(positive, negative, axis=0)
    auc = test.statistic / (len(positive) * len(negative))
    ratio = np.log2((positive.mean(axis=0) + 1e-12) / (negative.mean(axis=0) + 1e-12))
    return pd.DataFrame(
        {
            "word": list(words),
            "genome_auc": auc,
            "log2_ratio": ratio,
            "p_value": test.pvalue,
            "q_value": benjamini_hochberg(test.pvalue),
            "mean_frequency_pathogenic": positive.mean(axis=0),
            "mean_frequency_nonpathogenic": negative.mean(axis=0),
        }
    )


def replication(discovery: pd.DataFrame, held_out: pd.DataFrame, top: int = 50) -> dict:
    """Check whether word effects found on training genomes hold on unseen genomes.

    Args:
        discovery: Result of genome_word_effects on training genomes.
        held_out: The same on validation and test genomes.
        top: Number of most extreme words per direction to check.

    Returns:
        Spearman correlation of effect sizes, and the share of the top words
        whose direction is the same on held-out genomes.
    """
    rho = spearmanr(discovery["genome_auc"], held_out["genome_auc"]).statistic
    order = np.argsort(discovery["genome_auc"].to_numpy())
    chosen = {
        "more_in_nonpathogenic": order[:top],
        "more_in_pathogenic": order[-top:],
    }
    agree = {}
    for name, rows in chosen.items():
        sign = np.sign(discovery["genome_auc"].to_numpy()[rows] - 0.5)
        other = np.sign(held_out["genome_auc"].to_numpy()[rows] - 0.5)
        agree[name] = float((sign == other).mean())
    return {"spearman_genome_auc": float(rho), "top_direction_agreement": agree}


def read_gc(sequences: List[str]) -> np.ndarray:
    """Return the fraction of known letters that are G or C in each read."""
    letters = count_words(sequences, 1).toarray()
    return (letters[:, 1] + letters[:, 2]) / np.maximum(letters.sum(axis=1), 1)


def linear_importances(
    train_features: sparse.csr_matrix, train_labels: np.ndarray, words: Sequence[str]
) -> pd.DataFrame:
    """Fit a logistic regression and report its weights on standardised frequencies."""
    scaler = StandardScaler(with_mean=False).fit(train_features)
    model = LogisticRegression(C=0.01, max_iter=300, tol=1e-3).fit(
        scaler.transform(train_features), train_labels
    )
    return pd.DataFrame({"word": list(words), "coefficient": model.coef_[0]})


def forest_importances(
    train_features: sparse.csr_matrix,
    train_labels: np.ndarray,
    words: Sequence[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Fit a random forest on word frequencies and report impurity importances."""
    forest = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed,
    ).fit(train_features, train_labels)
    return pd.DataFrame(
        {"word": list(words), "importance": forest.feature_importances_}
    )


def strided_corpus(sequences: List[str], k: int):
    """Yield non-overlapping words, one sentence per reading frame of each read."""
    for sequence in sequences:
        for run in KNOWN_RUN.findall(sequence):
            for frame in range(k):
                words = [run[i : i + k] for i in range(frame, len(run) - k + 1, k)]
                if len(words) >= 2:
                    yield words


def rc_similarity(vectors: np.ndarray, k: int, seed: int = 42) -> dict:
    """Measure whether a word's vector is close to its reverse complement's vector.

    Reads come from either strand, so a word and its reverse complement play the
    same role; vectors that capture meaning should reflect that.

    Args:
        vectors: Word vectors in all_words(k) order.
        k: Word size.
        seed: Fixed random setting.

    Returns:
        Mean cosine similarity of word and reverse complement, of random word
        pairs, and the fraction of words whose reverse complement is their
        nearest neighbour (reverse-complement palindromes excluded).
    """
    words = all_words(k)
    index = {word: number for number, word in enumerate(words)}
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    partner = np.asarray([index[reverse_complement_word(word)] for word in words])
    own = (unit * unit[partner]).sum(axis=1)
    rng = np.random.default_rng(seed)
    other = (unit * unit[rng.permutation(len(words))]).sum(axis=1)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -2)
    keep = partner != np.arange(len(words))
    nearest = similarity.argmax(axis=1)
    return {
        "rc_cosine_mean": float(own[keep].mean()),
        "random_pair_cosine_mean": float(other.mean()),
        "rc_is_nearest_neighbour": float((nearest[keep] == partner[keep]).mean()),
    }


def embedding_tests(
    train_sequences: List[str], k: int, vector_size: int, seed: int, workers: int
) -> Dict[str, dict]:
    """Run the reverse-complement test on three kinds of word vectors.

    Args:
        train_sequences: Training reads.
        k: Word size.
        vector_size: Number of values in a word vector.
        seed: Fixed random setting.
        workers: Word2Vec CPU workers.

    Returns:
        Test results for overlapping Word2Vec, non-overlapping (strided)
        Word2Vec, and an SVD embedding of word co-occurrence counts.
    """
    from gensim.models import Word2Vec

    results = {}
    overlapping, _ = fit_word_vectors(
        train_sequences, k, vector_size, 5, 5, seed, workers
    )
    results["word2vec_overlapping"] = rc_similarity(overlapping, k, seed)
    model = Word2Vec(
        sentences=list(strided_corpus(train_sequences[:50000], k)),
        vector_size=vector_size,
        window=5,
        min_count=1,
        sg=1,
        negative=5,
        epochs=5,
        workers=workers,
        seed=seed,
    )
    strided = np.zeros((4**k, vector_size), dtype=np.float32)
    for number, word in enumerate(all_words(k)):
        if word in model.wv.key_to_index:
            strided[number] = model.wv[word]
    results["word2vec_strided"] = rc_similarity(strided, k, seed)
    codes = count_words(train_sequences[:50000], k)
    cooccurrence = np.log1p((codes.T @ codes).toarray().astype(np.float64))
    u, s, _ = np.linalg.svd(cooccurrence, full_matrices=False)
    results["count_svd_cooccurrence"] = rc_similarity(
        u[:, :vector_size] * np.sqrt(s[:vector_size]), k, seed
    )
    return results


def run_interpretation(
    splits_dir: Path,
    output_dir: Path,
    ks: Sequence[int] = (3, 4, 6),
    fit_reads: int = 400000,
    seed: int = 42,
    workers: int = 16,
) -> dict:
    """Per-genome word analysis, model importances, and embedding checks.

    Args:
        splits_dir: Directory written by prepare_splits.
        output_dir: Destination for tables and the summary.
        ks: Word sizes for the per-genome word tests.
        fit_reads: Training reads used for the importance models and embeddings.
        seed: Fixed random setting.
        workers: Word2Vec CPU workers.

    Returns:
        The summary that was saved.
    """
    records, _ = load_splits(splits_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def arrays(names: Sequence[str]) -> tuple:
        rows = [row for name in names for row in records[name]]
        return (
            [row["sequence"] for row in rows],
            np.asarray([row["genome"] for row in rows]),
            np.asarray([int(row["label"] == POSITIVE_LABEL) for row in rows]),
        )

    train = arrays(["train"])
    held = arrays(["val", "test_1"])
    summary: dict = {"fit_reads": fit_reads, "word_effects": {}}
    for k in ks:
        words = all_words(k)
        _, y_train, f_train = genome_frequencies(*train, k)
        _, y_held, f_held = genome_frequencies(*held, k)
        discovery = genome_word_effects(f_train, y_train, words)
        held_out = genome_word_effects(f_held, y_held, words)
        merged = discovery.merge(
            held_out[["word", "genome_auc", "q_value"]],
            on="word",
            suffixes=("", "_held_out"),
        )
        merged.sort_values("genome_auc").to_csv(
            output_dir / ("word_effects_k" + str(k) + ".csv"), index=False
        )
        summary["word_effects"]["k" + str(k)] = {
            **replication(discovery, held_out),
            "genomes_train_nonpathogenic_pathogenic": [
                int((y_train == 0).sum()),
                int((y_train == 1).sum()),
            ],
            "genomes_held_out_nonpathogenic_pathogenic": [
                int((y_held == 0).sum()),
                int((y_held == 1).sum()),
            ],
            "significant_in_train_q05": int((discovery["q_value"] < 0.05).sum()),
        }
    sequences, _, labels = train
    gc_tables = []
    for name, (seqs, genomes, lab) in {"train": train, "held_out": held}.items():
        frame = pd.DataFrame({"genome": genomes, "gc": read_gc(seqs), "label": lab})
        grouped = frame.groupby("genome").agg(gc=("gc", "mean"), label=("label", "first"))
        gc_tables.append(grouped.assign(set=name).reset_index())
    gc_table = pd.concat(gc_tables)
    gc_table.to_csv(output_dir / "genome_gc.csv", index=False)
    summary["genome_gc_auc"] = {
        name: fast_auc(rows["label"].to_numpy(), rows["gc"].to_numpy())
        for name, rows in gc_table.groupby("set")
    }
    subset = slice(0, fit_reads)
    counts = count_words(sequences[subset], 6)
    frequencies = sparse.csr_matrix(counts.multiply(1 / counts.sum(axis=1)))
    linear = linear_importances(frequencies, labels[subset], all_words(6))
    linear.sort_values("coefficient").to_csv(
        output_dir / "logistic_coefficients_k6.csv", index=False
    )
    counts4 = count_words(sequences[subset], 4)
    freq4 = sparse.csr_matrix(counts4.multiply(1 / counts4.sum(axis=1)))
    forest = forest_importances(freq4, labels[subset], all_words(4), seed)
    forest.sort_values("importance", ascending=False).to_csv(
        output_dir / "forest_importances_k4.csv", index=False
    )
    effects = pd.read_csv(output_dir / "word_effects_k6.csv").set_index("word")
    summary["lr_coefficient_vs_genome_effect_spearman_k6"] = float(
        spearmanr(
            linear.set_index("word")["coefficient"].loc[effects.index],
            effects["genome_auc"],
        ).statistic
    )
    summary["embedding_rc_tests_k6"] = embedding_tests(
        sequences[:fit_reads], 6, 64, seed, workers
    )
    (output_dir / "interpretation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("Saved interpretation: " + str(output_dir), flush=True)
    return summary
