import itertools
import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
from gensim.models import Word2Vec
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

from src.representations import pool_vectors, stable_hash


LETTER_CODES = np.full(256, 4, dtype=np.uint8)
for _code, _letter in enumerate("ACGT"):
    LETTER_CODES[ord(_letter)] = _code
KNOWN_RUN = re.compile("[ACGT]+")
REPRESENTATIONS = ("counts", "tfidf", "mean_vectors", "weighted_vectors")


def all_words(k: int) -> List[str]:
    """List every possible DNA word of one size in a fixed order.

    Args:
        k: Number of letters in each word.

    Returns:
        All 4**k words in alphabetical order, which matches count_words columns.
    """
    return ["".join(word) for word in itertools.product("ACGT", repeat=k)]


def encode_sequences(sequences: List[str]) -> np.ndarray:
    """Convert sequences to letter codes A=0, C=1, G=2, T=3, anything else 4.

    Args:
        sequences: Uppercase sequences; shorter ones are padded with code 4.

    Returns:
        An array with one row per sequence.
    """
    width = max((len(sequence) for sequence in sequences), default=0)
    encoded = np.full((len(sequences), width), 4, dtype=np.uint8)
    for row, sequence in enumerate(sequences):
        encoded[row, : len(sequence)] = LETTER_CODES[
            np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        ]
    return encoded


def count_words(sequences: List[str], k: int, chunk: int = 20000) -> sparse.csr_matrix:
    """Count overlapping words in each sequence, skipping windows with unknown letters.

    Args:
        sequences: Uppercase DNA sequences.
        k: Number of letters in each word.
        chunk: Number of sequences encoded at once, to bound memory.

    Returns:
        A sequence-by-word count table whose columns follow all_words(k).

    Raises:
        ValueError: If k is not positive.
    """
    if k < 1:
        raise ValueError("Word size must be at least 1.")
    blocks = []
    for start in range(0, len(sequences), chunk):
        codes = encode_sequences(sequences[start : start + chunk])
        rows, width = codes.shape
        windows = width - k + 1
        if windows < 1:
            blocks.append(sparse.csr_matrix((rows, 4**k), dtype=np.float32))
            continue
        index = np.zeros((rows, windows), dtype=np.int64)
        unknown = np.zeros((rows, windows), dtype=bool)
        for offset in range(k):
            letters = codes[:, offset : offset + windows]
            index = index * 4 + np.minimum(letters, 3)
            unknown |= letters == 4
        row_numbers = np.broadcast_to(np.arange(rows)[:, None], index.shape)
        keep = ~unknown
        table = sparse.coo_matrix(
            (
                np.ones(int(keep.sum()), dtype=np.float32),
                (row_numbers[keep], index[keep]),
            ),
            shape=(rows, 4**k),
        )
        blocks.append(table.tocsr())
    if not blocks:
        return sparse.csr_matrix((0, 4**k), dtype=np.float32)
    return sparse.vstack(blocks, format="csr")


class WordCorpus:
    """Re-iterable word lists for Word2Vec, one per uninterrupted known-letter run."""

    def __init__(self, sequences: List[str], k: int):
        self.sequences = sequences
        self.k = k

    def __iter__(self) -> Iterator[List[str]]:
        k = self.k
        for sequence in self.sequences:
            for run in KNOWN_RUN.findall(sequence):
                if len(run) >= k:
                    yield [run[start : start + k] for start in range(len(run) - k + 1)]


def fit_word_vectors(
    train_sequences: List[str],
    k: int,
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    seed: int = 42,
    workers: int = 1,
) -> tuple:
    """Learn word vectors from training sequences only, without labels.

    Args:
        train_sequences: Training sequences.
        k: Number of letters in each word.
        vector_size: Number of values in a word vector.
        window: Maximum number of neighbouring words on either side.
        epochs: Number of passes over the training sequences.
        seed: Fixed random setting.
        workers: CPU workers. Values above 1 lose exact repeatability.

    Returns:
        Vectors in all_words(k) order, with zeros for words never seen in
        training, and the learned model.

    Raises:
        ValueError: If settings are invalid or no training word can be formed.
    """
    if min(vector_size, window, epochs, workers) < 1:
        raise ValueError("Vector size, window, passes, and workers must be at least 1.")
    corpus = WordCorpus(train_sequences, k)
    if not any(True for _ in corpus):
        raise ValueError("No training words can be formed with this word size.")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        negative=5,
        epochs=epochs,
        workers=workers,
        seed=seed,
        hashfxn=stable_hash,
    )
    vectors = np.zeros((4**k, vector_size), dtype=np.float32)
    for number, word in enumerate(all_words(k)):
        if word in model.wv.key_to_index:
            vectors[number] = model.wv[word]
    return vectors, model


def build_representations(
    sequences_by_split: Dict[str, List[str]],
    k: int,
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    seed: int = 42,
    workers: int = 1,
    train: str = "train",
    fit_limit: int = 0,
) -> dict:
    """Learn representations from the training split and apply them to every split.

    Only the training split influences the inverse-document weights and word
    vectors; the other splits are transformed with what training learned.

    Args:
        sequences_by_split: Sequences for each split name.
        k: Number of letters in each word.
        vector_size: Number of values in a word vector.
        window: Maximum number of neighbouring words on either side.
        epochs: Number of passes over the training sequences.
        seed: Fixed random setting.
        workers: Word2Vec CPU workers.
        train: Name of the split used for learning.
        fit_limit: Learn only from this many leading training reads (0 = all);
            every training read is still transformed.

    Returns:
        Words, inverse-document weights, word vectors, and for each split the
        counts, tfidf, mean_vectors, and weighted_vectors tables.

    Raises:
        ValueError: If the training split is missing or settings are invalid.
    """
    if train not in sequences_by_split:
        raise ValueError("The training split is missing.")
    counts = {name: count_words(rows, k) for name, rows in sequences_by_split.items()}
    fitted = fit_limit or len(sequences_by_split[train])
    weighting = TfidfTransformer().fit(counts[train][:fitted])
    vectors, _ = fit_word_vectors(
        sequences_by_split[train][:fitted], k, vector_size, window, epochs, seed, workers
    )
    splits = {}
    for name, table in counts.items():
        weighted = weighting.transform(table)
        splits[name] = {
            "counts": table,
            "tfidf": weighted,
            "mean_vectors": pool_vectors(table, vectors),
            "weighted_vectors": pool_vectors(weighted, vectors),
        }
    return {
        "words": all_words(k),
        "idf": weighting.idf_,
        "word_vectors": vectors,
        "splits": splits,
    }


def save_representations(representations: dict, directory: Path, settings: dict):
    """Save learned representations so later runs can skip Word2Vec.

    Args:
        representations: Result of build_representations.
        directory: Destination directory.
        settings: Values that must match for the saved result to be reused.
    """
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "idf.npy", representations["idf"], allow_pickle=False)
    np.save(
        directory / "word_vectors.npy",
        representations["word_vectors"],
        allow_pickle=False,
    )
    for name, tables in representations["splits"].items():
        for kind in ("counts", "tfidf"):
            sparse.save_npz(directory / (name + "_" + kind + ".npz"), tables[kind])
        for kind in ("mean_vectors", "weighted_vectors"):
            np.save(
                directory / (name + "_" + kind + ".npy"),
                tables[kind],
                allow_pickle=False,
            )
    temporary = directory / "settings.json.part"
    temporary.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    temporary.replace(directory / "settings.json")


def load_representations(
    directory: Path, settings: dict, k: int, names: List[str]
) -> Optional[dict]:
    """Load saved representations when they were made with identical settings.

    Args:
        directory: Directory written by save_representations.
        settings: Settings the saved result must match.
        k: Number of letters in each word.
        names: Split names that must be present.

    Returns:
        The representations, or None if nothing matching is saved.
    """
    path = directory / "settings.json"
    if not path.exists() or json.loads(path.read_text()) != json.loads(
        json.dumps(settings)
    ):
        return None
    try:
        splits = {}
        for name in names:
            splits[name] = {
                "counts": sparse.load_npz(directory / (name + "_counts.npz")),
                "tfidf": sparse.load_npz(directory / (name + "_tfidf.npz")),
                "mean_vectors": np.load(directory / (name + "_mean_vectors.npy")),
                "weighted_vectors": np.load(
                    directory / (name + "_weighted_vectors.npy")
                ),
            }
        return {
            "words": all_words(k),
            "idf": np.load(directory / "idf.npy"),
            "word_vectors": np.load(directory / "word_vectors.npy"),
            "splits": splits,
        }
    except (OSError, ValueError):
        return None


def load_or_build_representations(
    sequences_by_split: Dict[str, List[str]],
    k: int,
    feature_dir: Path,
    split_digests: dict,
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    seed: int = 42,
    workers: int = 1,
    fit_limit: int = 0,
    train_limit: int = 0,
) -> dict:
    """Reuse saved representations for these settings, or learn and save them.

    Args:
        sequences_by_split: Sequences for each split name, including train.
        k: Number of letters in each word.
        feature_dir: Parent directory of the per-word-size caches.
        split_digests: Fingerprints of the split files, so a changed sample
            never reuses an old cache.
        vector_size: Number of values in a word vector.
        window: Maximum number of neighbouring words on either side.
        epochs: Word2Vec passes over the training reads.
        seed: Fixed random setting.
        workers: Word2Vec CPU workers.
        fit_limit: Learn only from this many leading training reads (0 = all).
        train_limit: Training reads kept by the caller, recorded so caches differ.

    Returns:
        The result of build_representations.
    """
    settings = {
        "k": k,
        "vector_size": vector_size,
        "window": window,
        "epochs": epochs,
        "seed": seed,
        "fit_limit": fit_limit,
        "train_limit": train_limit,
        "splits": split_digests,
    }
    directory = feature_dir / ("k_" + str(k))
    representations = load_representations(
        directory, settings, k, list(sequences_by_split)
    )
    if representations is None:
        representations = build_representations(
            sequences_by_split,
            k,
            vector_size,
            window,
            epochs,
            seed,
            workers,
            fit_limit=fit_limit,
        )
        save_representations(representations, directory, settings)
    else:
        print("  using saved representations: " + str(directory), flush=True)
    return representations
