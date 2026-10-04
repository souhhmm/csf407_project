import hashlib
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


def stable_hash(word: str) -> int:
    """Return the same word hash across Python processes."""
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")


def keep_words(words: List[str]) -> List[str]:
    """Pass precomputed words to the text counter unchanged."""
    return words


def pool_vectors(weights, vectors: np.ndarray) -> np.ndarray:
    """Average word vectors using each sequence's word weights.

    Args:
        weights: Sequence-by-word count or TF-IDF table.
        vectors: Word vectors in the same column order as weights.

    Returns:
        One vector per sequence; sequences with no words get a zero vector.
    """
    totals = np.asarray(weights.sum(axis=1)).reshape(-1, 1)
    combined = np.asarray(weights @ vectors, dtype=np.float32)
    return np.divide(combined, totals, out=np.zeros_like(combined), where=totals > 0)


def fit_representations(
    documents: List[List[str]],
    segments: List[List[str]],
    vector_size: int = 64,
    window: int = 5,
    epochs: int = 10,
    seed: int = 42,
) -> dict:
    """Learn four sequence representations without using class labels.

    Args:
        documents: Ordered words for each complete sequence.
        segments: Word lists from uninterrupted runs of known letters.
        vector_size: Number of values in a learned word vector.
        window: Maximum number of neighbouring words to consider on either side.
        epochs: Number of passes over the sequences when learning word vectors.
        seed: Fixed random setting.

    Returns:
        Counts, TF-IDF, word vectors, two whole-sequence vectors, and the model.

    Raises:
        ValueError: If the settings are invalid or the corpus has no words.
    """
    if min(vector_size, window, epochs) < 1:
        raise ValueError("Vector size, window, and passes must all be at least 1.")
    if not any(documents):
        raise ValueError("No words can be formed with this word size.")
    counter = CountVectorizer(analyzer=keep_words, lowercase=False, dtype=np.float32)
    counts = counter.fit_transform(documents)
    words = counter.get_feature_names_out().tolist()
    weighting = TfidfTransformer()
    tfidf = weighting.fit_transform(counts)
    model = Word2Vec(
        sentences=segments,
        vector_size=vector_size,
        window=window,
        min_count=1,
        sg=1,
        negative=5,
        epochs=epochs,
        workers=1,
        seed=seed,
        hashfxn=stable_hash,
    )
    vectors = np.asarray([model.wv[word] for word in words], dtype=np.float32)
    return {
        "counts": counts,
        "tfidf": tfidf,
        "words": words,
        "idf": weighting.idf_,
        "model": model,
        "word_vectors": vectors,
        "mean_vectors": pool_vectors(counts, vectors),
        "weighted_vectors": pool_vectors(tfidf, vectors),
    }


def project_to_two_dimensions(values, seed: int = 42) -> tuple:
    """Reduce a representation to two coordinates for an exploratory chart.

    Args:
        values: Sequence vectors or a sparse word-count table.
        seed: Fixed random setting for sparse reduction.

    Returns:
        Coordinates and the method, captured variation, and degenerate-case status.
    """
    rows, columns = values.shape
    coordinates = np.zeros((rows, 2), dtype=np.float32)
    is_sparse = sparse.issparse(values)
    method = "TruncatedSVD" if is_sparse else "PCA"
    if is_sparse:
        mean = np.asarray(values.mean(axis=0)).ravel()
        variation = float(
            np.asarray(values.multiply(values).mean(axis=0)).sum()
            - np.square(mean).sum()
        )
    else:
        variation = float(np.var(values, axis=0).sum())
    if rows < 2 or columns == 0 or variation <= 1e-10:
        return coordinates, {
            "method": method,
            "captured_variation": 0.0,
            "status": "no measurable variation",
        }
    if columns == 1:
        coordinates[:, 0] = values.toarray().ravel() if is_sparse else values[:, 0]
        return coordinates, {
            "method": "single feature",
            "captured_variation": 1.0,
            "status": "one coordinate",
        }
    components = (
        min(2, rows - 1, columns - 1) if is_sparse else min(2, rows - 1, columns)
    )
    reducer = (
        TruncatedSVD(n_components=components, random_state=seed)
        if is_sparse
        else PCA(n_components=components)
    )
    coordinates[:, :components] = reducer.fit_transform(values)
    captured = float(np.nan_to_num(reducer.explained_variance_ratio_).sum())
    return coordinates, {
        "method": method,
        "captured_variation": captured,
        "status": "ok",
    }


def nearest_words(
    model: Word2Vec, frequencies: Counter, queries: int = 20, neighbours: int = 5
) -> pd.DataFrame:
    """Find learned neighbours of frequent words.

    Args:
        model: The learned word representations.
        frequencies: Observed word counts.
        queries: Maximum number of frequent words to inspect.
        neighbours: Maximum number of neighbours per word.

    Returns:
        Query words, neighbours, vector similarities, and matching-letter fractions.
    """
    rows = []
    count = min(neighbours, len(model.wv) - 1)
    if count > 0:
        ranked = sorted(frequencies, key=lambda word: (-frequencies[word], word))[
            :queries
        ]
        for word in ranked:
            for neighbour, similarity in model.wv.most_similar(word, topn=count):
                rows.append(
                    {
                        "word": word,
                        "neighbour": neighbour,
                        "vector_similarity": float(similarity),
                        "matching_letter_fraction": sum(
                            a == b for a, b in zip(word, neighbour)
                        )
                        / len(word),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=["word", "neighbour", "vector_similarity", "matching_letter_fraction"],
    )


def _correlation(left: np.ndarray, right: np.ndarray):
    if len(left) < 3 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return None
    return float(spearmanr(left, right).statistic)


def compare_contexts(
    segments_by_sequence: List[List[List[str]]],
    words: List[str],
    vectors: np.ndarray,
    frequencies: Counter,
    k: int,
    window: int,
    seed: int,
    nonoverlapping: bool = False,
    max_sequences: int = 200,
    max_words: int = 256,
    max_pairs: int = 3000,
) -> tuple:
    """Compare shared surroundings with similarity of learned word vectors.

    Args:
        segments_by_sequence: Separate word lists for each sequence's known-letter runs.
        words: Vocabulary matching the rows of vectors.
        vectors: Learned word vectors.
        frequencies: Observed word counts.
        k: Number of letters in a word.
        window: Neighbour distance used during Word2Vec training.
        seed: Fixed random setting.
        nonoverlapping: Whether to examine only neighbours that share no sequence positions.
        max_sequences: Maximum randomly selected sequences to examine.
        max_words: Maximum frequent words whose surroundings are compared.
        max_pairs: Maximum randomly selected word pairs to compare.

    Returns:
        A word-pair table and a summary including a shuffled-vector comparison.
    """
    rng = np.random.default_rng(seed)
    word_indices = {word: number for number, word in enumerate(words)}
    frequent = sorted(words, key=lambda word: (-frequencies[word], word))[:max_words]
    target_indices = {word: number for number, word in enumerate(frequent)}
    selected = rng.choice(
        len(segments_by_sequence),
        size=min(max_sequences, len(segments_by_sequence)),
        replace=False,
    )
    radius = k + window if nonoverlapping else window
    offsets = [
        distance
        for distance in range(-radius, radius + 1)
        if distance and (not nonoverlapping or abs(distance) >= k)
    ]
    row_indices = []
    column_indices = []
    for sequence_index in selected:
        for segment in segments_by_sequence[sequence_index]:
            for position, word in enumerate(segment):
                if word not in target_indices:
                    continue
                for offset in offsets:
                    neighbour = position + offset
                    if 0 <= neighbour < len(segment):
                        row_indices.append(target_indices[word])
                        column_indices.append(word_indices[segment[neighbour]])
    cooccurrences = sparse.coo_matrix(
        (np.ones(len(row_indices), dtype=np.float32), (row_indices, column_indices)),
        shape=(len(frequent), len(words)),
    ).tocsr()
    supported = np.flatnonzero(np.asarray(cooccurrences.sum(axis=1)).ravel() >= 5)
    columns = [
        "word_left",
        "word_right",
        "context_similarity",
        "vector_similarity",
        "matching_letter_fraction",
    ]
    summary = {
        "sequences_examined": len(selected),
        "words_compared": len(supported),
        "nonoverlapping": nonoverlapping,
    }
    if len(supported) < 3:
        return pd.DataFrame(columns=columns), {
            **summary,
            "pairs": 0,
            "correlation": None,
            "shuffled_correlation": None,
        }
    frequent = [frequent[number] for number in supported]
    contexts = normalize(cooccurrences[supported])
    context_similarity = (contexts @ contexts.T).toarray()
    word_vectors = normalize(vectors[[word_indices[word] for word in frequent]])
    vector_similarity = word_vectors @ word_vectors.T
    left, right = np.triu_indices(len(frequent), k=1)
    if len(left) > max_pairs:
        pairs = rng.choice(len(left), size=max_pairs, replace=False)
        left, right = left[pairs], right[pairs]
    context_values = context_similarity[left, right]
    vector_values = vector_similarity[left, right]
    shuffled = rng.permutation(len(frequent))
    shuffled_values = vector_similarity[shuffled[left], shuffled[right]]
    table = pd.DataFrame(
        {
            "word_left": [frequent[number] for number in left],
            "word_right": [frequent[number] for number in right],
            "context_similarity": context_values,
            "vector_similarity": vector_values,
            "matching_letter_fraction": [
                sum(a == b for a, b in zip(frequent[first], frequent[second])) / k
                for first, second in zip(left, right)
            ],
        }
    )
    return table, {
        **summary,
        "pairs": len(table),
        "correlation": _correlation(context_values, vector_values),
        "shuffled_correlation": _correlation(context_values, shuffled_values),
    }
