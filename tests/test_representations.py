from collections import Counter

import numpy as np
import pytest
from gensim.models import Word2Vec
from scipy import sparse

from src.representations import (
    compare_contexts,
    fit_representations,
    pool_vectors,
    project_to_two_dimensions,
)


def test_counts_and_tfidf_match_hand_calculations():
    documents = [["AAA", "AAA", "AAC"], ["AAC", "CCC"], []]
    result = fit_representations(documents, documents[:2], vector_size=8, epochs=2)
    assert result["words"] == ["AAA", "AAC", "CCC"]
    expected_counts = np.array([[2, 1, 0], [0, 1, 1], [0, 0, 0]])
    np.testing.assert_array_equal(result["counts"].toarray(), expected_counts)
    expected_idf = np.log(4 / np.array([2, 3, 2])) + 1
    np.testing.assert_allclose(result["idf"], expected_idf, rtol=1e-6)
    expected = expected_counts * expected_idf
    lengths = np.linalg.norm(expected, axis=1, keepdims=True)
    expected = np.divide(
        expected, lengths, out=np.zeros_like(expected), where=lengths > 0
    )
    np.testing.assert_allclose(result["tfidf"].toarray(), expected, rtol=1e-6)
    np.testing.assert_array_equal(result["mean_vectors"][2], np.zeros(8))


def test_vector_averages_include_repeated_words_and_handle_empty_rows():
    counts = sparse.csr_matrix([[2, 1], [0, 0], [0, 2]], dtype=np.float32)
    vectors = np.array([[1, 0], [0, 2]], dtype=np.float32)
    expected = [[2 / 3, 2 / 3], [0, 0], [0, 2]]
    np.testing.assert_allclose(pool_vectors(counts, vectors), expected, rtol=1e-6)


def test_word_vectors_repeat_and_survive_saving(tmp_path):
    documents = [["AAA", "AAC", "CCC", "AAC"] * 20] * 20
    first = fit_representations(documents, documents, vector_size=8, epochs=4, seed=17)
    second = fit_representations(documents, documents, vector_size=8, epochs=4, seed=17)
    np.testing.assert_array_equal(first["word_vectors"], second["word_vectors"])
    path = tmp_path / "word2vec.model"
    first["model"].save(str(path))
    loaded = Word2Vec.load(str(path))
    np.testing.assert_array_equal(loaded.wv["AAA"], first["model"].wv["AAA"])
    assert np.isfinite(first["word_vectors"]).all()
    assert np.any(first["model"].syn1neg != 0)


@pytest.mark.parametrize(
    "values",
    [np.ones((4, 3)), np.array([[1, 2, 3]]), sparse.csr_matrix(np.ones((4, 3)))],
)
def test_no_variation_does_not_create_nan_coordinates(values):
    points, information = project_to_two_dimensions(values)
    assert points.shape == (values.shape[0], 2)
    assert np.isfinite(points).all()
    assert information["captured_variation"] == 0


def test_single_feature_is_kept_without_inventing_a_second_coordinate():
    points, information = project_to_two_dimensions(sparse.csr_matrix([[1], [2], [4]]))
    np.testing.assert_array_equal(points, [[1, 0], [2, 0], [4, 0]])
    assert information["method"] == "single feature"


def test_nonoverlapping_context_control_excludes_shared_sequence_positions():
    segments = [[["AAA", "AAC"]] for _ in range(20)]
    words = ["AAA", "AAC"]
    vectors = np.eye(2, dtype=np.float32)
    frequencies = Counter({"AAA": 20, "AAC": 20})
    _, standard = compare_contexts(segments, words, vectors, frequencies, 3, 1, 42)
    _, nonoverlapping = compare_contexts(
        segments, words, vectors, frequencies, 3, 1, 42, nonoverlapping=True
    )
    assert standard["words_compared"] == 2
    assert nonoverlapping["words_compared"] == 0
    assert nonoverlapping["correlation"] is None
