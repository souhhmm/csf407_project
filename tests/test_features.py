import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from src.features import (
    WordCorpus,
    all_words,
    build_representations,
    count_words,
    load_representations,
    save_representations,
)
from src.representations import keep_words
from src.sequences import make_words, word_segments


SEQUENCES = ["ACGTNACGTTTGA", "NNNNAC", "GGGGGGGGGG", "acgtRTTGCA".upper(), "ACG"]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_fast_counts_match_the_step_one_word_maker(k):
    expected = CountVectorizer(
        analyzer=keep_words, vocabulary=all_words(k), dtype=np.float32
    ).transform([make_words(sequence, k) for sequence in SEQUENCES])
    assert (count_words(SEQUENCES, k) != expected).nnz == 0
    assert count_words(SEQUENCES, k).shape == (len(SEQUENCES), 4**k)


def test_corpus_matches_step_one_context_groups():
    for k in (2, 3):
        assert [sentence for sentence in WordCorpus(SEQUENCES, k)] == [
            segment for sequence in SEQUENCES for segment in word_segments(sequence, k)
        ]


def test_count_words_handles_chunks_and_short_input():
    assert count_words([], 3).shape == (0, 64)
    assert count_words(["AC"], 3).nnz == 0
    sequences = SEQUENCES * 5
    assert (count_words(sequences, 3, chunk=4) != count_words(sequences, 3)).nnz == 0
    with pytest.raises(ValueError):
        count_words(SEQUENCES, 0)


def _splits(rng, other_letters):
    train = ["".join(rng.choice(list("ACGT"), 60)) for _ in range(30)]
    other = ["".join(rng.choice(list(other_letters), 60)) for _ in range(10)]
    return {"train": train, "val": other, "test": other[:5]}


def test_only_training_data_shapes_learned_representations():
    first = _splits(np.random.default_rng(0), "ACGT")
    second = {**first, "val": ["AAAAAAAAAAAAAAAA"] * 12, "test": ["CCCCCCCCCCCC"] * 3}
    one = build_representations(first, 3, vector_size=8, epochs=2)
    two = build_representations(second, 3, vector_size=8, epochs=2)
    np.testing.assert_array_equal(one["idf"], two["idf"])
    np.testing.assert_array_equal(one["word_vectors"], two["word_vectors"])
    np.testing.assert_array_equal(
        one["splits"]["train"]["mean_vectors"], two["splits"]["train"]["mean_vectors"]
    )
    assert one["splits"]["val"]["counts"].shape == (10, 64)
    assert one["splits"]["val"]["mean_vectors"].shape == (10, 8)


def test_words_never_seen_in_training_get_zero_vectors():
    splits = {"train": ["ACACACACACACAC"] * 5, "val": ["GGGGGGGG"]}
    result = build_representations(splits, 2, vector_size=4, epochs=1)
    words = all_words(2)
    assert np.all(result["word_vectors"][words.index("GG")] == 0)
    assert np.all(result["splits"]["val"]["mean_vectors"] == 0)
    assert np.any(result["word_vectors"][words.index("AC")] != 0)
    with pytest.raises(ValueError):
        build_representations({"train": ["NNNN"]}, 2, vector_size=4, epochs=1)
    with pytest.raises(ValueError):
        build_representations({"val": ["ACGT"]}, 2)


def test_saved_representations_are_reused_only_for_identical_settings(tmp_path):
    splits = _splits(np.random.default_rng(1), "ACGT")
    result = build_representations(splits, 3, vector_size=8, epochs=1)
    settings = {"k": 3, "vector_size": 8}
    save_representations(result, tmp_path, settings)
    loaded = load_representations(tmp_path, settings, 3, list(splits))
    np.testing.assert_array_equal(loaded["word_vectors"], result["word_vectors"])
    assert (
        loaded["splits"]["val"]["tfidf"] != result["splits"]["val"]["tfidf"]
    ).nnz == 0
    assert load_representations(tmp_path, {**settings, "k": 4}, 3, list(splits)) is None
    assert load_representations(tmp_path / "missing", settings, 3, list(splits)) is None
