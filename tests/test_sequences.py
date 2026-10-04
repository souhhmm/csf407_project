import math

import pytest

from dna_language.sequences import (
    clean_sequence,
    make_words,
    sequence_statistics,
    word_segments,
)


def test_overlapping_words():
    assert make_words("ATGCA", 3) == ["ATG", "TGC", "GCA"]
    assert make_words("AT", 3) == []


def test_unknown_letters_do_not_create_words_or_false_neighbours():
    assert make_words("ATNGCA", 2) == ["AT", "GC", "CA"]
    assert word_segments("ATNGCA", 2) == [["AT"], ["GC", "CA"]]
    assert word_segments("NNNN", 3) == []


def test_case_spacing_and_rna_are_explicit():
    assert clean_sequence(" aT g\nc ") == "ATGC"
    assert make_words("AUGCA", 3, "rna") == ["AUG", "UGC", "GCA"]
    with pytest.raises(ValueError, match="Unexpected"):
        clean_sequence("AUGC", "dna")
    with pytest.raises(ValueError, match="Unexpected"):
        clean_sequence("ATGC", "rna")


@pytest.mark.parametrize("sequence", ["", "  ", "AC-GT", "ACGTZ"])
def test_invalid_sequences_are_reported(sequence):
    with pytest.raises(ValueError):
        clean_sequence(sequence)


def test_nonpositive_word_sizes_are_rejected():
    for function in (make_words, word_segments):
        with pytest.raises(ValueError, match="at least 1"):
            function("NNN", 0)


def test_letter_proportions_exclude_unknown_letters():
    stats = sequence_statistics("ACGTNN")
    assert stats["length"] == 6
    assert stats["ambiguous_letters"] == 2
    assert stats["gc_fraction"] == 0.5
    assert stats["A_fraction"] == 0.25


def test_all_unknown_letters_have_unavailable_proportions():
    stats = sequence_statistics("NNNN")
    assert stats["ambiguous_letters"] == 4
    assert math.isnan(stats["gc_fraction"])
    assert math.isnan(stats["A_fraction"])
