import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import extend_saved_analysis
from src.exploration import compare_class_frequencies, compare_shuffled_letters


def test_class_comparison_uses_rates_and_handles_zero_counts():
    frequencies = pd.DataFrame(
        [
            ("a", "AA", 40),
            ("a", "CC", 60),
            ("a", "GG", 0),
            ("b", "AA", 10),
            ("b", "CC", 10),
            ("b", "GG", 0),
        ],
        columns=["label", "word", "count"],
    )
    table = compare_class_frequencies(frequencies).set_index("word")
    assert table.loc["AA", "fraction_numerator"] == 0.5
    assert table.loc["AA", "fraction_denominator"] == 0.4
    assert table.loc["AA", "log2_ratio"] == pytest.approx(
        np.log2((10.5 / 21.5) / (40.5 / 101.5))
    )
    assert np.isfinite(table["log2_ratio"]).all()
    assert compare_class_frequencies(frequencies.loc[frequencies["label"] == "a"]).empty


@pytest.mark.parametrize(
    "sequence,alphabet,expected",
    [
        ("AAAANNCCCC", "dna", {"AAA": 2, "CCC": 2}),
        ("UUUUNNGGGG", "rna", {"UUU": 2, "GGG": 2}),
    ],
)
def test_shuffling_keeps_unknown_gaps_and_counts(sequence, alphabet, expected):
    table = compare_shuffled_letters(
        [{"sequence": sequence}], 3, alphabet, repeats=3
    ).set_index("word")
    assert table["count"].to_dict() == expected
    assert table["shuffled_mean"].to_dict() == expected
    assert (table["shuffled_sd"] == 0).all()
    assert (table["log2_ratio"] == 0).all()


def test_shuffling_is_repeatable_and_conserves_usable_windows():
    records = [{"sequence": "ACGTACGTRACGTTTNNACGT"}]
    first = compare_shuffled_letters(records, 3, repeats=5, seed=8)
    second = compare_shuffled_letters(records, 3, repeats=5, seed=8)
    pd.testing.assert_frame_equal(first, second)
    assert first["count"].sum() == 12
    assert first["shuffled_mean"].sum() == pytest.approx(12)
    assert np.isfinite(first["log2_ratio"]).all()


def test_saved_analysis_rejects_a_different_input(tmp_path):
    source = tmp_path / "sequences.csv"
    source.write_text("id,sequence,label\none,ACGT,a\n")
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps({"input_sha256": "different"}))
    with pytest.raises(ValueError, match="does not match the saved analysis"):
        extend_saved_analysis(source, output)
