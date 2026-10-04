import csv
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import sparse

from src.analysis import analyze_dataset


def test_full_pipeline_saves_aligned_representations_and_minimal_charts(tmp_path):
    source = tmp_path / "sequences.csv"
    records = [
        ("one", "ATGCATGCATGC", "a"),
        ("two", "ATGCATGCATGA", "a"),
        ("three", "CCCCGGGGCCCC", "b"),
        ("four", "GGGGCCCCGGGA", "b"),
        ("five", "NNNNNNNNNNNN", "a"),
    ]
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "sequence", "label"])
        writer.writerows(records)
    output = tmp_path / "results"
    summary = analyze_dataset(source, output, ks=(3, 30), vector_size=8, epochs=2)
    directory = output / "k_3"
    rows = pd.read_csv(directory / "row_ids.csv")
    assert rows["id"].tolist() == [record[0] for record in records]
    counts = sparse.load_npz(directory / "counts.npz")
    tfidf = sparse.load_npz(directory / "tfidf.npz")
    assert counts.shape == tfidf.shape
    assert counts.shape[0] == 5
    assert counts.getrow(4).nnz == 0
    assert np.load(directory / "mean_vectors.npy").shape == (5, 8)
    assert summary["word_sizes"][0]["empty_sequences"] == 1
    assert summary["word_sizes"][1]["status"] == "skipped"
    assert json.loads((output / "summary.json").read_text())["sequence_count"] == 5
    charts = list(output.rglob("*.png"))
    assert len(charts) == 4
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in charts)
    assert not (directory / "word_space.png").exists()
    assert (output / "report.md").exists()


def test_cli_reports_invalid_input_without_a_traceback(tmp_path):
    result = subprocess.run(
        [sys.executable, "run.py", "analyze", "--input", str(tmp_path / "missing.csv")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Could not complete" in result.stderr
    assert "Traceback" not in result.stderr
