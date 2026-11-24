import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.classifiers import fast_auc
from src.features import all_words, count_words, encode_sequences
from src.networks import (
    GRIDS,
    NEURAL_MODELS,
    ReadCNN,
    kmer_tokens,
    make_network,
    reverse_complement,
)
from src.neural_models import run_neural_models
from src.neural_training import ReadInputs, predict_scores, train_network
from src.splits import prepare_splits


SIZES = {"train": 150, "val": 60, "test_1": 60, "test_2": 60}
CPU = torch.device("cpu")


def test_reverse_complement_swaps_strands_and_keeps_unknowns():
    codes = torch.from_numpy(encode_sequences(["AACGTN"]))
    flipped = reverse_complement(codes)
    assert flipped.tolist() == [[4, 0, 1, 2, 3, 3]]
    assert torch.equal(reverse_complement(flipped), codes)


def test_kmer_tokens_match_count_table_and_mark_unknown_windows():
    sequences = ["ACGTACGTTT", "ACNGTACGTT"]
    tokens = kmer_tokens(torch.from_numpy(encode_sequences(sequences)), 2)
    counts = count_words(sequences, 2).toarray()
    for row in range(2):
        found = np.bincount(tokens[row].numpy(), minlength=17)
        assert found[:16].tolist() == counts[row].astype(int).tolist()
    assert (tokens[1] == 16).sum().item() == 2
    assert all_words(2)[int(tokens[0][0])] == "AC"


def test_every_network_gives_one_score_per_read():
    codes = torch.from_numpy(encode_sequences(["ACGT" * 15] * 3))
    vectors = np.random.default_rng(0).normal(size=(64, 8)).astype(np.float32)
    for model in ("cnn", "gru", "transformer"):
        params = {name: values[0] for name, values in GRIDS[model].items()}
        network = make_network(model, params, 3, 0, 60, vectors)
        assert network(codes).shape == (3,)
    mlp = make_network("mlp", {"hidden": 16}, 3, 64, 60, None)
    assert mlp(torch.zeros(3, 64)).shape == (3,)
    assert set(NEURAL_MODELS) == {
        "mlp",
        "cnn",
        "gru",
        "transformer",
        "dnabert2",
        "dnabert2_scratch",
    }


def test_prediction_is_the_same_for_a_read_and_its_reverse_complement():
    torch.manual_seed(0)
    reads = ["ACGGTCATTGACCA" * 5, "TTTGACCGGATAAC" * 5]
    codes = encode_sequences(reads)
    network = ReadCNN(8, 5, 0.0)
    forward = predict_scores(network, ReadInputs(codes, CPU), CPU)
    backward = predict_scores(
        network,
        ReadInputs(reverse_complement(torch.from_numpy(codes)).numpy(), CPU),
        CPU,
    )
    assert np.allclose(forward, backward, atol=1e-6)


def test_training_learns_a_separable_signal_and_restores_best_weights():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 400)
    reads = ["".join(rng.choice(list("GC" if label else "AT"), 40)) for label in labels]
    inputs = ReadInputs(encode_sequences(reads), CPU)
    torch.manual_seed(0)
    network, history = train_network(
        ReadCNN(8, 5, 0.0), inputs, labels, inputs, labels, 1e-2, 0, CPU, 6, 3, 64
    )
    best = max(row["val_auc"] for row in history)
    assert best > 0.9
    restored = predict_scores(network, inputs, CPU)
    assert fast_auc(labels, restored) == pytest.approx(best, abs=1e-6)


def test_run_saves_tables_figures_and_report(synthetic_sources, tmp_path, monkeypatch):
    # Shrink the networks so the run is quick on a CPU.
    from src import networks

    for name, grid in {
        "mlp": {"hidden": [16, 32]},
        "cnn": {"filters": [8, 16]},
        "gru": {"hidden": [8, 16]},
        "transformer": {"layers": [1, 2]},
    }.items():
        monkeypatch.setitem(networks.GRIDS, name, grid)
    monkeypatch.setitem(
        networks.FIXED,
        "transformer",
        {"lr": 5e-4, "dropout": 0.1, "d_model": 32, "heads": 4},
    )
    data_dir, sources = synthetic_sources
    splits = tmp_path / "splits"
    prepare_splits(data_dir, splits, SIZES, 7, sources=sources)
    output = tmp_path / "neural"
    run_neural_models(
        splits,
        output,
        tmp_path / "features",
        ks=(3,),
        vector_size=8,
        epochs=1,
        seeds=(1, 2),
        bootstrap=10,
        max_epochs=2,
        batch_size=64,
        device="cpu",
    )
    for name in ("tuning.csv", "results.csv", "results_summary.csv", "history.csv"):
        assert (output / name).exists()
    assert {path.name for path in (output / "figures").glob("*.png")} == {
        "neural_comparison.png",
        "training_curves.png",
    }
    summary = pd.read_csv(output / "results_summary.csv")
    assert set(summary[summary["group"] == "model"]["model"]) == {
        "mlp",
        "cnn",
        "gru",
        "transformer",
    }
    assert any((output / "scores").glob("*.npz"))
    assert (summary["group"] == "control").any()
    tuning = pd.read_csv(output / "tuning.csv")
    assert "test_1_auc" not in tuning.columns
    assert "Models ranked by validation AUC" in (output / "report.md").read_text()
