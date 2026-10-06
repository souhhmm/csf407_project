import json
import subprocess
import sys

import pandas as pd
import pytest

from src.simple_models import run_simple_models
from src.splits import prepare_splits


SIZES = {"train": 150, "val": 60, "test_1": 60, "test_2": 60}


@pytest.fixture
def run(synthetic_sources, tmp_path):
    data_dir, sources = synthetic_sources
    splits_dir = tmp_path / "splits"
    prepare_splits(data_dir, splits_dir, SIZES, 7, sources=sources)
    output = tmp_path / "results"
    summary = run_simple_models(
        splits_dir,
        output,
        tmp_path / "features",
        ks=(3, 4),
        models=("logistic_regression", "naive_bayes", "decision_tree"),
        vector_size=8,
        epochs=1,
        seeds=(1, 2),
        trees=10,
        bootstrap=20,
    )
    return output, summary, splits_dir, tmp_path


def test_run_saves_tables_figures_and_report(run):
    output, summary, _, _ = run
    for name in (
        "tuning.csv",
        "results.csv",
        "results_summary.csv",
        "summary.json",
        "report.md",
    ):
        assert (output / name).exists()
    charts = list((output / "figures").glob("*.png"))
    assert {path.name for path in charts} == {
        "model_heatmap.png",
        "baseline_comparison.png",
        "generalisation.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in charts)
    report = (output / "report.md").read_text()
    assert "shuffled" in report and "Models ranked by validation AUC" in report
    assert json.loads((output / "summary.json").read_text())["seeds"] == [1, 2]


def test_baselines_controls_and_models_behave_as_expected(run):
    output, _, _, _ = run
    summary = pd.read_csv(output / "results_summary.csv")
    majority = summary[summary["model"] == "majority"].iloc[0]
    assert majority["val_auc_mean"] == 0.5 and majority["test_1_auc_mean"] == 0.5
    gc = summary[summary["representation"] == "gc_fraction"].iloc[0]
    assert gc["test_1_auc_mean"] > 0.85
    control = summary[summary["group"] == "control"]
    assert len(control) == 8
    assert (control["seeds"] == 5).all()
    assert control["test_1_auc_std"].gt(0).all()
    assert control["test_1_auc_mean"].between(0.1, 0.9).all()
    models = summary[summary["group"] == "model"]
    assert models["val_auc_mean"].max() > 0.85
    assert set(models["model"]) == {
        "logistic_regression",
        "naive_bayes",
        "decision_tree",
    }
    assert not (
        (models["model"] == "naive_bayes")
        & models["representation"].isin(["mean_vectors", "weighted_vectors"])
    ).any()


def test_deterministic_models_run_once_and_stochastic_models_per_seed(run):
    output, _, _, _ = run
    results = pd.read_csv(output / "results.csv")
    models = results[results["group"] == "model"]
    seeds = models.groupby("model")["seed"].agg(lambda values: sorted(set(values)))
    assert seeds["logistic_regression"] == [1]
    assert seeds["decision_tree"] == [1, 2]
    assert results.loc[results["group"] == "control", "seed"].nunique() == 5
    intervals = results["test_1_auc_low"]
    assert intervals.notna().all() and (intervals <= results["test_1_auc"]).all()
    tuning = pd.read_csv(output / "tuning.csv")
    assert not any(column.startswith("test") for column in tuning)


def test_saved_representations_make_a_rerun_identical(run, tmp_path):
    output, _, splits_dir, base = run
    again = base / "again"
    run_simple_models(
        splits_dir,
        again,
        base / "features",
        ks=(3, 4),
        models=("logistic_regression", "naive_bayes", "decision_tree"),
        vector_size=8,
        epochs=1,
        seeds=(1, 2),
        trees=10,
        bootstrap=20,
    )
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "results_summary.csv").drop(columns="params"),
        pd.read_csv(again / "results_summary.csv").drop(columns="params"),
    )


def test_invalid_settings_are_refused(synthetic_sources, tmp_path):
    with pytest.raises(ValueError, match="Unknown models"):
        run_simple_models(tmp_path, tmp_path, tmp_path, models=("nonsense",))
    with pytest.raises(ValueError, match="seed"):
        run_simple_models(tmp_path, tmp_path, tmp_path, seeds=())
    with pytest.raises(ValueError, match="prepare-splits"):
        run_simple_models(tmp_path / "none", tmp_path, tmp_path)


def test_cli_reports_missing_splits_without_a_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "run.py",
            "simple-models",
            "--splits-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "prepare-splits" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_does_not_load_numpy_before_thread_limits_are_set():
    result = subprocess.run(
        [sys.executable, "-c", "import run, sys; print('numpy' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
