import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.classifiers import (
    GRIDS,
    MODELS,
    fast_auc,
    genome_bootstrap_interval,
    resample_genomes,
    is_compatible,
    score_metrics,
    select_configuration,
)


def test_fast_auc_matches_scikit_learn_including_ties():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = np.round(rng.random(200) + 0.3 * labels, 1)
    assert fast_auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores))
    assert np.isnan(fast_auc(np.ones(5), np.arange(5)))


def test_metrics_for_perfect_and_constant_scores():
    labels = np.array([0, 0, 1, 1])
    perfect = score_metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect == {"auc": 1.0, "accuracy": 1.0, "f1": 1.0, "balanced_accuracy": 1.0}
    constant = score_metrics(labels, np.full(4, 0.5))
    assert constant["auc"] == 0.5
    assert constant["balanced_accuracy"] == 0.5


def test_bootstrap_resamples_whole_genomes():
    rng = np.random.default_rng(1)
    genomes = np.repeat(np.arange(20), 10)
    labels = (genomes % 2).astype(int)
    scores = labels + rng.normal(0, 0.7, 200)
    low, high = genome_bootstrap_interval(labels, scores, genomes, 200, 3)
    assert low < fast_auc(labels, scores) < high
    assert genome_bootstrap_interval(labels, scores, genomes, 0) != (low, high)
    assert all(np.isnan(genome_bootstrap_interval(labels, scores, np.zeros(200), 50)))


def test_compatibility_rules():
    assert not is_compatible("naive_bayes", "mean_vectors", 64)
    assert is_compatible("naive_bayes", "counts", 4096)
    assert not is_compatible("gradient_boosting", "counts", 4096)
    assert is_compatible("gradient_boosting", "mean_vectors", 64)
    assert all(model in GRIDS for model in MODELS)


@pytest.mark.parametrize("model", MODELS)
def test_each_model_learns_a_separable_signal_and_selects_on_validation(model):
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, 400)
    features = np.abs(rng.normal(0, 1, (400, 5))) + labels[:, None] * [2, 0, 0, 1, 0]
    best, estimator, rows = select_configuration(
        model, features[:300], labels[:300], features[300:], labels[300:], 0, 1, 20
    )
    assert max(row["val_auc"] for row in rows) > 0.85
    assert estimator.predict_proba(features[300:]).shape == (100, 2)
    assert set(best) == set(GRIDS[model])
    assert len(rows) == int(np.prod([len(values) for values in GRIDS[model].values()]))


def test_majority_baseline_is_chance():
    labels = np.array([0, 1] * 50)
    features = np.zeros((100, 1))
    _, _, rows = select_configuration("majority", features, labels, features, labels, 0)
    assert rows[0]["val_auc"] == 0.5
    assert rows[0]["val_accuracy"] == 0.5


def test_resampling_keeps_the_number_of_genomes_in_each_class():
    labels = np.array([0] * 30 + [1] * 6)
    genomes = np.array(
        [str(number // 3) for number in range(30)] + ["a", "a", "b", "b", "c", "c"]
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        rows = resample_genomes(labels, genomes, rng)
        # 10 negative genomes of 3 reads and 3 positive genomes of 2 reads each.
        assert (labels[rows] == 0).sum() == 30
        assert (labels[rows] == 1).sum() == 6
