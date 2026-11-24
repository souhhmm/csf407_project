import numpy as np
import pandas as pd

from src.genome_report import (
    aggregation_curve,
    genome_level_metrics,
    genome_table,
    genome_weighted_auc,
)
from src.interpret import (
    benjamini_hochberg,
    genome_word_effects,
    rc_similarity,
    reverse_complement_word,
)


def _reads():
    rng = np.random.default_rng(0)
    genomes = np.repeat([f"g{i}" for i in range(10)], 40)
    labels = np.repeat([0] * 4 + [1] * 6, 40)
    scores = np.clip(0.3 + 0.4 * labels + rng.normal(0, 0.15, labels.size), 0, 1)
    return labels, scores, genomes


def test_genome_weighted_auc_equals_read_auc_when_genomes_are_equal_size():
    labels, scores, genomes = _reads()
    from src.classifiers import fast_auc

    assert genome_weighted_auc(labels, scores, genomes) == pytest_approx(
        fast_auc(labels, scores)
    )


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-9)


def test_genome_table_and_metrics_count_each_genome_once():
    labels, scores, genomes = _reads()
    table = genome_table(labels, scores, genomes)
    assert len(table) == 10 and table["reads"].eq(40).all()
    metrics = genome_level_metrics(table)
    assert metrics["genomes_nonpathogenic"] == 4 and metrics["genomes_pathogenic"] == 6
    assert metrics["genome_auc"] == 1.0


def test_aggregating_more_reads_does_not_hurt():
    labels, scores, genomes = _reads()
    curve = pd.DataFrame(aggregation_curve(labels, scores, genomes))
    assert curve["genome_auc_mean"].iloc[-1] >= curve["genome_auc_mean"].iloc[0]


def test_word_effects_find_the_planted_word_and_adjust_p_values():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 8 + [1] * 8)
    table = rng.dirichlet(np.ones(16) * 20, size=16)
    table[labels == 1, 3] *= 3
    table /= table.sum(axis=1, keepdims=True)
    effects = genome_word_effects(table, labels, [str(i) for i in range(16)])
    assert effects["genome_auc"].idxmax() == 3
    assert (effects["q_value"] >= effects["p_value"]).all()
    assert benjamini_hochberg(np.array([0.01, 0.02, 0.5])).max() <= 1.0


def test_reverse_complement_similarity_is_perfect_for_symmetric_vectors():
    assert reverse_complement_word("AAC") == "GTT"
    from src.features import all_words

    words = all_words(2)
    index = {word: number for number, word in enumerate(words)}
    base = np.random.default_rng(0).normal(size=(16, 8))
    vectors = np.asarray(
        [
            base[min(number, index[reverse_complement_word(word)])]
            for number, word in enumerate(words)
        ]
    )
    result = rc_similarity(vectors, 2)
    assert result["rc_cosine_mean"] > 0.999
