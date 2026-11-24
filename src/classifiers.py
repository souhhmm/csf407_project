import itertools
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "naive_bayes",
    "gradient_boosting",
)
GRIDS = {
    "majority": {},
    "logistic_regression": {"C": [0.001, 0.01, 0.1, 1.0]},
    "decision_tree": {"max_depth": [4, 8, 16]},
    "random_forest": {"min_samples_leaf": [5, 50]},
    "naive_bayes": {"alpha": [0.1, 1.0]},
    "gradient_boosting": {"learning_rate": [0.05, 0.1]},
}
# Models whose result depends on the random seed; the rest are deterministic.
STOCHASTIC = {"decision_tree", "random_forest", "gradient_boosting"}
NONNEGATIVE_REPRESENTATIONS = {"counts", "tfidf"}
MAX_BOOSTING_FEATURES = 1024
METRICS = ("auc", "accuracy", "f1", "balanced_accuracy")


def is_compatible(model: str, representation: str, features: int) -> bool:
    """Decide whether a model suits a representation.

    Args:
        model: Model name from MODELS.
        representation: Representation name.
        features: Number of columns in the representation.

    Returns:
        False for naive Bayes on negative-valued vectors, and for gradient
        boosting on very wide tables, which must be made dense first.
    """
    if model == "naive_bayes":
        return representation in NONNEGATIVE_REPRESENTATIONS
    if model == "gradient_boosting":
        return features <= MAX_BOOSTING_FEATURES
    return True


def make_estimator(
    model: str, params: dict, seed: int, n_jobs: int, trees: int, is_sparse: bool
):
    """Create an untrained classifier.

    Args:
        model: Model name from MODELS, or majority.
        params: Hyperparameters from GRIDS.
        seed: Fixed random setting.
        n_jobs: CPU workers for models that can use several.
        trees: Number of trees in a random forest.
        is_sparse: Whether the input table is sparse, which decides scaling.

    Returns:
        An estimator with fit and predict_proba.

    Raises:
        ValueError: If the model name is unknown.
    """
    if model == "majority":
        return DummyClassifier(strategy="prior")
    if model == "logistic_regression":
        return make_pipeline(
            StandardScaler(with_mean=not is_sparse),
            LogisticRegression(C=params["C"], max_iter=300, tol=1e-3),
        )
    if model == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=params["max_depth"], min_samples_leaf=20, random_state=seed
        )
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=params["min_samples_leaf"],
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if model == "naive_bayes":
        return MultinomialNB(alpha=params["alpha"])
    if model == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError("Unknown model: " + model)


def fit_estimator(estimator, model: str, features, labels: np.ndarray):
    """Train an estimator, densifying the table only where the model needs it.

    Args:
        estimator: Result of make_estimator.
        model: Model name.
        features: Training table, dense or sparse.
        labels: Training labels, 1 for the positive class.

    Returns:
        The fitted estimator.
    """
    if model == "gradient_boosting" and sparse.issparse(features):
        features = features.toarray()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimator.fit(features, labels)
    return estimator


def predict_scores(estimator, model: str, features) -> np.ndarray:
    """Return each read's estimated probability of the positive class.

    Args:
        estimator: A fitted estimator.
        model: Model name.
        features: Table to score, dense or sparse.

    Returns:
        One score per row.
    """
    if model == "gradient_boosting" and sparse.issparse(features):
        features = features.toarray()
    return estimator.predict_proba(features)[:, 1]


def fast_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate the area under the ROC curve from ranks.

    Args:
        labels: 1 for the positive class and 0 otherwise.
        scores: Higher values mean more positive.

    Returns:
        The area, or NaN when only one class is present.
    """
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Calculate ranking and thresholded metrics, treating 1 as the positive class.

    Args:
        labels: True labels, 1 for the positive class.
        scores: Estimated probabilities of the positive class.

    Returns:
        AUC, plus accuracy, F1, and balanced accuracy at a 0.5 threshold.
    """
    predicted = (scores >= 0.5).astype(int)
    return {
        "auc": fast_auc(labels, scores),
        "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
    }


def resample_genomes(
    labels: np.ndarray, genomes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw read positions by resampling whole genomes within each class.

    Resampling within a class keeps the number of genomes per class fixed, so
    interval width is not inflated by random swings in class composition.

    Args:
        labels: True labels, 1 for the positive class.
        genomes: Source genome of each read.
        rng: Random number generator.

    Returns:
        Read positions of the resampled genomes.
    """
    rows = []
    for label in np.unique(labels):
        members = {}
        for position in np.flatnonzero(labels == label):
            members.setdefault(genomes[position], []).append(position)
        pool = list(members.values())
        for number in rng.integers(0, len(pool), size=len(pool)):
            rows.append(np.asarray(pool[number]))
    return np.concatenate(rows)


def genome_bootstrap_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    genomes: np.ndarray,
    resamples: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    """Estimate a 95% interval for AUC by resampling whole genomes within each class.

    Reads from one genome are not independent, so resampling reads would give
    intervals that are too narrow.

    Args:
        labels: True labels, 1 for the positive class.
        scores: Estimated probabilities of the positive class.
        genomes: Source genome of each read.
        resamples: Number of bootstrap resamples; 0 disables the interval.
        seed: Fixed random setting.

    Returns:
        Lower and upper bounds, or NaNs if disabled or too few genomes.
    """
    if resamples < 1 or len(np.unique(genomes)) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        rows = resample_genomes(labels, genomes, rng)
        value = fast_auc(labels[rows], scores[rows])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def select_configuration(
    model: str,
    train_features,
    train_labels: np.ndarray,
    val_features,
    val_labels: np.ndarray,
    seed: int,
    n_jobs: int = 1,
    trees: int = 200,
) -> tuple:
    """Choose hyperparameters by validation AUC, without seeing any test data.

    Args:
        model: Model name from MODELS, or majority.
        train_features: Training table.
        train_labels: Training labels, 1 for the positive class.
        val_features: Validation table.
        val_labels: Validation labels.
        seed: Fixed random setting.
        n_jobs: CPU workers for models that can use several.
        trees: Number of trees in a random forest.

    Returns:
        The best parameters, its fitted estimator, and one row of validation
        metrics for every tried setting.
    """
    grid = GRIDS[model]
    names = list(grid)
    rows = []
    best = None
    for values in itertools.product(*(grid[name] for name in names)):
        params = dict(zip(names, values))
        started = time.perf_counter()
        estimator = make_estimator(
            model, params, seed, n_jobs, trees, sparse.issparse(train_features)
        )
        fit_estimator(estimator, model, train_features, train_labels)
        elapsed = time.perf_counter() - started
        metrics = score_metrics(
            val_labels, predict_scores(estimator, model, val_features)
        )
        rows.append(
            {
                "model": model,
                "params": json.dumps(params),
                "seed": seed,
                "fit_seconds": elapsed,
                **{"val_" + name: value for name, value in metrics.items()},
            }
        )
        if best is None or metrics["auc"] > best[0] or np.isnan(best[0]):
            best = (metrics["auc"], params, estimator)
    return best[1], best[2], rows


def save_scores(directory: Optional[Path], tag: str, scores: Dict[str, np.ndarray]):
    """Save every read's score for each evaluated split, for genome-level analysis.

    Args:
        directory: Destination directory, or None to skip saving.
        tag: File name stem identifying the configuration and seed.
        scores: Score array for each split name.
    """
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / (tag + ".npz"),
        **{name: values.astype(np.float32) for name, values in scores.items()},
    )
