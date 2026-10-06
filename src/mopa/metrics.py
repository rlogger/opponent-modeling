"""Leakage-safe representation, uncertainty, and policy metrics."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score


def probe_acc(z, y, cv=5):
    """Linear probe: tests how usable the information in the latent space is."""
    return float(
        cross_val_score(LogisticRegression(max_iter=1000), z, y, cv=cv).mean()
    )


def oracle_acc(X, y, cv=5, hidden=128, seed=0):
    """Supervised MLP baseline predicting ground-truth labels from trajectories."""
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier((hidden,), max_iter=800, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv).mean())


def survival_time_probe_acc(survival_time, y, cv=5):
    """Shortcut control: linear probe on episode length / survival time alone."""
    x = np.asarray(survival_time, dtype=np.float32).reshape(-1, 1)
    return probe_acc(x, y, cv=cv)


def train_only_survival_time_probe_acc(
    survival_train: np.ndarray,
    y_train: np.ndarray,
    survival_query: np.ndarray,
    y_query: np.ndarray,
) -> float:
    """Fit the episode-length shortcut only on train and score the holdout."""
    train = np.asarray(survival_train, dtype=np.float32).reshape(-1, 1)
    query = np.asarray(survival_query, dtype=np.float32).reshape(-1, 1)
    scaled_train, scaled_query, _, _ = train_only_standardize(train, query)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(scaled_train, y_train)
    return float(accuracy_score(y_query, clf.predict(scaled_query)))


def metrics(z, y, n_classes):
    """Return ``(probe_accuracy, GMM_ARI)`` on centered latents.

    Prefer :func:`train_only_metrics` for published numbers (avoids full-pool
    centering leakage).
    """
    zc = (z - z.mean(0)) / (z.std(0) + 1e-6)
    probe = probe_acc(zc, y)
    gm = GaussianMixture(n_classes, n_init=8, random_state=0).fit(zc)
    return probe, float(adjusted_rand_score(y, gm.predict(zc)))


def train_only_standardize(
    z_train: np.ndarray,
    z_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Center/scale with train statistics only."""
    mu = z_train.mean(0)
    sd = z_train.std(0) + 1e-6
    return ((z_train - mu) / sd).astype(np.float32), (
        (z_query - mu) / sd
    ).astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def train_only_metrics(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_query: np.ndarray,
    y_query: np.ndarray,
    n_classes: int,
    seed: int = 0,
) -> dict[str, float]:
    """Fit probe + GMM on train latents; score held-out query."""
    zt, zq, _, _ = train_only_standardize(z_train, z_query)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(zt, y_train)
    probe = float(accuracy_score(y_query, clf.predict(zq)))
    gm = GaussianMixture(n_classes, n_init=8, random_state=seed).fit(zt)
    ari = float(adjusted_rand_score(y_query, gm.predict(zq)))
    return {"probe": probe, "ari": ari}


def train_only_oracle_acc(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    *,
    hidden: int = 128,
    seed: int = 0,
) -> float:
    """Fit the supervised trajectory ceiling on train examples only."""
    from sklearn.neural_network import MLPClassifier

    Xt, Xq, _, _ = train_only_standardize(X_train, X_query)
    clf = MLPClassifier((hidden,), max_iter=800, random_state=seed)
    clf.fit(Xt, y_train)
    return float(accuracy_score(y_query, clf.predict(Xq)))


def _validated_probs(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true)
    if p.ndim != 2 or y.shape != (len(p),) or y.dtype.kind not in "iu":
        raise ValueError("probs must have shape (N, K) aligned with y_true")
    if len(y) == 0:
        raise ValueError("probability metrics require at least one example")
    if p.shape[1] < 2 or np.any(p < 0.0) or not np.all(np.isfinite(p)):
        raise ValueError("probs must contain finite non-negative class probabilities")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("y_true contains a class outside probs")
    row_sum = p.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError("each probability row must have positive mass")
    return p / row_sum, y.astype(np.int64)


def multiclass_nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-9) -> float:
    """Mean negative log-likelihood for multiclass probabilities."""
    p, y = _validated_probs(probs, y_true)
    return -float(np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0)).mean())


def multiclass_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean multiclass Brier score (sum of squared class errors)."""
    p, y = _validated_probs(probs, y_true)
    target = np.eye(p.shape[1], dtype=np.float64)[y]
    return float(np.sum((p - target) ** 2, axis=1).mean())


def reliability_bins(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float] | list[int]]:
    """Return JSON-ready top-label reliability data for plotting/auditing."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _validated_probs(probs, y_true)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = pred == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts: list[int] = []
    accuracy: list[float] = []
    confidence: list[float] = []
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (
            conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1]
        )
        counts.append(int(mask.sum()))
        accuracy.append(float(correct[mask].mean()) if np.any(mask) else 0.0)
        confidence.append(float(conf[mask].mean()) if np.any(mask) else 0.0)
    return {
        "bin_edges": [float(x) for x in edges],
        "count": counts,
        "accuracy": accuracy,
        "confidence": confidence,
    }


def expected_calibration_error(probs, y_true, n_bins=10):
    """ECE for multiclass soft predictions ``probs`` of shape ``(N, K)``."""
    bins = reliability_bins(probs, y_true, n_bins=n_bins)
    total = max(1, sum(bins["count"]))
    return float(
        sum(
            (n / total) * abs(acc - conf)
            for n, acc, conf in zip(
                bins["count"], bins["accuracy"], bins["confidence"]
            )
        )
    )


def classwise_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """One-vs-rest ECE averaged across classes."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    p, y = _validated_probs(probs, y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    values = []
    for k in range(p.shape[1]):
        conf = p[:, k]
        target = y == k
        ece = 0.0
        for i in range(n_bins):
            mask = (conf >= edges[i]) & (
                conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1]
            )
            if np.any(mask):
                ece += float(mask.mean()) * abs(
                    float(target[mask].mean()) - float(conf[mask].mean())
                )
        values.append(ece)
    return float(np.mean(values))


def temperature_scale_logits(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    grid_size: int = 241,
) -> float:
    """Fit one positive temperature on held-out calibration logits.

    A deterministic log-spaced grid keeps this utility dependency-light and
    makes the selected value exactly reproducible.
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true)
    if x.ndim != 2 or y.shape != (len(x),) or y.dtype.kind not in "iu" or grid_size < 3:
        raise ValueError("logits/y shape mismatch or grid_size < 3")
    candidates = np.exp(np.linspace(-3.0, 3.0, grid_size))
    losses = [multiclass_nll(softmax_with_temperature(x, t), y) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Convert logits to probabilities using a positive temperature."""
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    x = np.asarray(logits, dtype=np.float64) / float(temperature)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float32)


def calibration_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, object]:
    """Complete held-out calibration summary plus reliability data."""
    return {
        "nll": multiclass_nll(probs, y_true),
        "brier": multiclass_brier(probs, y_true),
        "ece": expected_calibration_error(probs, y_true, n_bins=n_bins),
        "classwise_ece": classwise_ece(probs, y_true, n_bins=n_bins),
        "reliability": reliability_bins(probs, y_true, n_bins=n_bins),
    }


def open_set_auroc(known_score: np.ndarray, is_known: np.ndarray) -> float:
    """AUROC for a declared known/unknown score (higher means more known)."""
    score = np.asarray(known_score, dtype=np.float64)
    target = np.asarray(is_known, dtype=bool)
    if score.shape != target.shape or len(np.unique(target)) != 2:
        raise ValueError("open-set AUROC needs aligned scores and both classes")
    return float(roc_auc_score(target.astype(np.int32), score))
