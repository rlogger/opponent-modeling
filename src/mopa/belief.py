"""Uncertainty-aware belief over opponent strategy (Part 1).

Converts encoder latents (or trajectory features) into a soft posterior over
discrete strategy types, with entropy for calibration / anytime curves.

The primary path fits a logistic probe on frozen latents (labels used only for
the probe head at eval time). For unsupervised use, call
:func:`softmax_belief_from_logits` with any scoring head.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class Belief:
    """Soft posterior over K strategy types."""

    probs: np.ndarray  # (N, K)
    entropy: np.ndarray  # (N,)
    classes: np.ndarray  # (K,)

    @property
    def hard(self) -> np.ndarray:
        return np.asarray(self.classes)[self.probs.argmax(axis=1)]

    def mean_entropy(self) -> float:
        return float(self.entropy.mean())


def categorical_entropy(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-row entropy in nats for a batch of categorical distributions."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1).astype(np.float32)


def softmax_belief_from_logits(logits: np.ndarray, classes: np.ndarray | None = None) -> Belief:
    """Build a :class:`Belief` from unnormalized logits ``(N, K)``."""
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    k = probs.shape[-1]
    if classes is None:
        classes = np.arange(k, dtype=np.int32)
    return Belief(probs=probs, entropy=categorical_entropy(probs), classes=classes)


def fit_latent_belief(
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_query: np.ndarray | None = None,
) -> Belief:
    """Fit a linear probe on latents and return soft posteriors on ``z_query``.

    If ``z_query`` is None, returns beliefs on the training latents.
    """
    z_train = np.asarray(z_train, dtype=np.float32)
    y_train = np.asarray(y_train)
    query = z_train if z_query is None else np.asarray(z_query, dtype=np.float32)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z_train, y_train)
    classes = clf.classes_.astype(np.int32)
    proba = clf.predict_proba(query).astype(np.float32)
    return Belief(
        probs=proba,
        entropy=categorical_entropy(proba),
        classes=classes,
    )


def belief_from_partial_trajectory(
    encoder_fn,
    traj_features: np.ndarray,
    probe_clf: LogisticRegression,
) -> Belief:
    """Encode partial trajectories then map latents → soft strategy posterior.

    ``encoder_fn(traj_features) -> z`` with ``z`` shape ``(N, lat)``.
    ``probe_clf`` must already be fitted on strategy labels.
    """
    z = np.asarray(encoder_fn(traj_features), dtype=np.float32)
    classes = probe_clf.classes_.astype(np.int32)
    proba = probe_clf.predict_proba(z).astype(np.float32)
    return Belief(
        probs=proba,
        entropy=categorical_entropy(proba),
        classes=classes,
    )
