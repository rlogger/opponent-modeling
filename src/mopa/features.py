"""Pure trajectory feature extraction utilities (NumPy-only)."""
from __future__ import annotations

import numpy as np

EP_LEN = 100
OCC_BINS = 8
OCC_RANGE = (-2.0, 2.0)


def episode_lengths(
    capture_t: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """Valid step counts matching ``survival_time`` in rollouts.

    ``capture_t[i] >= 0`` means episode ``i`` ended at that env step; otherwise
    the episode ran the full ``num_steps`` horizon. Lengths are clipped to
    ``[1, num_steps]`` for safe sequence indexing.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    ct = np.asarray(capture_t, dtype=np.int32)
    captured = ct >= 0
    lengths = np.where(captured, ct, num_steps).astype(np.int32)
    return np.clip(lengths, 1, num_steps)


def length_matched_prefix(
    lengths: np.ndarray,
    k: int,
) -> np.ndarray:
    """Clip requested prefix ``k`` to each episode's true length."""
    if k < 1:
        raise ValueError("k must be >= 1")
    lengths = np.asarray(lengths, dtype=np.int32)
    return np.minimum(lengths, k).astype(np.int32)


def predator_sequence_features(
    prey_pos: np.ndarray,
    pred_pos: np.ndarray,
    lengths: np.ndarray | None = None,
    *,
    velocity_mode: str = "legacy_forward",
) -> np.ndarray:
    """Build per-step sequence features ``(N, T, F)`` from positions.

    Uses action-horizon steps ``t = 0..T-1`` with positions at ``t`` (after the
    previous transition / initial state). ``velocity_mode='legacy_forward'``
    preserves the original feature ``p[t + 1] - p[t]``.  The causal mode uses
    only information available at state ``t``: zero velocity initially and
    ``p[t] - p[t - 1]`` thereafter. Post-length steps are zeroed when
    ``lengths`` is provided.
    """
    if prey_pos.ndim != 3 or prey_pos.shape[-1] != 2:
        raise ValueError("prey_pos must have shape (N, T + 1, 2)")
    if pred_pos.ndim != 4 or pred_pos.shape[-1] != 2:
        raise ValueError("pred_pos must have shape (N, T + 1, P, 2)")
    if prey_pos.shape[:2] != pred_pos.shape[:2]:
        raise ValueError("prey_pos and pred_pos horizon must match")

    n, t_plus, _ = prey_pos.shape
    t_max = t_plus - 1
    if velocity_mode not in {"causal_past", "legacy_forward"}:
        raise ValueError(
            "velocity_mode must be 'causal_past' or 'legacy_forward'"
        )

    # Absolute positions at each action step + one-step velocity proxy.
    prey_t = prey_pos[:, :t_max]
    pred_t = pred_pos[:, :t_max].reshape(n, t_max, -1)
    if velocity_mode == "legacy_forward":
        prey_v = prey_pos[:, 1 : t_max + 1] - prey_pos[:, :t_max]
        pred_v = (
            pred_pos[:, 1 : t_max + 1] - pred_pos[:, :t_max]
        ).reshape(n, t_max, -1)
    else:
        prey_v = np.zeros_like(prey_t)
        pred_v = np.zeros_like(pred_pos[:, :t_max])
        prey_v[:, 1:] = prey_pos[:, 1:t_max] - prey_pos[:, : t_max - 1]
        pred_v[:, 1:] = pred_pos[:, 1:t_max] - pred_pos[:, : t_max - 1]
        pred_v = pred_v.reshape(n, t_max, -1)
    seq = np.concatenate([prey_t, pred_t, prey_v, pred_v], axis=-1).astype(np.float32)
    if lengths is not None:
        lens = np.clip(np.asarray(lengths, np.int32), 1, t_max)
        mask = np.arange(t_max)[None, :] < lens[:, None]
        seq = seq * mask[..., None].astype(np.float32)
    return seq


def window(pos: np.ndarray, k: int, kmax: int | None = None) -> np.ndarray:
    """Flatten a fixed first-episode window with masked future steps."""
    if kmax is None:
        kmax = EP_LEN
    if not 0 <= k <= kmax:
        raise ValueError(f"k must be in [0, {kmax}], got {k}")
    ep = pos[:, :kmax]
    flat = ep.reshape(len(ep), kmax, -1).copy()
    flat[:, k:] = 0.0
    return flat.reshape(len(ep), -1).astype(np.float32)


def sequence_window(
    sequence: np.ndarray,
    start: int,
    stop: int,
    *,
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Flatten one masked interval from a shared trajectory feature tensor.

    All encoder baselines and the supervised oracle can use this function so
    they see the same ``(prey, predator, velocity)`` schema.  Values outside
    ``[start, stop)`` and beyond an episode's valid length are zero.
    """
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("sequence must have shape (N, T, F)")
    if not 0 <= start < stop <= x.shape[1]:
        raise ValueError(f"invalid sequence interval [{start}, {stop})")
    mask = np.zeros(x.shape[:2], dtype=bool)
    mask[:, start:stop] = True
    if lengths is not None:
        lens = np.clip(np.asarray(lengths, dtype=np.int32), 1, x.shape[1])
        if lens.shape != (len(x),):
            raise ValueError("lengths must have shape (N,)")
        mask &= np.arange(x.shape[1])[None, :] < lens[:, None]
    return (x * mask[..., None]).reshape(len(x), -1).astype(np.float32)


def sequence_slice(
    sequence: np.ndarray,
    start: int,
    width: int,
    *,
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Pack a fixed-width trajectory interval into relative-time features."""
    x = np.asarray(sequence, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("sequence must have shape (N, T, F)")
    if start < 0 or width < 1 or start + width > x.shape[1]:
        raise ValueError("slice must fit inside the sequence horizon")
    out = np.array(x[:, start : start + width], copy=True)
    if lengths is not None:
        lens = np.asarray(lengths, dtype=np.int32)
        if lens.shape != (len(x),):
            raise ValueError("lengths must have shape (N,)")
        valid = start + np.arange(width)[None, :] < lens[:, None]
        out *= valid[..., None].astype(np.float32)
    return out.reshape(len(x), -1).astype(np.float32)


def trailing_sequence_slice(
    sequence: np.ndarray,
    ends: np.ndarray,
    width: int,
) -> np.ndarray:
    """Pack each episode's latest valid window ending at ``ends[i]``.

    Short prefixes are left-aligned and zero-padded. Longer prefixes retain the
    latest ``width`` steps, which makes a fixed-window baseline genuinely
    online instead of remaining frozen on the first window forever.
    """
    x = np.asarray(sequence, dtype=np.float32)
    end_array = np.asarray(ends, dtype=np.int32)
    if x.ndim != 3:
        raise ValueError("sequence must have shape (N, T, F)")
    if width < 1 or width > x.shape[1]:
        raise ValueError("width must fit inside the sequence horizon")
    if end_array.shape != (len(x),):
        raise ValueError("ends must have shape (N,)")
    if np.any(end_array < 1) or np.any(end_array > x.shape[1]):
        raise ValueError("ends must be within the sequence horizon")

    output = np.zeros((len(x), width, x.shape[-1]), dtype=np.float32)
    for index, end in enumerate(end_array):
        start = max(0, int(end) - width)
        values = x[index, start : int(end)]
        output[index, : len(values)] = values
    return output.reshape(len(x), -1)


def standardize(
    X: np.ndarray,
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features with train-set statistics."""
    if mu is None:
        mu = X.mean(0)
    if sd is None:
        sd = X.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32), mu, sd


def occupancy(
    pos: np.ndarray,
    t0: int,
    t1: int,
    bins: int = OCC_BINS,
    value_range: tuple[float, float] = OCC_RANGE,
) -> np.ndarray:
    """Per-episode normalized 2-D occupancy histogram over steps ``[t0, t1)``."""
    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError("pos must have shape (N, T + 1, 2)")
    if not 0 <= t0 < t1 <= pos.shape[1]:
        raise ValueError(f"invalid occupancy interval [{t0}, {t1})")

    out = np.zeros((len(pos), bins * bins), np.float32)
    hist_range = [value_range, value_range]
    for i in range(len(pos)):
        h, _, _ = np.histogram2d(
            pos[i, t0:t1, 0],
            pos[i, t0:t1, 1],
            bins=bins,
            range=hist_range,
        )
        h = h.ravel().astype(np.float32)
        total = h.sum()
        out[i] = h / total if total > 0 else h
    return out
