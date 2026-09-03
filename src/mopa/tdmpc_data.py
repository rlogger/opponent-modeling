"""Episode-bounded sequence replay and model-error evaluation for TD-MPC.

Replaces the upstream sequential buffer (which deliberately let samples cross
episode boundaries) with a sampler over the continuous dataset that never
crosses an episode: windows start at a valid transition and are padded past
the episode end with the terminal flags placed at the last valid transition,
so ``TDMPC2.update``'s ``finished`` mask excludes the padding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mopa.context import CONTEXT_DIM

__all__ = [
    "SequenceReplay",
    "attach_context",
    "multistep_model_error",
    "reward_calibration",
    "state_statistics",
    "termination_calibration",
]


def attach_context(
    data: dict[str, np.ndarray],
    context: np.ndarray | None,
    *,
    source: str,
) -> np.ndarray:
    """Return a ``(N, T + 1, C)`` context tensor for the world-model dataset.

    ``source="causal"`` uses the supplied frozen-encoder context;
    ``"oracle"`` broadcasts the objective one-hot; ``"zero"`` gives zeros.
    """
    n, t_plus = data["state"].shape[:2]
    if source == "causal":
        if context is None:
            raise ValueError("causal context requires an encoder output")
        ctx = np.asarray(context, np.float32)
        if ctx.shape != (n, t_plus, CONTEXT_DIM):
            raise ValueError(f"context must have shape {(n, t_plus, CONTEXT_DIM)}")
        return ctx
    if source == "oracle":
        one_hot = np.eye(CONTEXT_DIM, dtype=np.float32)[np.asarray(data["objective_label"])]
        return np.repeat(one_hot[:, None, :], t_plus, axis=1)
    if source == "zero":
        return np.zeros((n, t_plus, CONTEXT_DIM), np.float32)
    raise ValueError("context source must be causal, oracle, or zero")


@dataclass
class SequenceReplay:
    """Uniform-over-valid-transitions sampler of ``horizon``-step windows."""

    state: np.ndarray  # (N, T+1, D)
    blue_action: np.ndarray  # (N, T, 2)
    red_action: np.ndarray  # (N, T, 2)
    reward: np.ndarray  # (N, T)
    terminated: np.ndarray  # (N, T) bool
    truncated: np.ndarray  # (N, T) bool
    valid_length: np.ndarray  # (N,)
    context: np.ndarray  # (N, T+1, C)
    horizon: int

    @classmethod
    def from_dataset(
        cls,
        data: dict[str, np.ndarray],
        episodes: np.ndarray,
        horizon: int,
        context: np.ndarray,
    ) -> "SequenceReplay":
        idx = np.asarray(episodes)
        return cls(
            state=np.asarray(data["state"])[idx],
            blue_action=np.asarray(data["blue_action"])[idx],
            red_action=np.asarray(data["red_action"])[idx],
            reward=np.asarray(data["blue_reward"])[idx],
            terminated=np.asarray(data["terminated_capture"])[idx],
            truncated=np.asarray(data["truncated_timeout"])[idx],
            valid_length=np.asarray(data["valid_length"])[idx],
            context=np.asarray(context, np.float32)[idx],
            horizon=int(horizon),
        )

    @property
    def n_transitions(self) -> int:
        return int(self.valid_length.sum())

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, jax.Array]:
        """Return ``(horizon, batch, ...)`` arrays; windows never cross episodes."""
        p = self.valid_length / self.valid_length.sum()
        eps = rng.choice(len(self.valid_length), size=batch_size, p=p)
        starts = np.floor(rng.random(batch_size) * self.valid_length[eps]).astype(np.int64)
        offsets = np.arange(self.horizon)[None, :]
        raw_t = starts[:, None] + offsets  # (B, H)
        last = self.valid_length[eps][:, None] - 1
        inside = raw_t <= last
        t = np.minimum(raw_t, last)
        e = eps[:, None]
        batch = {
            "observations": self.state[e, t],
            "next_observations": self.state[e, t + 1],
            "actions": self.blue_action[e, t],
            "red_actions": self.red_action[e, t],
            "rewards": self.reward[e, t],
            "terminated": self.terminated[e, t] & inside,
            "truncated": self.truncated[e, t] & inside,
            "context": self.context[e, t],
            "next_context": self.context[e, t + 1],
        }
        return {k: jnp.asarray(np.swapaxes(v, 0, 1)) for k, v in batch.items()}


def state_statistics(states: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean / std of the Markov state over valid transitions' start states."""
    s = np.asarray(states)[:, :-1][np.asarray(valid_mask, bool)]
    mean = s.mean(axis=0).astype(np.float32)
    std = (s.std(axis=0) + 1e-3).astype(np.float32)
    return mean, std


# --------------------------------------------------------------------------- #
# Model-error evaluation on recorded trajectories (component ladder rung 3)
# --------------------------------------------------------------------------- #
def _encode(agent, obs: np.ndarray, key) -> jax.Array:  # noqa: ANN001
    return agent.model.encode(jnp.asarray(obs), agent.model.encoder.params, key)


def multistep_model_error(
    agent: Any,
    replay: SequenceReplay,
    *,
    horizons: tuple[int, ...] = (1, 3, 10),
    max_starts: int = 4096,
    seed: int = 0,
    obs_std: np.ndarray | None = None,
) -> dict[str, Any]:
    """Open-loop latent prediction error with recorded joint actions vs persistence.

    For each ``k`` in ``horizons`` and each valid start ``t`` with
    ``t + k <= valid_length``: roll the learned dynamics ``k`` steps from
    ``encode(s_t)`` using the recorded blue (and, in factored mode, recorded
    red) actions and compare to ``encode(s_{t+k})``. Persistence predicts
    ``encode(s_t)``. Errors are mean squared errors in latent units; in the
    identity-encoder baseline the latent is the normalized state, so the
    position-only RMSE in arena units is reported as well.
    """
    rng = np.random.default_rng(seed)
    model = agent.model
    key = jax.random.PRNGKey(seed)
    out: dict[str, Any] = {"horizons": list(horizons), "per_horizon": {}}
    for k in horizons:
        eligible = [
            (e, t)
            for e in range(len(replay.valid_length))
            for t in range(int(replay.valid_length[e]) - k + 1)
        ]
        if not eligible:
            out["per_horizon"][str(k)] = None
            continue
        sel = np.asarray(eligible)
        if len(sel) > max_starts:
            sel = sel[rng.choice(len(sel), size=max_starts, replace=False)]
        e, t = sel[:, 0], sel[:, 1]
        x = _encode(agent, replay.state[e, t], key)
        x0 = x
        for step in range(k):
            u = jnp.asarray(replay.blue_action[e, t + step])
            c = jnp.asarray(replay.context[e, t + step])
            v = jnp.asarray(replay.red_action[e, t + step])
            x = model.next(x, model.transition_inputs(u, c, v), model.dynamics_model.params)
        target = _encode(agent, replay.state[e, t + k], key)
        model_err = np.asarray(jnp.mean((x - target) ** 2, axis=-1))
        persist_err = np.asarray(jnp.mean((x0 - target) ** 2, axis=-1))
        row: dict[str, Any] = {
            "n_starts": int(len(sel)),
            "model_mse": float(model_err.mean()),
            "persistence_mse": float(persist_err.mean()),
            "model_beats_persistence": bool(model_err.mean() < persist_err.mean()),
            "ratio_model_over_persistence": float(model_err.mean() / max(persist_err.mean(), 1e-12)),
        }
        if model.encoder_type == "identity" and obs_std is not None:
            std = jnp.asarray(obs_std, jnp.float32)
            pos = slice(0, 4)  # agent positions (P + 1 = 2 agents)
            d_model = (x[:, pos] - target[:, pos]) * std[pos]
            d_persist = (x0[:, pos] - target[:, pos]) * std[pos]
            row["position_rmse_model"] = float(jnp.sqrt(jnp.mean(d_model**2)))
            row["position_rmse_persistence"] = float(jnp.sqrt(jnp.mean(d_persist**2)))
        out["per_horizon"][str(k)] = row
    return out


def _binned_calibration(pred: np.ndarray, actual: np.ndarray, bins: int = 10) -> dict[str, Any]:
    edges = np.quantile(pred, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    which = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = which == b
        if sel.any():
            rows.append(
                {
                    "bin": b,
                    "count": int(sel.sum()),
                    "predicted_mean": float(pred[sel].mean()),
                    "actual_mean": float(actual[sel].mean()),
                }
            )
    return {"bins": rows}


def reward_calibration(
    agent: Any, replay: SequenceReplay, *, max_samples: int = 8192, seed: int = 0
) -> dict[str, Any]:
    """One-step reward prediction from encoded states on recorded transitions."""
    rng = np.random.default_rng(seed)
    model = agent.model
    e, t = np.nonzero(np.arange(replay.state.shape[1] - 1)[None, :] < replay.valid_length[:, None])
    if len(e) > max_samples:
        pick = rng.choice(len(e), size=max_samples, replace=False)
        e, t = e[pick], t[pick]
    x = _encode(agent, replay.state[e, t], jax.random.PRNGKey(seed))
    a = model.transition_inputs(
        jnp.asarray(replay.blue_action[e, t]),
        jnp.asarray(replay.context[e, t]),
        jnp.asarray(replay.red_action[e, t]),
    )
    pred = np.asarray(model.reward(x, a, model.reward_model.params)[0])
    actual = replay.reward[e, t]
    resid = pred - actual
    return {
        "n_samples": int(len(e)),
        "mse": float(np.mean(resid**2)),
        "mae": float(np.mean(np.abs(resid))),
        "actual_variance": float(np.var(actual)),
        "explained_variance": float(1.0 - np.var(resid) / max(np.var(actual), 1e-12)),
        "calibration": _binned_calibration(pred, actual),
    }


def termination_calibration(
    agent: Any, replay: SequenceReplay, *, max_samples: int = 8192, seed: int = 0
) -> dict[str, Any] | None:
    """Continuation-probability calibration against ``1 - terminated_capture``."""
    model = agent.model
    if not model.predict_continues:
        return None
    rng = np.random.default_rng(seed)
    e, t = np.nonzero(np.arange(replay.state.shape[1] - 1)[None, :] < replay.valid_length[:, None])
    if len(e) > max_samples:
        # Keep every capture transition (rare) plus a random subsample of the rest.
        cap = replay.terminated[e, t]
        keep_cap = np.flatnonzero(cap)
        rest = np.flatnonzero(~cap)
        pick = np.concatenate([keep_cap, rng.choice(rest, size=max(max_samples - len(keep_cap), 0), replace=False)])
        e, t = e[pick], t[pick]
    x = _encode(agent, replay.state[e, t], jax.random.PRNGKey(seed))
    a = model.transition_inputs(
        jnp.asarray(replay.blue_action[e, t]),
        jnp.asarray(replay.context[e, t]),
        jnp.asarray(replay.red_action[e, t]),
    )
    p_cont = np.asarray(jax.nn.sigmoid(model.continue_logits(x, a, model.continue_model.params)))
    y = 1.0 - replay.terminated[e, t].astype(np.float64)
    brier = float(np.mean((p_cont - y) ** 2))
    # ECE over ten equal-width probability bins.
    bins = np.clip((p_cont * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        sel = bins == b
        if sel.any():
            ece += sel.mean() * abs(p_cont[sel].mean() - y[sel].mean())
    # AUROC for predicting capture (1 - y) with 1 - p_cont.
    score, label = 1.0 - p_cont, 1.0 - y
    pos, neg = score[label > 0.5], score[label < 0.5]
    auroc = None
    if len(pos) and len(neg):
        auroc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    return {
        "n_samples": int(len(e)),
        "n_capture_transitions": int((label > 0.5).sum()),
        "brier": brier,
        "ece_10_bins": float(ece),
        "capture_auroc": auroc,
        "hard_threshold_accuracy": float(np.mean((p_cont > 0.5) == (y > 0.5))),
        "capture_recall_at_0.5": (
            float(np.mean(p_cont[label > 0.5] <= 0.5)) if (label > 0.5).any() else None
        ),
    }
