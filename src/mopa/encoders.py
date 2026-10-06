"""Self-supervised opponent-trajectory encoders: VAE (reconstruct) vs JEPA
(predict a future prefix's representation).

Protocol upgrades for Part 1 foundation:

  * multi-seed reporting;
  * GRU encoder with length/pad masking;
  * unit-sphere latents + L2-on-normalized prediction (no VICReg);
  * online encode API returning frozen params;
  * collapse diagnostics (across-episode std, effective rank).

Labels remain evaluation-only for the primary SSL path.
"""
from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal

HID = 64
STEPS = 5000
BATCH = 128
EMA = 0.996


class Enc(nn.Module):
    """MLP context encoder with LayerNorm on the hidden layers."""

    lat: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.LayerNorm()(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x)))
        x = nn.relu(nn.LayerNorm()(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x)))
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(x)


class EncVAE(nn.Module):
    lat: int = 2

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        x = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(x))
        return nn.Dense(self.lat)(x), nn.Dense(self.lat)(x)  # mu, logvar


class Dec(nn.Module):
    out: int

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        z = nn.relu(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z))
        return nn.Dense(self.out)(z)


class Pred(nn.Module):
    lat: int = 2

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.LayerNorm()(nn.Dense(HID, kernel_init=orthogonal(np.sqrt(2)))(z)))
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(z)


class GRUEnc(nn.Module):
    """GRU sequence encoder for variable-length trajectories.

    Takes padded sequences ``x`` of shape ``(N, T, F)`` and returns per-prefix
    latents of shape ``(N, T, lat)``: output ``[:, k - 1]`` encodes the first
    ``k`` steps. When ``lengths`` is provided, timesteps ``t >= lengths[i]``
    freeze the hidden state (pad-safe).

    Hand-rolled GRU (explicit kernels + ``lax.scan``) avoids Flax ``nn.RNN`` /
    ``nn.scan`` APIs that require newer JAX than JaxMARL currently pins.
    """

    lat: int = 2
    hid: int = HID

    @nn.compact
    def __call__(self, x, lengths=None):
        # x: (N, T, F)
        x = nn.relu(
            nn.LayerNorm()(
                nn.Dense(self.hid, kernel_init=orthogonal(np.sqrt(2)))(x)
            )
        )
        wz = self.param("wz", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        bz = self.param("bz", nn.initializers.zeros, (self.hid,))
        wr = self.param("wr", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        br = self.param("br", nn.initializers.zeros, (self.hid,))
        wh = self.param("wh", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        bh = self.param("bh", nn.initializers.zeros, (self.hid,))
        uz = self.param("uz", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        ur = self.param("ur", orthogonal(np.sqrt(2)), (self.hid, self.hid))
        uh = self.param("uh", orthogonal(np.sqrt(2)), (self.hid, self.hid))

        n, t_max, _ = x.shape
        if lengths is None:
            valid = jnp.ones((n, t_max), dtype=bool)
        else:
            lens = jnp.asarray(lengths, dtype=jnp.int32)
            valid = jnp.arange(t_max)[None, :] < lens[:, None]

        def step(h, inputs):
            xt, step_valid = inputs
            z = jax.nn.sigmoid(xt @ wz + h @ uz + bz)
            r = jax.nn.sigmoid(xt @ wr + h @ ur + br)
            nh = jnp.tanh(xt @ wh + (r * h) @ uh + bh)
            h_new = (1.0 - z) * nh + z * h
            h_out = jnp.where(step_valid[:, None], h_new, h)
            return h_out, h_out

        h0 = jnp.zeros((n, self.hid), dtype=x.dtype)
        x_t = jnp.swapaxes(x, 0, 1)  # (T, N, H)
        valid_t = jnp.swapaxes(valid, 0, 1)  # (T, N)
        _, h_t = jax.lax.scan(step, h0, (x_t, valid_t))
        h = jnp.swapaxes(h_t, 0, 1)  # (N, T, H)
        h = nn.LayerNorm()(h)
        return nn.Dense(self.lat, kernel_init=orthogonal(1.0))(h)


def unit_normalize(z):
    """Map trajectory embeddings to the unit sphere for stable JEPA training."""
    return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-6)


def squared_l2_prediction_loss(predicted, target):
    """Mean squared L2 distance between paired embedding vectors."""
    predicted = jnp.asarray(predicted)
    target = jnp.asarray(target)
    if predicted.shape != target.shape:
        raise ValueError("predicted and target embeddings must have the same shape")
    return jnp.mean(jnp.sum((predicted - target) ** 2, axis=-1))


def latent_std_diagnostics(z: np.ndarray) -> dict[str, np.ndarray | float]:
    """Across-episode / per-dimension std and mean L2 norm of latents."""
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"expected z shape (N, lat), got {z.shape}")
    per_dim = z.std(axis=0)
    return {
        "per_dim_std": per_dim.astype(np.float32),
        "across_std_mean": float(per_dim.mean()),
        "mean_norm": float(np.linalg.norm(z, axis=-1).mean()),
    }


def effective_rank(z: np.ndarray, eps: float = 1e-12) -> float:
    """Participation-ratio effective rank of centered latents."""
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or z.shape[0] < 2:
        return 0.0
    zc = z - z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(zc, compute_uv=False)
    p = s**2
    total = float(p.sum())
    if total < eps:
        return 1.0  # collapsed / near-constant → rank 1
    p = p / total
    return float(1.0 / ((p**2).sum() + eps))


def collapse_diagnostics(z: np.ndarray) -> dict[str, float]:
    """Compact collapse summary for episode-level latents ``(N, lat)``."""
    stats = latent_std_diagnostics(z)
    return {
        "across_std_mean": float(stats["across_std_mean"]),
        "mean_norm": float(stats["mean_norm"]),
        "effective_rank": effective_rank(z),
    }


def gather_prefix_latents(z: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Select ``z[i, lengths[i] - 1]`` for each episode."""
    z = np.asarray(z)
    lengths = np.asarray(lengths, dtype=np.int32)
    idx = np.clip(lengths - 1, 0, z.shape[1] - 1)
    return z[np.arange(len(z)), idx]


def train_vae_with_params(Xc, rng, lat=2, steps=STEPS):
    """Fit beta-VAE on training contexts; return latents and encoder params."""
    enc, dec = EncVAE(lat=lat), Dec(out=Xc.shape[1])
    rng, ke, kd = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "d": dec.init(kd, jnp.zeros((1, lat)))}
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xj = jnp.asarray(Xc)
    n = len(Xc)

    def loss_fn(p, x, rng, beta):
        mu, lv = enc.apply(p["e"], x)
        z = mu + jnp.exp(0.5 * lv) * jax.random.normal(rng, mu.shape)
        rec = jnp.mean(jnp.sum((dec.apply(p["d"], z) - x) ** 2, -1))
        kl = jnp.mean(-0.5 * jnp.sum(1 + lv - mu**2 - jnp.exp(lv), -1))
        return rec + beta * kl

    @jax.jit
    def upd(params, opt, idx, rng, beta):
        g = jax.grad(loss_fn)(params, Xj[idx], rng, beta)
        u, opt = tx.update(g, opt)
        return optax.apply_updates(params, u), opt

    for s in range(steps):
        rng, bk, sk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, opt = upd(params, opt, idx, sk, float(min(1.0, s / 1500)))
    return np.asarray(enc.apply(params["e"], Xj)[0]), params["e"]


def encode_vae(params, X, lat=2) -> np.ndarray:
    """Frozen beta-VAE mean encoder."""
    return np.asarray(EncVAE(lat=lat).apply(params, jnp.asarray(X))[0])


def train_vae(Xc, rng, lat=2, steps=STEPS):
    """Compatibility wrapper returning training-set VAE means only."""
    z, _ = train_vae_with_params(Xc, rng, lat=lat, steps=steps)
    return z


def train_jepa_with_params(Xc, Xt, rng, lat=2, steps=STEPS):
    """Fit fixed-window JEPA; return training latents and online encoder params."""
    enc, pred = Enc(lat=lat), Pred(lat=lat)
    rng, ke, kp = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "p": pred.init(kp, jnp.zeros((1, lat)))}
    target = params["e"]
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xcj, Xtj = jnp.asarray(Xc), jnp.asarray(Xt)
    n = len(Xc)

    def loss_fn(p, tgt, xc, xt):
        z = unit_normalize(enc.apply(p["e"], xc))
        pz = unit_normalize(pred.apply(p["p"], z))
        t = jax.lax.stop_gradient(unit_normalize(enc.apply(tgt, xt)))
        return squared_l2_prediction_loss(pz, t)

    @jax.jit
    def upd(params, target, opt, idx):
        g = jax.grad(loss_fn)(params, target, Xcj[idx], Xtj[idx])
        u, opt = tx.update(g, opt)
        params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(
            lambda a, b: EMA * a + (1 - EMA) * b, target, params["e"]
        )
        return params, target, opt

    for s in range(steps):
        rng, bk = jax.random.split(rng)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, target, opt = upd(params, target, opt, idx)
    z = np.asarray(unit_normalize(enc.apply(params["e"], Xcj)))
    return z, params["e"]


def encode_jepa(params, X, lat=2) -> np.ndarray:
    """Frozen fixed-window JEPA encoder."""
    return np.asarray(unit_normalize(Enc(lat=lat).apply(params, jnp.asarray(X))))


def train_jepa(Xc, Xt, rng, lat=2, steps=STEPS):
    """Compatibility wrapper returning training-set window-JEPA latents."""
    z, _ = train_jepa_with_params(Xc, Xt, rng, lat=lat, steps=steps)
    return z


def train_jepa_gru_with_params(
    X,
    lengths,
    rng,
    lat=2,
    hid=HID,
    steps=STEPS,
    tmin=3,
    target_mode: str = "later_prefix",
):
    """Variable-length JEPA with a pad-masked GRU encoder.

    ``X`` is zero-padded ``(N, T, F)``; ``lengths`` gives each episode's true
    valid step count (``1..T``). Online / predicted / EMA-target latents are
    unit-normalized; loss is L2 on the sphere (no VICReg).

    ``target_mode``:
      * ``"later_prefix"`` (default): sample ``t_tgt ∈ (t_ctx, t_full]``
      * ``"full"``: always predict the full-episode embedding at ``t_full``

    Returns ``(z, online_params, target_ema)`` where ``z`` has shape
    ``(N, T, lat)``.
    """
    if target_mode not in ("later_prefix", "full"):
        raise ValueError(f"unknown target_mode={target_mode!r}")
    if tmin < 1:
        raise ValueError("tmin must be >= 1")

    enc, pred = GRUEnc(lat=lat, hid=hid), Pred(lat=lat)
    rng, ke, kp = jax.random.split(rng, 3)
    lengths_np = np.asarray(lengths, dtype=np.int32)
    if lengths_np.shape != (len(X),):
        raise ValueError("lengths must have shape (N,)")
    if np.any(lengths_np < 1) or np.any(lengths_np > X.shape[1]):
        raise ValueError("lengths must be within the padded sequence horizon")
    # A one-step episode has no strictly later target. Exclude it from JEPA
    # updates instead of lengthening it into zero padding; it can still be
    # encoded by the returned frozen encoder.
    eligible = lengths_np >= 2
    if not np.any(eligible):
        raise ValueError("JEPA training needs at least one sequence of length >= 2")
    X_train = np.asarray(X)[eligible]
    train_lengths = lengths_np[eligible]
    params = {
        "e": enc.init(ke, X_train[:1], train_lengths[:1]),
        "p": pred.init(kp, jnp.zeros((1, lat))),
    }
    target = params["e"]
    tx = optax.adam(1e-3)
    opt = tx.init(params)
    Xj = jnp.asarray(X_train)
    lens = jnp.asarray(train_lengths, jnp.int32)
    n = len(X_train)
    use_full = target_mode == "full"

    def loss_fn(p, tgt, x, batch_lens, t_ctx, t_tgt):
        zs = enc.apply(p["e"], x, batch_lens)
        z_ctx = unit_normalize(zs[jnp.arange(len(x)), t_ctx - 1])
        pz = unit_normalize(pred.apply(p["p"], z_ctx))
        zs_t = jax.lax.stop_gradient(enc.apply(tgt, x, batch_lens))
        z_tgt = unit_normalize(zs_t[jnp.arange(len(x)), t_tgt - 1])
        return squared_l2_prediction_loss(pz, z_tgt)

    @jax.jit
    def upd(params, target, opt, idx, tk):
        t_full = lens[idx]
        min_ctx = jnp.minimum(jnp.asarray(tmin, dtype=jnp.int32), t_full - 1)
        # Per-example integer sampling because short valid episodes can have a
        # smaller lower bound than ``tmin`` without exposing their padding.
        span = t_full - min_ctx
        context_key, target_key = jax.random.split(tk)
        uniform = jax.random.uniform(context_key, t_full.shape)
        t_ctx = min_ctx + jnp.floor(uniform * span).astype(jnp.int32)
        if use_full:
            t_tgt = t_full
        else:
            # Sample a strictly later valid prefix when room exists.
            t_tgt = jax.random.randint(target_key, t_full.shape, t_ctx + 1, t_full + 1)
        g = jax.grad(loss_fn)(params, target, Xj[idx], lens[idx], t_ctx, t_tgt)
        u, opt = tx.update(g, opt)
        params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(
            lambda a, b: EMA * a + (1 - EMA) * b, target, params["e"]
        )
        return params, target, opt

    for s in range(steps):
        rng, bk, tk = jax.random.split(rng, 3)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, target, opt = upd(params, target, opt, idx, tk)

    z = encode_jepa_gru(params["e"], X, lengths_np, lat=lat, hid=hid)
    return z, params["e"], target


def train_jepa_gru(
    X,
    lengths,
    rng,
    lat=2,
    hid=HID,
    steps=STEPS,
    tmin=3,
    target_mode: str = "later_prefix",
):
    """Compatibility wrapper: return only per-prefix latents ``(N, T, lat)``."""
    z, _, _ = train_jepa_gru_with_params(
        X,
        lengths,
        rng,
        lat=lat,
        hid=hid,
        steps=steps,
        tmin=tmin,
        target_mode=target_mode,
    )
    return z


def encode_jepa_gru(params, X, lengths=None, lat=2, hid=HID) -> np.ndarray:
    """Frozen online encode with optional pad lengths. Returns ``(N, T, lat)``."""
    enc = GRUEnc(lat=lat, hid=hid)
    Xj = jnp.asarray(X)
    if lengths is None:
        z = enc.apply(params, Xj)
    else:
        z = enc.apply(params, Xj, jnp.asarray(lengths, dtype=jnp.int32))
    return np.asarray(unit_normalize(z))


def supervised_contrastive_loss(z, labels, temperature=0.20):
    """Identity-grounded contrastive loss for an episode-level latent.

    Labels identify positive pairs only. This path is a supervised ceiling and
    must not be reported on the SSL axis.
    """
    z = unit_normalize(z)
    logits = z @ z.T / temperature
    n = logits.shape[0]
    eye = jnp.eye(n, dtype=bool)
    same = labels[:, None] == labels[None, :]
    positives = same & (~eye)
    log_prob = jax.nn.log_softmax(jnp.where(eye, -1e9, logits), axis=1)
    per_row = -jnp.sum(positives * log_prob, axis=1) / jnp.maximum(
        positives.sum(axis=1), 1
    )
    return jnp.mean(per_row)


def train_identity_jepa_with_params(
    Xc,
    Xt,
    identity,
    rng,
    lat=2,
    steps=STEPS,
    contrastive_weight=1.0,
    *,
    uses_eval_labels: bool = False,
):
    """Train a stable JEPA encoder with an identity-grounded contrastive term.

    Requires ``uses_eval_labels=True`` to acknowledge this is a supervised
    ceiling, not the primary SSL encoder.
    """
    if not uses_eval_labels:
        raise ValueError(
            "train_identity_jepa_with_params uses evaluation labels; "
            "pass uses_eval_labels=True to acknowledge the supervised ceiling"
        )

    enc, pred = Enc(lat=lat), Pred(lat=lat)
    rng, ke, kp = jax.random.split(rng, 3)
    params = {"e": enc.init(ke, Xc[:1]), "p": pred.init(kp, jnp.zeros((1, lat)))}
    target = params["e"]
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-3, weight_decay=1e-5),
    )
    opt = tx.init(params)
    Xcj, Xtj = jnp.asarray(Xc), jnp.asarray(Xt)
    yj = jnp.asarray(identity, dtype=jnp.int32)
    n = len(Xc)

    def loss_fn(p, tgt, xc, xt, label):
        z = unit_normalize(enc.apply(p["e"], xc))
        pz = unit_normalize(pred.apply(p["p"], z))
        target_z = jax.lax.stop_gradient(unit_normalize(enc.apply(tgt, xt)))
        jepa = jnp.mean(1.0 - jnp.sum(pz * target_z, axis=-1))
        contrastive = supervised_contrastive_loss(z, label)
        return jepa + contrastive_weight * contrastive

    @jax.jit
    def upd(params, target, opt, idx):
        g = jax.grad(loss_fn)(params, target, Xcj[idx], Xtj[idx], yj[idx])
        u, opt = tx.update(g, opt, params)
        params = optax.apply_updates(params, u)
        target = jax.tree_util.tree_map(
            lambda a, b: EMA * a + (1 - EMA) * b, target, params["e"]
        )
        return params, target, opt

    for _ in range(steps):
        rng, bk = jax.random.split(rng)
        idx = jax.random.choice(bk, n, (min(BATCH, n),), replace=False)
        params, target, opt = upd(params, target, opt, idx)
    return np.asarray(unit_normalize(enc.apply(params["e"], Xcj))), params["e"]


def train_identity_jepa(
    Xc,
    Xt,
    identity,
    rng,
    lat=2,
    steps=STEPS,
    contrastive_weight=1.0,
    *,
    uses_eval_labels: bool = False,
):
    """Return the ID-JEPA latent when rollout-time encoder parameters are not needed."""
    z, _ = train_identity_jepa_with_params(
        Xc,
        Xt,
        identity,
        rng,
        lat=lat,
        steps=steps,
        contrastive_weight=contrastive_weight,
        uses_eval_labels=uses_eval_labels,
    )
    return z


def encode_identity_jepa(params, X, lat: int | None = None):
    """Run the trained ID-JEPA encoder on standardized context features."""
    if lat is None:
        lat = int(params["params"]["Dense_2"]["kernel"].shape[-1])
    z = Enc(lat=lat).apply(params, jnp.asarray(X))
    return np.asarray(unit_normalize(z))
