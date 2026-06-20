"""A small DDPM that learns to generate car trajectory snippets through SF
junctions, then GUIDES denoising adversarially to synthesize hard near-miss
conflicts — the Safe-Sim / MotionDiffuser idea in miniature, for an offline
scenario bank that the curriculum trains against.

  collect ego-relative H-step trajectories from a policy rollout
  -> train a denoiser (predict noise; standard DDPM)
  -> sample plausible trajectories
  -> GUIDED sample: add a collision-cost gradient that pulls a generated
     "challenger" trajectory toward the ego path -> a plausible near-miss.

Self-contained and CPU-sized (small MLP denoiser, ~50 diffusion steps).
"""
from __future__ import annotations

import os

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "runs"))
H = 16          # trajectory horizon (steps)
T = 50          # diffusion steps


# ----------------------------- data -----------------------------
def collect_trajectories(n_snippets=4000, seed=0):
    """Ego-relative H-step displacement trajectories from a policy rollout."""
    import jax as _jax
    from ..data.map_loader import load_road_network
    from ..env import kinematic as K
    from ..env.routing import build_route_pool
    from ..demo.render import load_params, rollout
    net = load_road_network()
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=1024)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=120, n_peds=10,
                     max_steps=300, v_max=16.0)
    ck = os.path.join(OUT, "trained.msgpack")
    params = load_params(env, ck)
    tr = rollout(env, params, _jax.random.PRNGKey(seed), sample=True)
    pos, head = tr["pos"], tr["heading"]          # (Tt, N, 2), (Tt, N)
    Tt, N, _ = pos.shape
    snips = []
    rng = np.random.default_rng(seed)
    for _ in range(n_snippets * 3):
        if len(snips) >= n_snippets:
            break
        t = rng.integers(0, Tt - H - 1); i = rng.integers(0, N)
        seg = pos[t:t + H + 1, i]                  # (H+1, 2)
        steps = np.linalg.norm(np.diff(seg, axis=0), axis=-1)
        if steps.max() > 6.0 or steps.sum() < 3.0:  # skip respawns / parked
            continue
        # ego frame: translate to start, rotate by -heading[t]
        d = seg[1:] - seg[0]
        c, s = np.cos(-head[t, i]), np.sin(-head[t, i])
        rot = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], -1)
        snips.append(rot.astype(np.float32))
    X = np.stack(snips)                            # (M, H, 2)
    scale = float(np.abs(X).max())
    return X / scale, scale


# ----------------------------- model -----------------------------
def time_embed(t, dim=64):
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    a = t[:, None].astype(jnp.float32) * freqs[None, :]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


class Denoiser(nn.Module):
    dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, x, t):
        h = jnp.concatenate([x, time_embed(t)], -1)
        for _ in range(3):
            h = nn.silu(nn.Dense(self.hidden)(h))
        return nn.Dense(self.dim)(h)


def make_schedule():
    betas = jnp.linspace(1e-4, 0.02, T)
    alphas = 1.0 - betas
    abar = jnp.cumprod(alphas)
    return betas, alphas, abar


# ----------------------------- train -----------------------------
def train(X, iters=3000, bs=256, lr=2e-4, seed=0):
    dim = H * 2
    betas, alphas, abar = make_schedule()
    net = Denoiser(dim=dim)
    key = jax.random.PRNGKey(seed)
    params = net.init(key, jnp.zeros((1, dim)), jnp.zeros((1,), jnp.int32))
    tx = optax.adam(lr); opt = tx.init(params)
    Xf = jnp.asarray(X.reshape(X.shape[0], -1))

    def loss_fn(params, xb, t, noise):
        xt = jnp.sqrt(abar[t])[:, None] * xb + jnp.sqrt(1 - abar[t])[:, None] * noise
        pred = net.apply(params, xt, t)
        return jnp.mean((pred - noise) ** 2)

    @jax.jit
    def step(params, opt, key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (bs,), 0, Xf.shape[0])
        xb = Xf[idx]
        t = jax.random.randint(k2, (bs,), 0, T)
        noise = jax.random.normal(k3, xb.shape)
        l, g = jax.value_and_grad(loss_fn)(params, xb, t, noise)
        upd, opt = tx.update(g, opt)
        return optax.apply_updates(params, upd), opt, l

    for i in range(iters):
        key, k = jax.random.split(key)
        params, opt, l = step(params, opt, k)
        if i % 500 == 0:
            print(f"  diff train it {i:4d}  loss {float(l):.4f}")
    return net, params, (betas, alphas, abar)


# ----------------------------- sample -----------------------------
def sample(net, params, sched, n, key, guide=None, guide_scale=0.0):
    """Reverse diffusion. guide(x0_hat)->scalar cost; its grad steers samples
    (classifier-style guidance) — used to make adversarial near-misses."""
    betas, alphas, abar = sched
    dim = H * 2
    x = jax.random.normal(key, (n, dim))
    for t in reversed(range(T)):
        tt = jnp.full((n,), t)
        eps = net.apply(params, x, tt)
        a, ab, b = alphas[t], abar[t], betas[t]
        x0_hat = (x - jnp.sqrt(1 - ab) * eps) / jnp.sqrt(ab)
        if guide is not None and guide_scale > 0:
            g = jax.grad(lambda z: guide(z).sum())(x0_hat)
            x0_hat = x0_hat - guide_scale * g
        mean = jnp.sqrt(a) * (1 - abar[t - 1]) / (1 - ab) * x + \
            jnp.sqrt(abar[t - 1]) * b / (1 - ab) * x0_hat if t > 0 else x0_hat
        key, kn = jax.random.split(key)
        noise = jax.random.normal(kn, x.shape) if t > 0 else 0.0
        x = mean + (jnp.sqrt(b) * noise if t > 0 else 0.0)
    return x.reshape(n, H, 2)


def main():
    print("collecting trajectories from policy rollout...")
    X, scale = collect_trajectories(4000)
    print(f"  dataset: {X.shape}  scale={scale:.1f}m")
    print("training DDPM denoiser...")
    net, params, sched = train(X, iters=3000)

    key = jax.random.PRNGKey(1)
    gen = np.asarray(sample(net, params, sched, 64, key)) * scale

    # adversarial: pull the challenger's midpoint toward a fixed ego point ahead
    ego_pt = jnp.array([12.0 / scale, 0.0])  # ~12 m straight ahead (ego path)

    def conflict_cost(x0):                    # x0: (n, H*2) normalized
        traj = x0.reshape(x0.shape[0], H, 2)
        mid = traj[:, H // 2]
        return jnp.sum((mid - ego_pt) ** 2, -1)
    adv = np.asarray(sample(net, params, sched, 64, jax.random.PRNGKey(2),
                            guide=conflict_cost, guide_scale=0.5)) * scale

    _viz(np.asarray(X) * scale, gen, adv)
    np.savez(os.path.join(OUT, "diffusion_scenarios.npz"), real=X * scale,
             generated=gen, adversarial=adv, scale=scale)
    print(f"saved generated + adversarial scenarios -> {OUT}/diffusion_scenarios.npz")


def _viz(real, gen, adv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=110)
    for ax, data, title, col in [
        (axes[0], real, "real (policy rollouts)", "#3b82f6"),
        (axes[1], gen, "diffusion-generated", "#22c55e"),
        (axes[2], adv, "adversarial (guided near-miss)", "#ef4444")]:
        for tr in data[:60]:
            ax.plot(tr[:, 0], tr[:, 1], color=col, alpha=0.4, lw=1)
        if "adversarial" in title:
            ax.plot([0, 16], [0, 0], "--", color="white", lw=1.5, label="ego path")
            ax.scatter([12], [0], color="white", s=30, zorder=5)
            ax.legend(fontsize=8, labelcolor="white", facecolor="#0e1116")
        ax.set_title(title, color="white"); ax.set_aspect("equal")
        ax.set_facecolor("#0e1116"); ax.tick_params(colors="#888")
    fig.patch.set_facecolor("#0e1116")
    p = os.path.join(OUT, "artifacts", "diffusion_scenarios.png")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"saved viz -> {p}")


if __name__ == "__main__":
    main()
