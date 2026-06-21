"""MAPPO/IPPO trainer for the kinematic env (JAX).

Shared-parameter PPO with a centralized critic. One iteration:
  collect a full episode across B parallel worlds -> GAE -> several PPO epochs.
Each (world, agent) is an independent trajectory for the shared policy.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from ..env import kinematic as K
from .networks import ActorCritic, gaussian_entropy, gaussian_logp


@dataclass(frozen=True)
class PPOConfig:
    n_worlds: int = 32
    epochs: int = 4
    minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


def _global_feat(obs):
    """Pooled scene summary per world, broadcast to each agent. obs: (..., N, O)."""
    return jnp.broadcast_to(obs.mean(-2, keepdims=True), obs.shape)


def make_train_state(env: K.Env, cfg: PPOConfig, key) -> TrainState:
    net = ActorCritic(act_dim=env.act_dim)
    dummy = jnp.zeros((env.n_agents, env.obs_dim))
    params = net.init(key, dummy, _global_feat(dummy))
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm),
                     optax.adam(cfg.lr))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


# env is a pytree (route arrays as leaves; scalar params are static treedef
# fields). Only n_worlds must be static (it sets array shapes).
@functools.partial(jax.jit, static_argnums=(3,))
def collect(env: K.Env, ts: TrainState, key, n_worlds: int):
    """Roll out one full episode across n_worlds worlds. Leaves: (B, T, N, ...)."""

    def one_world_rollout(world_key):
        kr, ks = jax.random.split(world_key)
        st, obs = K.reset(env, kr)

        def step_fn(carry, k):
            st, obs = carry
            gf = _global_feat(obs)
            mean, log_std, value = ts.apply_fn(ts.params, obs, gf)
            ka, kn = jax.random.split(k)
            noise = jax.random.normal(ka, mean.shape)
            action = mean + jnp.exp(log_std) * noise
            logp = gaussian_logp(action, mean, log_std)
            nst, nobs, reward, done, info = K.step(env, st, action, kn)
            out = dict(obs=obs, gf=gf, action=action, logp=logp,
                       value=value, reward=reward,
                       cost=info["just_crashed"].astype(jnp.float32),
                       # raw State fields so the verifier can relabel cost off-device
                       # (rl/verifier.cost_signal). seg geometry is gathered on host.
                       pos=st.pos, heading=st.heading, speed=st.speed,
                       route_idx=st.route_idx, wp_ptr=st.wp_ptr,
                       spawn_grace=st.spawn_grace,
                       crashed=info["just_crashed"],
                       # pedestrian state — per-world (M,2)/(M,), NOT per-agent:
                       # after vmap+scan these become (B,T,M,2)/(B,T,M).
                       ped_pos=st.ped_pos, ped_crossing=st.ped_crossing)
            return (nst, nobs), out

        ks_steps = jax.random.split(ks, env.max_steps)
        (last_st, last_obs), traj = jax.lax.scan(step_fn, (st, obs), ks_steps)
        _, _, last_value = ts.apply_fn(ts.params, last_obs, _global_feat(last_obs))
        traj["last_value"] = last_value
        traj["final_crashes"] = last_st.crashes
        traj["final_goals"] = last_st.goals
        return traj

    world_keys = jax.random.split(key, n_worlds)
    batch = jax.vmap(one_world_rollout)(world_keys)  # leaves: (B, T, N, ...)
    return batch


def verifier_cost(env: K.Env, batch) -> jnp.ndarray:
    """Relabel a collected rollout with the verifier's per-step cost (handoff §8).

    The rollout runs on device; here (on host) we gather the road geometry for each
    logged (route_idx, wp_ptr) and run the same rule predicates the offline verifier
    grades with — so the signal the policy is trained against IS the verifier's. This
    is the reward-model pattern: score completed rollouts, feed the score to PPO.
    Returns (B, T, N) to drop straight into `batch["cost"]`.
    """
    import numpy as np

    from .verifier import step_cost

    rxy = np.asarray(env.routes_xy)              # (R, W, 2)
    rlanes = np.asarray(env.routes_lanes)        # (R, W)
    rspeed = np.asarray(env.routes_speed)        # (R, W)
    ri = np.asarray(batch["route_idx"])          # (B, T, N)
    wp = np.asarray(batch["wp_ptr"])
    B, T, N = ri.shape
    seg_start = rxy[ri, np.maximum(wp - 1, 0)]   # (B, T, N, 2)
    seg_end = rxy[ri, wp]
    lane_count = rlanes[ri, wp]
    speed_limit = rspeed[ri, wp]

    def r2(x):                                   # (B,T,...) -> (B*T,...)
        return x.reshape((B * T,) + x.shape[2:])

    # Ped arrays are per-world (shared across agents): reshape (B,T,M,...) -> (B*T,M,...).
    pp = r2(np.asarray(batch["ped_pos"]))      # (B*T, M, 2)
    pc = r2(np.asarray(batch["ped_crossing"])) # (B*T, M)
    cost = step_cost(
        r2(np.asarray(batch["pos"])), r2(seg_start), r2(seg_end), r2(lane_count),
        float(env.lane_width), r2(np.asarray(batch["heading"])),
        r2(np.asarray(batch["speed"])), r2(np.asarray(batch["spawn_grace"])),
        r2(np.asarray(batch["crashed"])), r2(speed_limit),
        ped_pos=pp, ped_crossing=pc,
        r_ped=float(env.ped_radius), r_yield=float(env.r_yield),
        cruise_cap=float(env.cruise_cap))
    return jnp.asarray(cost.reshape(B, T, N))


def compute_gae(reward, value, last_value, gamma, lam):
    """reward/value: (T, N). last_value: (N,). One episode ending at horizon."""
    def scan_fn(carry, x):
        gae, next_v = carry
        r, v = x
        delta = r + gamma * next_v - v
        gae = delta + gamma * lam * gae
        return (gae, v), gae

    (_, _), adv = jax.lax.scan(
        scan_fn, (jnp.zeros_like(last_value), last_value),
        (reward, value), reverse=True)
    returns = adv + value
    return adv, returns


@functools.partial(jax.jit, static_argnums=(1,))
def update(env: K.Env, cfg: PPOConfig, ts: TrainState, batch, lam=0.0):
    # GAE per (world, agent): vmap over worlds, then over agents.
    def world_gae(reward, value, last_value):
        # reward/value: (T, N) ; last_value: (N,)
        adv, ret = jax.vmap(
            lambda r, v, lv: compute_gae(r, v, lv, cfg.gamma, cfg.gae_lambda),
            in_axes=(1, 1, 0), out_axes=1)(reward, value, last_value)
        return adv, ret

    # PPO-Lagrangian: subtract an adaptive multiplier * per-step crash cost. lam
    # is updated by dual ascent in the train loop toward a crash-rate target.
    reward_eff = batch["reward"] - lam * batch["cost"]
    adv, ret = jax.vmap(world_gae)(reward_eff, batch["value"],
                                   batch["last_value"])  # (B, T, N)

    def flat(x):
        return x.reshape((-1,) + x.shape[3:])
    obs = flat(batch["obs"])
    gf = flat(batch["gf"])
    action = flat(batch["action"])
    old_logp = flat(batch["logp"])
    advantage = flat(adv)
    returns = flat(ret)
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    n = obs.shape[0]
    mb = n // cfg.minibatches

    def ppo_loss(params, ob, g, ac, olp, advv, rets):
        mean, log_std, value = ts.apply_fn(params, ob, g)
        logp = gaussian_logp(ac, mean, log_std)
        ratio = jnp.exp(logp - olp)
        unclipped = ratio * advv
        clipped = jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * advv
        pg = -jnp.minimum(unclipped, clipped).mean()
        vloss = 0.5 * ((value - rets) ** 2).mean()
        ent = gaussian_entropy(log_std).mean()
        loss = pg + cfg.vf_coef * vloss - cfg.ent_coef * ent
        return loss, (pg, vloss, ent)

    def epoch(carry, perm_key):
        ts = carry
        perm = jax.random.permutation(perm_key, n)
        def mb_step(ts, i):
            idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
            grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
            (loss, aux), grads = grad_fn(
                ts.params, obs[idx], gf[idx], action[idx],
                old_logp[idx], advantage[idx], returns[idx])
            ts = ts.apply_gradients(grads=grads)
            return ts, loss
        ts, losses = jax.lax.scan(mb_step, ts, jnp.arange(cfg.minibatches))
        return ts, losses.mean()

    keys = jax.random.split(jax.random.PRNGKey(0), cfg.epochs)
    ts, losses = jax.lax.scan(epoch, ts, keys)

    metrics = {
        "loss": losses.mean(),
        "return": returns.mean(),
        "adv": advantage.mean(),
        "ep_reward": batch["reward"].sum(1).mean(),     # per-world-agent episode sum
        "crashes_per_car": batch["final_crashes"].mean(),
        "goals_per_agent": batch["final_goals"].mean(),
    }
    return ts, metrics
