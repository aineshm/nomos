"""Runtime safety layer: a reciprocal velocity-obstacle (ORCA-style) / CBF safety
filter that wraps the RL policy's actions to drive collisions toward zero.

Why not brake-only: a greedy "slow down for threats" filter causes sudden-stop
cascades that CREATE crashes in dense traffic (we measured it net-negative). The
principled fix works in VELOCITY space: find the velocity closest to what the
policy wanted that keeps every neighbor outside its velocity obstacle, and apply
only HALF the correction (reciprocity) since the other car corrects too — so cars
steer smoothly around each other instead of all stopping.

Pipeline per car:
  policy action -> desired next velocity v_des
  -> for each near neighbor, push v_des out of its velocity-obstacle cutoff (×0.5)
  -> clamp to speed limit, plus a hard stopping-distance cap and pedestrian stop
  -> convert the safe velocity back to a (accel, steer) bicycle action.

Vectorized (JAX) over the spatial-hash candidates, so it scales to thousands.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..env import kinematic as K

TAU = 2.5          # collision look-ahead horizon (s)
MARGIN = 1.2       # extra clearance beyond collision radius (m)


def safe_action(env: K.Env, st: K.State, action: jnp.ndarray) -> jnp.ndarray:
    N = env.n_agents
    fdir = jnp.stack([jnp.cos(st.heading), jnp.sin(st.heading)], -1)   # (N,2)
    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])

    # desired next velocity implied by the policy action
    accel = jnp.clip(action[:, 0], -1, 1) * env.accel_max
    delta = jnp.clip(action[:, 1], -1, 1) * env.steer_max
    new_speed = jnp.clip(st.speed + accel * env.dt, 0.0, vmax)
    new_head = K._wrap(st.heading + (new_speed / env.wheelbase)
                       * jnp.tan(delta) * env.dt)
    v_des = new_speed[:, None] * jnp.stack([jnp.cos(new_head), jnp.sin(new_head)], -1)

    vel = st.speed[:, None] * fdir                       # current velocities (N,2)
    cand = K._candidates(env, st.pos)
    valid = (cand >= 0) & (cand != jnp.arange(N)[:, None])
    x = st.pos[cand] - st.pos[:, None, :]                # (N,C,2)  i -> j
    xn = jnp.linalg.norm(x, axis=-1) + 1e-6
    vj = vel[cand]                                       # neighbor velocities
    R = env.collision_radius + MARGIN

    # --- reciprocal velocity-obstacle correction (ORCA cutoff-circle form) ---
    v_rel = v_des[:, None, :] - vj                       # (N,C,2)
    w = v_rel - x / TAU                                  # offset from VO cutoff center
    wlen = jnp.linalg.norm(w, axis=-1) + 1e-6
    approaching = jnp.sum(v_rel * x, -1) > 0.0
    inside = (R / TAU - wlen) > 0.0
    active = valid & (xn < env.lead_cone) & approaching & inside
    # u = minimal change to push v_rel onto the cutoff-circle boundary
    u = ((R / TAU - wlen) / wlen)[..., None] * w         # (N,C,2)
    u = jnp.where(active[..., None], u, 0.0)
    correction = 0.5 * jnp.sum(u, axis=1)                # reciprocity: take half
    cn = jnp.linalg.norm(correction, axis=-1, keepdims=True)
    correction = correction * jnp.minimum(cn, vmax[:, None]) / (cn + 1e-6)
    v_safe = v_des + correction

    # --- hard caps: stopping distance to nearest in-path car, and pedestrians ---
    fwd = jnp.sum((st.pos[cand] - st.pos[:, None, :]) * fdir[:, None, :], -1)
    lat = jnp.abs((st.pos[cand] - st.pos[:, None, :])[..., 0] * fdir[:, None, 1]
                  - (st.pos[cand] - st.pos[:, None, :])[..., 1] * fdir[:, None, 0])
    in_path = valid & (fwd > 0) & (lat < env.lane_width * 0.8) & (fwd < env.lead_cone)
    gap = jnp.min(jnp.where(in_path, fwd, env.lead_cone), -1) - env.collision_radius
    v_stop = jnp.sqrt(2.0 * env.accel_max * jnp.clip(gap - MARGIN, 0.0, 1e6))

    prel = st.ped_pos[None, :, :] - st.pos[:, None, :]
    pf = jnp.sum(prel * fdir[:, None, :], -1)
    pl = jnp.abs(prel[..., 0] * fdir[:, None, 1] - prel[..., 1] * fdir[:, None, 0])
    p_in = (pf > 0) & (pl < env.lane_width) & (pf < env.lead_cone)
    gap_ped = jnp.min(jnp.where(p_in, pf, env.lead_cone), -1) - env.ped_radius
    v_ped = jnp.sqrt(2.0 * env.accel_max * jnp.clip(gap_ped - MARGIN, 0.0, 1e6))

    speed_cap = jnp.minimum(jnp.minimum(v_stop, v_ped), vmax)
    sp = jnp.linalg.norm(v_safe, axis=-1)
    des_speed = jnp.minimum(sp, speed_cap)
    des_head = jnp.where(sp > 1e-3, jnp.arctan2(v_safe[:, 1], v_safe[:, 0]), new_head)

    # --- convert safe velocity back to a bicycle (accel, steer) action ---
    out_accel = jnp.clip((des_speed - st.speed) / env.dt, -env.accel_max, env.accel_max)
    head_err = K._wrap(des_head - st.heading)
    turn_cap = (st.speed / env.wheelbase) * jnp.tan(env.steer_max) * env.dt + 1e-3
    out_steer = jnp.clip(head_err / turn_cap, -1.0, 1.0)
    return jnp.stack([out_accel / env.accel_max, out_steer, action[:, 2]], -1)
