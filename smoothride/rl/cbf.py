"""Higher-order Control Barrier Function (HOCBF) safety filter for the kinematic
bicycle — the principled runtime backstop the research recommends.

Barrier per neighbor: h = ||p_i - p_j||^2 - d_safe^2  (>=0 is safe). h has
relative degree 2 (accel/steer enter only through the 2nd derivative), so we use
the HOCBF condition  ḧ + α1·ḣ + α0·h >= 0, which is LINEAR in the controls
u = [a, tan δ]:

    A · u >= b(state),   A = [2 r·e_θ,  2 (v²/L) r·e_θ⊥]

We then solve the minimal-deviation QP min||u - u_des||² s.t. the single most
critical constraint (closed-form half-space projection), clip to actuator limits.
Crucially the steering coefficient is ~v²·(lateral offset), so the filter STEERS
to avoid when there's speed and BRAKES when slow — using both controls, unlike a
brake-only shield. Vectorized over the spatial-hash candidates.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ..env import kinematic as K

D_SAFE = 4.0     # m, barrier safe distance
ALPHA0 = 1.5     # class-K gains for the HOCBF
ALPHA1 = 3.0


def cbf_action(env: K.Env, st: K.State, action: jnp.ndarray) -> jnp.ndarray:
    N = env.n_agents
    e = jnp.stack([jnp.cos(st.heading), jnp.sin(st.heading)], -1)      # e_θ (N,2)
    eperp = jnp.stack([-jnp.sin(st.heading), jnp.cos(st.heading)], -1)  # e_θ⊥
    vmax = jnp.minimum(env.v_max, env.routes_speed[st.route_idx, st.wp_ptr])

    a_des = jnp.clip(action[:, 0], -1, 1) * env.accel_max
    w_des = jnp.tan(jnp.clip(action[:, 1], -1, 1) * env.steer_max)     # tan δ
    v = st.speed

    cand = K._candidates(env, st.pos)
    valid = (cand >= 0) & (cand != jnp.arange(N)[:, None])
    r = st.pos[:, None, :] - st.pos[cand]              # (N,C,2) j -> i
    d2 = jnp.sum(r * r, -1)
    vel = v[:, None] * e
    dv = vel[:, None, :] - vel[cand]                   # Δv = ṗ_i - ṗ_j  (N,C,2)

    h = d2 - D_SAFE ** 2
    hdot = 2.0 * jnp.sum(r * dv, -1)
    # constraint A·[a, w] >= b
    A_a = 2.0 * jnp.sum(r * e[:, None, :], -1)                          # (N,C)
    A_w = 2.0 * (v[:, None] ** 2 / env.wheelbase) * jnp.sum(r * eperp[:, None, :], -1)
    b = -(2.0 * jnp.sum(dv * dv, -1) + ALPHA1 * hdot + ALPHA0 * h)      # (N,C)

    # slack of the desired control for each neighbor; engage only real threats
    slack = A_a * a_des[:, None] + A_w * w_des[:, None] - b
    engaged = valid & (d2 < env.lead_cone ** 2) & (hdot < 0.0) & (slack < 0.0)
    slack = jnp.where(engaged, slack, jnp.inf)
    j = jnp.argmin(slack, axis=1)                      # most-critical neighbor
    sl = jnp.take_along_axis(slack, j[:, None], 1)[:, 0]
    Aa = jnp.take_along_axis(A_a, j[:, None], 1)[:, 0]
    Aw = jnp.take_along_axis(A_w, j[:, None], 1)[:, 0]
    norm2 = Aa * Aa + Aw * Aw + 1e-6

    # half-space projection: u = u_des + max(0, -slack)/||A||^2 * A
    push = jnp.clip(-sl, 0.0, 1e9) / norm2
    a_safe = a_des + push * Aa
    w_safe = w_des + push * Aw

    a_safe = jnp.clip(a_safe, -env.accel_max, env.accel_max)
    w_lim = jnp.tan(env.steer_max)
    delta = jnp.arctan(jnp.clip(w_safe, -w_lim, w_lim))
    return jnp.stack([a_safe / env.accel_max, delta / env.steer_max, action[:, 2]], -1)
