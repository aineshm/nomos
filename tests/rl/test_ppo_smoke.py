"""End-to-end PPO training smoke test (Task 8).

Exercises the full collect → verifier_cost → update loop with the
Deep Sets dict-obs policy.  The test asserts only that the returned
metrics are finite — i.e. no NaN/Inf survives the minibatch path.
"""
import jax
import jax.numpy as jnp

from smoothride.rl import ppo
from tests.env.test_kinematic_peds import _env


def test_one_ppo_iteration_runs_end_to_end() -> None:
    """One full PPO iteration must complete and emit finite metrics.

    Config: n_worlds=2 keeps the JIT-compiled scan small.
    epochs=1, minibatches=2 are enough to exercise the minibatch
    loop without a long compile.  Total flat obs size =
    n_worlds * max_steps * n_agents = 2 * 300 * 4 = 2 400, which
    is divisible by minibatches=2 (mb_size=1200).
    """
    env = _env(cruise_cap=4.0)
    cfg = ppo.PPOConfig(n_worlds=2, epochs=1, minibatches=2)

    ts = ppo.make_train_state(env, cfg, jax.random.PRNGKey(0))
    batch = ppo.collect(env, ts, jax.random.PRNGKey(1), cfg.n_worlds)
    batch = {**batch, "cost": ppo.verifier_cost(env, batch)}
    ts2, metrics = ppo.update(env, cfg, ts, batch, lam=1.0)

    assert jnp.isfinite(metrics["loss"]), (
        f"loss is not finite: {metrics['loss']}"
    )
    assert jnp.isfinite(metrics["ep_reward"]), (
        f"ep_reward is not finite: {metrics['ep_reward']}"
    )
    # Sanity: update must return a valid TrainState (not the same object).
    assert ts2 is not ts, "update must return a new TrainState"
