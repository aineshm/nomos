"""Smoke test: env v2 reset/step under JIT, and vmapped over parallel worlds."""
import jax

from smoothride.data.map_loader import load_road_network
from smoothride.env import kinematic as K
from smoothride.env.routing import build_route_pool

net = load_road_network()
x0, y0, x1, y1 = net.bounds()
pool = build_route_pool(net, n_routes=512)
env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=24, n_peds=12, max_steps=300)
print(f"obs_dim={env.obs_dim} act_dim={env.act_dim} "
      f"agents={env.n_agents} peds={env.n_peds}")

key = jax.random.PRNGKey(0)
st, obs = K.reset(env, key)
assert set(obs) == {"ego", "cars", "cars_mask", "peds", "peds_mask"}, obs.keys()
assert obs["ego"].shape == (env.n_agents, env.obs_dim), obs["ego"].shape

step = jax.jit(lambda s, a, k: K.step(env, s, a, k))
for i in range(env.max_steps):
    key, ka, ks = jax.random.split(key, 3)
    act = jax.random.uniform(ka, (env.n_agents, env.act_dim), minval=-1, maxval=1)
    st, obs, r, done, info = step(st, act, ks)
print(f"[single] crashes/car={float(info['crashes_per_car']):.2f} "
      f"goals={int(info['total_goals'])} ped_hits={int(info['ped_hits'])}")

B = 32
vreset = jax.jit(jax.vmap(lambda k: K.reset(env, k)))
vstep = jax.jit(jax.vmap(lambda s, a, k: K.step(env, s, a, k)))
bst, bobs = vreset(jax.random.split(jax.random.PRNGKey(1), B))
assert bobs["ego"].shape == (B, env.n_agents, env.obs_dim), bobs["ego"].shape
acts = jax.vmap(lambda k: jax.random.uniform(
    k, (env.n_agents, env.act_dim), minval=-1, maxval=1))(
    jax.random.split(jax.random.PRNGKey(2), B))
bst, bobs, br, bdone, binfo = vstep(
    bst, acts, jax.random.split(jax.random.PRNGKey(3), B))
print(f"[vmap x{B}] obs(ego)={tuple(bobs['ego'].shape)} reward={tuple(br.shape)}")
print("OK")
