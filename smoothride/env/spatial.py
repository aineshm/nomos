"""Spatial-hash neighbor search (JAX, fixed-shape) so the env scales to thousands
of cars at O(N*C) instead of O(N^2).

Agents are binned into a uniform grid; each agent's candidate neighbors are the
agents in its own cell + the 8 adjacent cells (3x3 block). cell_size must be >=
the largest interaction radius so any true neighbor lands in the 3x3 block.

`grid_candidates` returns, per agent, a fixed list of candidate agent indices
(-1 = empty slot). Consumers gather those and reduce (min distance, K-nearest,
lead gap, junction conflicts) over the small candidate set instead of all N.
"""
from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp


def grid_dims(world_min, world_max, cell_size):
    """Static grid dimensions (python ints) for a world extent."""
    ncx = max(1, int(math.ceil((float(world_max[0]) - float(world_min[0])) / cell_size)))
    ncy = max(1, int(math.ceil((float(world_max[1]) - float(world_min[1])) / cell_size)))
    return ncx, ncy


_OFFSETS = jnp.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                     dtype=jnp.int32)  # (9, 2)


@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6))
def grid_candidates(pos, world_min, cell_size, ncx, ncy, cap, _C):
    """pos: (N,2). Returns cand: (N, 9*cap) int32 agent indices (-1 = empty)."""
    N = pos.shape[0]
    cix = jnp.clip(((pos[:, 0] - world_min[0]) / cell_size).astype(jnp.int32), 0, ncx - 1)
    ciy = jnp.clip(((pos[:, 1] - world_min[1]) / cell_size).astype(jnp.int32), 0, ncy - 1)
    cell_id = ciy * ncx + cix                      # (N,)
    ncells = ncx * ncy

    # slot of each agent within its cell, via sort + running-max segment starts
    order = jnp.argsort(cell_id)
    sorted_cell = cell_id[order]
    ranks = jnp.arange(N)
    is_new = jnp.concatenate([jnp.array([True]), sorted_cell[1:] != sorted_cell[:-1]])
    seg_start = jax.lax.associative_scan(jnp.maximum, jnp.where(is_new, ranks, 0))
    slot_sorted = (ranks - seg_start).astype(jnp.int32)
    slot = jnp.zeros(N, jnp.int32).at[order].set(slot_sorted)

    # scatter into table[cell, slot]; overflow (slot>=cap) dumped to throwaway row
    valid = slot < cap
    wcell = jnp.where(valid, cell_id, ncells)
    wslot = jnp.where(valid, slot, 0)
    table = jnp.full((ncells + 1, cap), -1, jnp.int32).at[wcell, wslot].set(jnp.arange(N))

    # gather 3x3 neighbor cells (clamped at borders)
    nbx = jnp.clip(cix[:, None] + _OFFSETS[None, :, 0], 0, ncx - 1)   # (N,9)
    nby = jnp.clip(ciy[:, None] + _OFFSETS[None, :, 1], 0, ncy - 1)
    ncell = nby * ncx + nbx                                           # (N,9)
    cand = table[ncell].reshape(N, 9 * cap)                          # (N, C)
    return cand


def candidate_count(cap: int) -> int:
    return 9 * cap


if __name__ == "__main__":
    # correctness: candidate-based nearest distance vs brute force
    import numpy as np
    key = jax.random.PRNGKey(0)
    N = 2000
    pos = jax.random.uniform(key, (N, 2), minval=0.0, maxval=1000.0)
    cell, cap = 35.0, 16
    ncx, ncy = grid_dims((0, 0), (1000, 1000), cell)
    C = candidate_count(cap)
    cand = grid_candidates(pos, jnp.array([0.0, 0.0]), cell, ncx, ncy, cap, C)

    # candidate min-distance (within cell radius)
    cp = pos[cand]                                   # (N, C, 2)
    selfmask = (cand < 0) | (cand == jnp.arange(N)[:, None])
    d = jnp.linalg.norm(cp - pos[:, None, :], axis=-1)
    d = jnp.where(selfmask, 1e9, d)
    grid_min = d.min(1)

    # brute force, but only count truth within the cell radius (fair comparison)
    dd = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1) + jnp.eye(N) * 1e9
    bf_min = dd.min(1)

    # within one cell_size the grid must match brute force exactly
    near = bf_min <= cell
    agree = jnp.allclose(grid_min[near], bf_min[near], atol=1e-4)
    print(f"N={N} cells={ncx}x{ncy} cand/agent={C}")
    print(f"agents with a neighbor <= {cell}m: {int(near.sum())}")
    print(f"grid matches brute force within cell radius: {bool(agree)}")
