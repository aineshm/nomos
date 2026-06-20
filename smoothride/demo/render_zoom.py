"""Render realistic-density SF traffic two ways:
  * CITY view  — every car as a speed-colored dot (red=stopped .. green=fast),
                 showing density + emergent congestion.
  * ZOOM view  — cars as oriented rectangles in a ~180 m window over the busiest
                 spot, so individual lanes and turn radius are visible.

Usage:
  python -m smoothride.demo.render_zoom --agents 6000 --peds 1200 \
      --ckpt runs/trained_city.msgpack
"""
from __future__ import annotations

import argparse
import os

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from ..data.map_loader import load_sf_graph, to_road_network  # noqa: E402
from ..env import kinematic as K  # noqa: E402
from ..env.routing import build_route_pool  # noqa: E402
from .render import OUT, load_params, rollout  # noqa: E402

HUGE_BBOX = (-122.4300, 37.7250, -122.3800, 37.8050)
CAR_L, CAR_W = 4.6, 2.0
SPEED_CMAP = plt.get_cmap("RdYlGn")


def _bg(ax, net, facecolor="#0e1116"):
    ax.set_facecolor(facecolor)
    ax.add_collection(LineCollection(net.node_xy[net.edges],
                                     colors="#39404d", linewidths=0.6))
    ax.set_aspect("equal")
    ax.axis("off")


def _densest_window(pos, win):
    """Center of the busiest area over the whole rollout."""
    flat = pos.reshape(-1, 2)
    H, xe, ye = np.histogram2d(flat[:, 0], flat[:, 1], bins=60)
    i, j = np.unravel_index(np.argmax(H), H.shape)
    cx = 0.5 * (xe[i] + xe[i + 1])
    cy = 0.5 * (ye[j] + ye[j + 1])
    return cx, cy


def render_city(net, tr, vmax, out_prefix, title, stride=2, fps=15):
    pos, speed, crashed = tr["pos"], tr["speed"], tr["crashed"]
    T, N, _ = pos.shape
    x0, y0, x1, y1 = net.bounds()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    fig.patch.set_facecolor("#0e1116")
    _bg(ax, net)
    ax.set_xlim(x0 - 40, x1 + 40)
    ax.set_ylim(y0 - 40, y1 + 40)
    ax.set_title(title, color="white", fontsize=12, pad=8)
    hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top",
                  color="white", fontsize=9, family="monospace")
    norm = Normalize(0, vmax)
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=5,
                    c=speed[0], cmap=SPEED_CMAP, norm=norm, zorder=3)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SPEED_CMAP), ax=ax,
                      fraction=0.03, pad=0.01)
    cb.set_label("speed m/s (red=stopped, green=fast)", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")

    def update(t):
        c = np.where(crashed[t], 0.0, speed[t])  # crashed shown stopped (red)
        sc.set_offsets(pos[t])
        sc.set_array(c)
        moving = (speed[t] > 1.0) & ~crashed[t]
        hud.set_text(f"t={t:3d}/{T}   cars={N}   moving={int(moving.sum()):4d}   "
                     f"stopped/jam={int((~moving).sum()):4d}")
        return sc, hud

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=range(0, T, stride))
    anim.save(out_prefix + "_city.gif", writer=animation.PillowWriter(fps=fps))
    for tag, t in [("start", 0), ("mid", T // 2), ("end", T - 1)]:
        update(t); fig.savefig(f"{out_prefix}_city_{tag}.png",
                               facecolor=fig.get_facecolor())
    plt.close(fig)


def _car_quads(p, h):
    """(M,2) pos, (M,) heading -> (M,4,2) rectangle corners."""
    cx, sx = np.cos(h), np.sin(h)
    fwd = np.stack([cx, sx], -1) * (CAR_L / 2)
    sde = np.stack([-sx, cx], -1) * (CAR_W / 2)
    return np.stack([p + fwd + sde, p + fwd - sde,
                     p - fwd - sde, p - fwd + sde], axis=1)


def render_zoom(net, tr, vmax, out_prefix, title, win=90.0, stride=1, fps=18,
                trail=16, center=None):
    """Zoom view with motion TRAILS so navigation (turns, lane changes,
    intersection transitions) is visible as the curved path each car traces."""
    pos, heading, speed = tr["pos"], tr["heading"], tr["speed"]
    crashed, ped = tr["crashed"], tr["ped"]
    T, N, _ = pos.shape
    cx, cy = center if center is not None else _densest_window(pos, win)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    fig.patch.set_facecolor("#0e1116")
    _bg(ax, net)
    ax.set_xlim(cx - win, cx + win)
    ax.set_ylim(cy - win, cy + win)
    ax.set_title(title, color="white", fontsize=12, pad=8)
    hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top",
                  color="white", fontsize=9, family="monospace")
    norm = Normalize(0, vmax)
    sm = ScalarMappable(norm=norm, cmap=SPEED_CMAP)
    trails = LineCollection([], zorder=2, linewidths=1.6)   # navigation paths
    ax.add_collection(trails)
    cars = PolyCollection([], zorder=3, edgecolors="white", linewidths=0.3)
    ax.add_collection(cars)
    peds = ax.scatter([], [], s=22, c="#f59e0b", marker="D", zorder=4)
    MAXJUMP = max(vmax * 0.2 * 2.0, 8.0)  # break trail across respawn teleports

    def update(t):
        inwin = (np.abs(pos[t, :, 0] - cx) < win) & (np.abs(pos[t, :, 1] - cy) < win)
        p, h = pos[t, inwin], heading[t, inwin]
        cars.set_verts(_car_quads(p, h))
        cars.set_facecolor(sm.to_rgba(np.where(crashed[t, inwin], 0.0, speed[t, inwin])))

        # build fading trails for in-window cars over the last `trail` steps
        t0 = max(0, t - trail)
        ph = pos[t0:t + 1, inwin]                 # (k, m, 2)
        if ph.shape[0] >= 2:
            a, b = ph[:-1], ph[1:]                # (k-1, m, 2)
            seg = np.stack([a, b], axis=2).reshape(-1, 2, 2)
            d = np.linalg.norm(b - a, axis=-1).reshape(-1)
            k = a.shape[0]
            rec = np.repeat(np.linspace(0.12, 0.85, k), a.shape[1])   # older=fainter
            spd = (0.5 * (a + b))                 # midpoint speed proxy via position delta
            sp = (np.linalg.norm(b - a, axis=-1) / (vmax * 0.2)).reshape(-1)
            keep = d < MAXJUMP
            rgba = sm.to_rgba(np.clip(sp[keep] * vmax, 0, vmax))
            rgba[:, 3] = rec[keep]
            trails.set_segments(seg[keep])
            trails.set_color(rgba)
        else:
            trails.set_segments([])

        pin = (np.abs(ped[t, :, 0] - cx) < win) & (np.abs(ped[t, :, 1] - cy) < win)
        peds.set_offsets(ped[t, pin] if pin.any() else np.empty((0, 2)))
        hud.set_text(f"t={t:3d}/{T}   cars in view={int(inwin.sum()):2d}   "
                     f"window~{int(2*win)}m   trails=navigation paths")
        return trails, cars, peds, hud

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=range(0, T, stride))
    anim.save(out_prefix + "_zoom.gif", writer=animation.PillowWriter(fps=fps))
    for tag, t in [("mid", T // 2), ("end", T - 1)]:
        update(t); fig.savefig(f"{out_prefix}_zoom_{tag}.png",
                               facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=6000)
    ap.add_argument("--peds", type=int, default=1200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--vmax", type=float, default=30.0)
    ap.add_argument("--ckpt", default=os.path.join(OUT, "trained.msgpack"))
    ap.add_argument("--name", default="city")
    ap.add_argument("--win", type=float, default=90.0)
    ap.add_argument("--downtown", action="store_true",
                    help="clean navigation zoom on the downtown map + downtown policy")
    ap.add_argument("--no-city", action="store_true", dest="no_city")
    ap.add_argument("--safe", action="store_true", help="apply runtime safety filter")
    ap.add_argument("--filt", default="cbf", choices=["vo", "cbf"])
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    if args.downtown:
        from ..data.map_loader import load_road_network
        net = load_road_network()
        vmax = min(args.vmax, 16.0)
    else:
        net = to_road_network(load_sf_graph(bbox=HUGE_BBOX,
                                            cache_name="sf_huge_drive.graphml"))
        vmax = args.vmax
    x0, y0, x1, y1 = net.bounds()
    pool = build_route_pool(net, n_routes=4096 if not args.downtown else 1024,
                            max_length_m=2500.0 if not args.downtown else 700.0)
    env = K.make_env(pool, (x0, y0), (x1, y1), n_agents=args.agents,
                     n_peds=args.peds, max_steps=args.steps, v_max=vmax)
    params = load_params(env, args.ckpt)
    tr = rollout(env, params, jax.random.PRNGKey(args.seed), sample=True,
                 safe=args.safe, filt=args.filt)
    out = os.path.join(OUT, "artifacts", args.name)
    if not args.no_city:
        render_city(net, tr, vmax, out, f"SF — {args.agents} cars")
    render_zoom(net, tr, vmax, out,
                "ZOOM — navigation: turns, lane changes, intersections",
                win=args.win)
    moving = (tr["speed"][-1] > 1.0)
    print(f"cars={args.agents}  moving_end={int(moving.sum())}  "
          f"crashes/car={float(tr['crashed'].sum(0).mean()):.2f}  "
          f"trips={int(tr['goals'][-1].sum())}")
    print(f"saved: {out}_zoom.gif (+ stills)")


if __name__ == "__main__":
    main()
