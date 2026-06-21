# Component specs — schema legend

Each `*.json` in this directory is one component of the Nomos stack, captured
as structured context so the design can be reasoned over as a graph (see
`../graph.json`). All findings are web-researched and source-cited (June 2026).

## Architecture in one line

**Hierarchical control.** TRAIN the multi-agent coordination policy on a fast
**kinematic env** (Modal GPU) → EXECUTE/DEMO it on top of a **frozen low-level
locomotion controller** (WheeledLab-derived) running rigid-body physics in **Isaac**.
The low-level controller is trained once and is never part of the coordination
training loop — so there is no body-transfer problem.

## Field schema

| field | meaning |
|---|---|
| `id` | stable slug; node id in `graph.json` and in `upstream`/`downstream` refs |
| `name` | human name |
| `layer` | one of: `data-map`, `simulation`, `rl`, `compute`, `viz` |
| `kind` | `oss-library` \| `oss-framework` \| `managed-service` \| `dataset` \| `public-api` \| `glue-code` |
| `role` | what it does in THIS project (one sentence) |
| `status` | `confirmed-fit` \| `recommended` \| `gap` (see below) |
| `decision` | the concrete recommendation / what to do |
| `license` | license (code and/or data) |
| `inputs` / `outputs` | data in / data out |
| `upstream` / `downstream` | component ids it consumes from / feeds into (the graph edges) |
| `alternatives` | other tools considered + why |
| `risks` | concrete failure modes / time-sinks |
| `effort` | rough hackathon effort |
| `sources` | source URLs backing the claims |

## `status` legend

| value | meaning |
|---|---|
| `confirmed-fit` | right tool — keep |
| `recommended` | add / make explicit (the doc under-specified it) |
| `gap` | the design requires it but it must be built |

## Components (12)

**data-map:** `osmnx`, `proximity-osm`, `traffic-data`
**simulation:** `kinematic-env` (training), `lowlevel-controller` (frozen), `wheeledlab` (asset), `isaac-lab` (demo), `road-mesh-builder` (demo geometry)
**rl:** `marl-framework`, `reward-system`
**compute:** `modal`
**viz:** `viz-demo`

## Critical path (the spine)
`osmnx` → `kinematic-env` → `marl-framework` (+`reward-system`) → `modal`
→ trained policy → `lowlevel-controller` → `isaac-lab` → `viz-demo`

## Two real training runs
1. **Low-level controller** — single-agent, Isaac + WheeledLab, trained once then frozen.
2. **Coordination policy** — multi-agent, kinematic env on Modal — the main event, with a measurable learning curve.
