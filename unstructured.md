Multi-agent, multi-reward reinforcement learning for a trafficless world.

https://uwrobotlearning.github.io/WheeledLab/

<!-- ════════════════════════════════════════════════════════════════════════
     📋 ANNOTATED — updated 2026-06-18 (hierarchical-control pivot)
     Structured component specs: ./components/*.json · dataflow: ./graph.json ·
     corrected stack + build order + honesty check: ./STACK.md

     🔑 THE ARCHITECTURE (one line):
     TRAIN the multi-agent coordination policy on a fast KINEMATIC env (Modal
     GPU). EXECUTE/DEMO it on top of a FROZEN low-level locomotion controller
     (WheeledLab-derived) running rigid-body physics in ISAAC. The low-level
     controller is trained once and is NEVER part of the coordination training
     loop — so there is no "train on a simple body then swap" transfer problem;
     we learn coordination ON TOP OF a fixed locomotion skill.

     Two real training runs:
       1. low-level controller — single-agent, Isaac+WheeledLab, trained once → frozen
       2. coordination policy   — multi-agent, kinematic env on Modal (the main event)
     ════════════════════════════════════════════════════════════════════════ -->

This is a reinforcement learning framework for training a robotic car in simulation to do various reward-based tasks like:
drift
go over an obstacle
stay in a lane
Train from completely ground up.

<!-- 🔎 WheeledLab — its role here (source-verified):
     • ~1/10-scale RC cars (MuSHR, HOUND). Backend: Isaac Lab on Isaac Sim
       (pins Isaac Sim v4.5.0 + IsaacLab v2.0.2). RL: RSL-RL / Stable-Baselines3.
     • Shipped tasks: drifting, elevation, visual nav (single-agent — verified
       in env source). License BSD-3. CoRL 2025 (arXiv 2502.07380).
     • IN OUR STACK it supplies the CAR ASSET + dynamics/domain-randomization
       recipe used to TRAIN the frozen low-level controller (single-agent is
       exactly right — that layer only needs locomotion, not coordination).
     • ⚑ Verify: pretrained weights may not be downloadable and its policies are
       task-specific (no generic setpoint tracker) → plan to train the low-level
       controller yourself from its configs. See components/wheeledlab.json. -->

I think we can expand on this, the theme of the hackathon is Imagine a world in 2040. We could theorize that every car will be self-driving. If every car will be self-driving, can we solve traffic, limit car crashes to zero, and function in the world without traffic lights or signs (last one is a maybe)?

I think we can use some sort of API to pull a map of San Francisco where all the roads are, and use some sort of census data or public population data to just estimate how many cars are on the road at a certain time in a certain area. It's just so we can get something semi-realistic, like within this X-mile area in downtown San Francisco, there's Y number of cars on the road.

<!-- 🔎 Map + car-count data (concrete tools):
     • MAP: ✅ OSMnx — ox.graph_from_place("San Francisco,...", network_type="drive")
       or graph_from_bbox(bbox=(W,S,E,N), ...) → NetworkX MultiDiGraph
       (nodes=intersections, edges=segments). MIT code / ODbL data. Use the v2
       API (bbox is ONE tuple). components/osmnx.json.
     • CAR COUNTS: no public feed gives "N cars on every SF street now." Use
       Caltrans PeMS (real 5-min FREEWAY counts, free account — register early)
       and/or Caltrans AADT × an hourly curve for a per-hour estimate; Census
       LODES for commute direction. These set num_agents + density in the sim.
       Be honest in the demo: counts are modeled/aggregate. components/traffic-data.json. -->

We can pull something like this for a real-life area in San Francisco. On top of that, we take WheeledLab's car asset + dynamics recipe and use it to build the frozen low-level controller that our trained agents drive through for the physics demo.

The init flow:
  OSMnx → road graph  ─┐
  Overpass / OSMnx-features (school/POI masks) ├─→ build the static world model
  PeMS / AADT (car counts & density)           ┘    (drivable graph + caution
  → load ONCE into the kinematic training env → multi-agent RL loop → Modal GPUs.

<!-- 🔎 The geospatial "what's near X / near schools?" question:
     use OSM proximity — Overpass `(around:500,lat,lon)[amenity=school]` or
     ox.features_from_point(...), + Nominatim for neighborhood names, then
     geopandas sjoin_nearest (reproject to a metric CRS) to flag roads near
     schools. Those masks feed caution-zone weighting in the reward.
     components/proximity-osm.json.
     NOTE on the simulator: the TRAINING "environment that loads the city graph
     and exposes step(action)->(obs,reward)" is the KINEMATIC ENV (vectorized
     JAX, drives on the OSMnx graph) — fast enough to actually train in a
     hackathon. Rigid-body physics shows up later, only for the demo (Isaac).
     components/kinematic-env.json. -->

Then we need to build a multi-agent reinforcement loop with multiple rewards at different weights. (modal is perfect here. We can train on modal gpu’s)

<!-- 🔎 ✅ Modal is the right call, with specifics:
     • Modal = serverless GPU compute, code-first Python. Native fan-out
       (.spawn_map) + modal.Queue/modal.Dict for the rollout-WORKERS + central
       LEARNER topology; modal.Volume (~2.5GB/s) for checkpoints; 24h/call.
     • Because TRAINING is the kinematic JAX env, ANY CUDA GPU works (A100/H100
       fine) — no RT Cores needed, so training is cheap & unconstrained. (The
       RT-Cores requirement only applies to the Isaac DEMO render, on a separate
       RTX/L40S box.)
     • ⚑ Free tier ~$30/mo is thin → apply for startup credits; kinematic env
       trains in minutes-to-hours so cost stays low. components/modal.json. -->

The reason I think this is a standout is that simple reinforcement learning assumes that the reward is stationary, so you move and the reward stays the same. It's very easy to measure and build some sort of policy model to actually get to that reward. With multiple cars, the environment changes entirely every time.  Not only is it building a reinforcement learning policy on how to navigate effectively and safely, but it's also building a reinforcement learning policy to predict how a car (another agent) is going to move and how it will interact with its environment.

<!-- 🔎 This is the MARL non-stationarity problem, and it has a name:
     the standard solution is CTDE — Centralized Training, Decentralized
     Execution — e.g. MAPPO/IPPO: a centralized critic sees global state (all
     cars) during training; each car executes from local obs at run time.
     Framing fix: you do NOT need a separate "predict the other car" model —
     with a centralized critic + others' states in the obs, anticipation is
     learned implicitly. That literally IS your "one continuous model, not two
     separate entities." Pair with the JAX kinematic env → JaxMARL.
     The policy's ACTIONS are setpoints (waypoint/velocity/heading), so the same
     trained policy drives both the kinematic env and the Isaac demo.
     components/marl-framework.json. -->

The rewards I can think of are three:
No crashing, which is the most important reward.
Least time idling, meaning if a car is on the road, They should be moving for as much of the time as possible.
Maybe even distance, kind of like a fun twist on the shortest path problem using this tiered RL loop
Maybe even being capable of turns and navigating through “congested” spaces like in Downtown SF (though this might be a little too vague)

<!-- 🔎 Reward design (concrete, + a gap to watch):
     r = w1·(-collision) + w2·(progress_toward_goal) + w3·(-detour_vs_shortest_path)
         + w4·(turn_smoothness).
     • "prioritize rewards dynamically" = CURRICULUM: collision weight dominant
       first, anneal in efficiency once crash-rate drops.
     • Reward-hacking watchlist: idle-forever (→ idle penalty), circle-driving to
       farm "moving" (→ reward progress-to-goal, not raw speed).
     • "shortest path" baseline is FREE: networkx Dijkstra on the OSMnx graph.
     • Collision = calibrated geometric footprint overlap in the kinematic env
       (no physics engine needed for the no-crash signal). components/reward-system.json. -->

So, in terms of how to actually do this, I need to think about how to effectively set up this RL loop and prioritize certain rewards over others dynamically. I also need to think about how to do this effectively at scale so you're n training for a specific region or specific area, or training for a specific behavior that can be embodied across all different types of city, suburban driving environments.So you'll probably need multiple grabs from our SF roads, or make the map that we use actually really big. We then need to think about the cost of training so many agents at the same time, I think for training: we could do simple robots that don't have any of the physical properties of a car, but for the demo we added physical properties that were pre-trained on Wheeled lab.

<!-- 🔎 ✅ THIS INSTINCT BECAME THE ARCHITECTURE.
     "simple robots without car physics for training, car physics for the demo
      (pretrained on WheeledLab)" = exactly the hierarchical design:
       • TRAIN coordination on the cheap kinematic env (no rigid-body physics) —
         fast, vectorized, real learning curve.
       • DEMO: run the trained policy ON TOP OF the frozen WheeledLab-derived
         low-level controller in Isaac → real car physics, no retraining.
     Key to make it hold together: give the kinematic model the SAME accel /
     steering-rate limits as the car, and keep learned driving gentle, so the
     setpoints the policy learns track well when executed in physics
     (the kinematic↔physics gap grows with aggressive maneuvers).
     • "multiple SF grabs / bigger map" for generalization = domain
       randomization over MAPS — cheap with OSMnx (sample many bbox tiles across
       SF + suburbs). components/kinematic-env.json, components/lowlevel-controller.json. -->

I kind of worked on something similar in the past. We were doing hotel robots, so robots that move around hotels. If there are people walking around, the robot needs to figure out how to not hit the people. The thing is, the people aren't staying still; they're moving, so you need to predict where a person will be at the same time and calculate if that person will be there at the same time that you're going to be there so you can avoid a collision. The way we avoided collisions was to create that prediction path, then treat each person as a magnetic opposing force field and move around it like that. Keep updating the force field center point for wherever the person is predicted to be at that given time. Predict where the person is was an easy RL problem, but the force field was more of a physics motion tuning rather than an RL solution. I don't want to use RL because we have to, I want to use RL because it is the best solution For multi-agent RL, this would be one continuous model, not two separate entities..

<!-- 🔎 Good instinct, maps cleanly onto CTDE:
     the old "predict person path + potential-field avoidance" is a hand-built
     version of what CTDE-MARL learns end-to-end. "One continuous model, not two
     entities" ✅ = CTDE: one policy that implicitly predicts-and-avoids, with a
     centralized critic during training. Optional: keep an artificial-potential-
     field term as reward shaping / safety prior to speed up early learning,
     then let RL take over (hybrid APF+RL is a legitimate de-risking move). -->

I think the reason this can be a good project is because a demo could be very visual, so we can show this kind of no-lag world, no-crash world with 100% opactiy. You could also have kind of like shadow cars moving at a lower opactiy, and you could really show the kind of bottleneck that we're solving with our model.

<!-- 🔎 The demo — and the baseline is FREE and HONEST:
     • The low-opacity "shadow cars" = the SAME coordination policy at an
       early/zero checkpoint (UNTRAINED → gridlock & crashes). The full-opacity
       cars = the TRAINED policy (smooth, zero crashes). The delta between them
       IS the RL result on screen — no separate traffic sim needed.
     • Render OFFLINE from logged trajectories (car_id,t,x,y,heading). Two paths:
       PRIMARY/always-works = deck.gl TripsLayer on a Mapbox SF basemap (sells
       "real San Francisco"); CINEMATIC upgrade = Isaac RTX render of the
       rigid-body cars (trained policy → frozen low-level controller). Build the
       deck.gl path FIRST so you always have a demo. components/viz-demo.json.
     • Surface the reward telemetry (crashes=0, idle-time ↓, detour) as on-screen
       counters. -->

<!-- ════════════════════════════════════════════════════════════════════════
     📦 Deliverables alongside this doc:
       • ./STACK.md            — stack, build order, the honesty check
       • ./components/*.json    — one structured spec per component (12)
       • ./graph.json           — nodes + edges: full dataflow/dependency graph
       • ./components/README.md  — schema legend
     ════════════════════════════════════════════════════════════════════════ -->
