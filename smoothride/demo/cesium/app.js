// Real 3D San Francisco + the Nomos fleet driving on it.
// Map = Cesium World Terrain + OSM Buildings. Cars = procedural GLB models
// (worldsim/assets: sedan/suv/coupe) driven along RL rollout trajectories.
//
// Data: public/manifest.json lists pre-rendered RL scenes (training snapshots +
// the champion held-out-Mission run); a HUD dropdown switches between them. The
// legacy synthetic lane demo (public/trajectories.json) stays available as an
// explicitly-labeled dropdown entry.
//
// Car color = state: blue en-route (brighter = faster), green once arrived
// (fades to a ghost — no longer an obstacle), red for 3 s after a crash, then
// the wreck is removed.
//
// Default Cesium ion token so the 3D map (World Terrain + OSM Buildings) renders
// out-of-the-box with no setup. Resolution order: ?ionToken= URL param > a local
// (git-ignored) config.js > this embedded default. Revoke/rotate at ion.cesium.com.
const DEFAULT_ION_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI0NGQxMjllNi04MjE2LTQyZGItYWI0NC1iYWNmY2QzYzIxMjMiLCJpZCI6NDQ3MDIxLCJpc3MiOiJodHRwczovL2FwaS5jZXNpdW0uY29tIiwiYXVkIjoidW5kZWZpbmVkX2RlZmF1bHQiLCJpYXQiOjE3ODE5OTE1NzZ9.8JXp9n3BBFNeOy-mksz2qTYS5NOZohHKmuQgdzzKbdo";
const Q = new URLSearchParams(location.search);
const TOKEN = Q.get("ionToken")
  || (window.CESIUM_ION_TOKEN && window.CESIUM_ION_TOKEN !== "PASTE_YOUR_CESIUM_ION_TOKEN"
        ? window.CESIUM_ION_TOKEN : null)
  || DEFAULT_ION_TOKEN;

const SF = { lon: -122.4090, lat: 37.7886 };
const MANIFEST_URL = "./public/manifest.json";
const LEGACY_URL = "./public/trajectories.json";
const LEGACY_KEY = "__legacy__";

// ?lite=1 — "meeting mode": coarser buildings, smaller caches, no facade skin.
// Use when screen-sharing on a call so the tab stays light on RAM/GPU.
const LITE = Q.get("lite") === "1";
// Tile cache. 256 MB keeps panning smooth without ballooning the tab; the old
// 1 GB setting could push the tab past 2 GB during a long session.
const TILE_CACHE_BYTES = (LITE ? 96 : 256) * 1024 * 1024;
// ?cars=N — cap the rendered fleet (HUD metrics follow the cap).
const CAR_CAP = parseInt(Q.get("cars"), 10) || Infinity;

// Facade images: each building is randomly skinned with one of these. URL-encode the
// folder name because it contains a space ("building sides").
const SIDE_IMAGES = [1, 2, 3, 4, 5].map((n) => `./building%20sides/side${n}.jpg`);
// "tile" = pictures stacked up the facade at a fixed size (default);
// "single" = stretch one photo per facade; "off" = plain OSM buildings.
const SKIN_MODE = LITE ? "off" : (Q.get("skin") || "tile");

// ---- car/ped state colors ----
const CRASH_RED = Cesium.Color.fromCssColorString("#ef4444");
const DONE_GREEN = Cesium.Color.fromCssColorString("#22c55e");
const DONE_GREEN_GHOST = DONE_GREEN.withAlpha(0.35);
const CRASH_LINGER_S = 3;   // wreck stays visible this long, then is removed
const ARRIVED_FADE_S = 4;   // arrived cars ghost out after this long (not obstacles)
function enRouteBlue(frac) { // brighter = faster
  return Cesium.Color.fromHsl(0.58, 0.85, 0.40 + 0.25 * Math.max(0, Math.min(1, frac)));
}

const PED_SHIRTS = ["#2a7de1", "#e14b4b", "#2bb673", "#e0a92b", "#7b5bd6", "#d94f8a", "#e7ecf3", "#1f9e9e"]
  .map((h) => Cesium.Color.fromCssColorString(h));
const PED_SKINS = ["#f0c9a8", "#e8b98c", "#c98a5b", "#a86a3c"]
  .map((h) => Cesium.Color.fromCssColorString(h));

// ---- the procedural fleet: 3 body GLBs (worldsim/assets) ----
const BODIES = ["sedan", "suv", "coupe"];

// deterministic per-car RNG -> a car keeps its body across frames/reloads
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function carBody(i) {
  const r = mulberry32(0x5eed + (i * 2654435761) | 0);
  return BODIES[Math.floor(r() * BODIES.length)];
}
function pedLook(i) {
  const r = mulberry32(0xa11ce + (i * 2654435761 | 0));
  return { shirt: PED_SHIRTS[Math.floor(r() * PED_SHIRTS.length)],
           skin: PED_SKINS[Math.floor(r() * PED_SKINS.length)] };
}

function msg(html, sticky) {
  const el = document.getElementById("msg");
  el.innerHTML = html;
  if (html) el.setAttribute("data-show", ""); else el.removeAttribute("data-show");
  if (!sticky && html) setTimeout(() => el.removeAttribute("data-show"), 4000);
}
window.addEventListener("error", (e) => msg(`<div>Error: <code>${e.message}</code></div>`, true));
window.addEventListener("unhandledrejection",
  (e) => msg(`<div>Error: <code>${(e.reason && (e.reason.message || e.reason)) || e.reason}</code></div>`, true));

// ---------------------------------------------------------------------------
// Boot: viewer + terrain + buildings once; scenes load/swap on top of it.
// ---------------------------------------------------------------------------

let worldTerrain = null; // kept so scene loads can batch-sample real heights

async function boot() {
  if (typeof Cesium === "undefined") return msg("<div>Cesium failed to load (CDN).</div>", true);
  if (TOKEN) Cesium.Ion.defaultAccessToken = TOKEN;

  const opts = {
    animation: true, timeline: true, geocoder: false, baseLayerPicker: false,
    homeButton: false, sceneModePicker: false, navigationHelpButton: false,
    fullscreenButton: false, infoBox: false, selectionIndicator: false,
  };
  if (TOKEN) {
    try { worldTerrain = await Cesium.createWorldTerrainAsync(); opts.terrainProvider = worldTerrain; }
    catch (e) { console.warn("World Terrain unavailable:", e); }
  }
  if (!worldTerrain) opts.baseLayer = false;

  const viewer = new Cesium.Viewer("cesiumContainer", opts);
  viewer.scene.globe.depthTestAgainstTerrain = !!worldTerrain;
  window.viewer = viewer; // exposed for the GIF capture harness

  if (TOKEN) {
    try {
      const osm = await Cesium.createOsmBuildingsAsync();
      viewer.scene.primitives.add(osm);
      window.osmBuildings = osm; // exposed for the GIF capture harness
      // Faster first paint: load coarser tiles, skip intermediate LODs instead of
      // streaming every level. Override the look with ?sse=<n> (lower = sharper).
      const sse = Number(Q.get("sse")) || (LITE ? 28 : 20);
      osm.maximumScreenSpaceError = sse;          // detail target
      osm.skipLevelOfDetail = true;               // jump straight toward the target LOD
      osm.baseScreenSpaceError = 1024;
      osm.skipScreenSpaceErrorFactor = 16;
      osm.skipLevels = 1;
      // SOLID buildings: dynamic-SSE dithers/fades tiles while the camera moves,
      // which reads as low-opacity ghost buildings — turn it off so facades stay opaque.
      osm.dynamicScreenSpaceError = false;
      // Load NEARBY first + keep what's loaded so panning back doesn't re-stream:
      osm.foveatedScreenSpaceError = true;        // sharp near screen centre...
      osm.foveatedConeSize = 0.3;                 // ...cheaper at the edges
      osm.preloadFlightDestinations = true;       // prefetch where the camera is heading
      osm.cacheBytes = TILE_CACHE_BYTES;
      osm.maximumCacheOverflowBytes = TILE_CACHE_BYTES / 2;
      if (SKIN_MODE !== "off") {
        try {
          const atlas = await buildImageAtlas(SIDE_IMAGES);
          // Square footprint (wall == floor) so the square atlas cell isn't
          // re-stretched on the wall -> pictures stay undistorted as they stack.
          skinBuildingsWithAtlas(osm, atlas, {
            mode: SKIN_MODE, center: [SF.lon, SF.lat], wallMeters: 16, floorMeters: 16,
          });
        } catch (e) { console.warn("atlas skin failed — plain buildings:", e); }
      }
    } catch (e) { console.warn("OSM Buildings unavailable:", e); }
  } else {
    viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", maximumLevel: 19,
      credit: "© OpenStreetMap contributors",
    }));
  }

  await setupScenePicker(viewer);
}

// ---------------------------------------------------------------------------
// Scene picker: manifest.json -> dropdown; legacy lane demo as a labeled extra.
// ---------------------------------------------------------------------------

async function setupScenePicker(viewer) {
  let manifest = null;
  try {
    const resp = await fetch(MANIFEST_URL, { cache: "no-store" });
    if (resp.ok) {
      const parsed = await resp.json();
      if (parsed && Array.isArray(parsed.scenes) && parsed.scenes.length) manifest = parsed;
    }
  } catch (_) { /* no manifest — legacy only */ }

  const select = document.getElementById("scene-select");
  const entries = [];
  if (manifest) {
    manifest.scenes.forEach((s) => entries.push({ key: "./public/" + s.file, label: s.label }));
  }
  entries.push({ key: LEGACY_KEY, label: "ambient traffic (synthetic, not RL)" });

  entries.forEach((e) => {
    const opt = document.createElement("option");
    opt.value = e.key; opt.textContent = e.label;
    select.appendChild(opt);
  });

  // Default: ?scene=<substring of file/label>, else the champion entry, else the
  // most-trained snapshot (last manifest entry), else legacy.
  const want = (Q.get("scene") || "").toLowerCase();
  let idx = entries.findIndex((e) =>
    want && (e.key.toLowerCase().includes(want) || e.label.toLowerCase().includes(want)));
  if (idx < 0) idx = entries.findIndex((e) => e.key.toLowerCase().includes("champion"));
  if (idx < 0) idx = Math.max(0, entries.length - 2); // last real scene before legacy
  select.selectedIndex = idx;
  select.addEventListener("change", () => loadScene(viewer, select.value));

  await loadScene(viewer, entries[idx].key);
}

// ---------------------------------------------------------------------------
// Scene loading. All per-scene entities live in one CustomDataSource so a
// scene switch is a clean teardown (cars, peds, roads all go together).
// ---------------------------------------------------------------------------

let sceneSource = null;   // CustomDataSource of the current scene
let hudTeardown = null;   // removes the current HUD onTick listener
let pedFx = null;         // dense-ped point collection + its per-frame updater
let loadSeq = 0;          // guards against overlapping loads (fast dropdown switches)

async function loadScene(viewer, key) {
  const seq = ++loadSeq;
  msg("<div>Loading scene…</div>", true);
  try {
    const legacy = key === LEGACY_KEY;
    const url = legacy ? LEGACY_URL : key;
    // cache:no-store so a freshly re-exported scene is never served stale.
    const raw = await (await fetch(url, { cache: "no-store" })).json();
    const scene = normalizeScene(raw, legacy);

    // Bake REAL terrain heights once per scene (batched, then interpolated per
    // frame). The exports carry synthetic elevation that doesn't match Cesium
    // World Terrain, and per-frame CLAMP_TO_GROUND costs a scene pick per car
    // per frame (7 fps at ~170 cars) — this is the fast + correct middle path.
    await bakeHeights(scene);

    if (seq !== loadSeq) return;  // a newer selection superseded this load
    if (sceneSource) { viewer.dataSources.remove(sceneSource, true); sceneSource = null; }
    if (hudTeardown) { hudTeardown(); hudTeardown = null; }
    if (pedFx) {
      viewer.scene.preUpdate.removeEventListener(pedFx.update);
      viewer.scene.primitives.remove(pedFx.collection);
      pedFx = null;
    }

    sceneSource = new Cesium.CustomDataSource("scene");
    await viewer.dataSources.add(sceneSource);

    addFleet(viewer, scene);
    msg("");
    flyToScene(viewer, scene);
  } catch (e) {
    if (seq !== loadSeq) return;
    console.error("scene load failed:", e);
    msg(`<div>Couldn't load this scene: <code>${e.message}</code></div>`, true);
  }
}

// Normalize both file formats to { meta, cars, peds, tripsSeries, roads }.
//   scene_*.json:       { meta, roads, worlds: { trained: { cars, peds, trips_series } } }
//   trajectories.json:  { meta (+trips_series), worlds: { trained: { cars } } }
function normalizeScene(raw, legacy) {
  const world = raw.worlds && (raw.worlds.trained || raw.worlds[Object.keys(raw.worlds)[0]]);
  if (!world || !world.cars || !world.cars.length) throw new Error("no cars in scene data");
  const meta = raw.meta || {};
  if (!meta.dt || !meta.n_steps) throw new Error("scene meta missing dt/n_steps");

  let cars = world.cars;
  if (cars.length > CAR_CAP) cars = cars.slice(0, CAR_CAP);

  const peds = (world.peds && world.peds.length)
    ? world.peds
    : (legacy ? synthPeds(cars, meta.n_steps, meta.dt) : []);

  return {
    meta, cars, peds,
    tripsSeries: world.trips_series || meta.trips_series || null,
    roads: raw.roads || null,
    legacy,
  };
}

// ---------------------------------------------------------------------------
// Terrain-height baking: one batched sample per scene, lerped to every frame.
// ---------------------------------------------------------------------------

const BAKE_STRIDE = 8;     // sample every Nth frame per track, lerp between
const BAKE_CHUNK = 3000;   // points per sampleTerrainMostDetailed call

async function bakeHeights(scene) {
  const tracks = [...scene.cars, ...scene.peds];
  if (!worldTerrain) { // no terrain -> flat ellipsoid, height 0 everywhere
    tracks.forEach((tr) => { tr.h = new Float32Array(tr.lng.length); });
    return;
  }
  const cartos = [], owners = [];
  tracks.forEach((tr) => {
    const n = tr.lng.length;
    const anchors = [];
    for (let t = 0; t < n; t += BAKE_STRIDE) anchors.push(t);
    if (anchors[anchors.length - 1] !== n - 1) anchors.push(n - 1);
    tr.__anchors = anchors;
    anchors.forEach((t) => {
      cartos.push(Cesium.Cartographic.fromDegrees(tr.lng[t], tr.lat[t]));
      owners.push(tr);
    });
  });
  try {
    for (let i = 0; i < cartos.length; i += BAKE_CHUNK)
      await Cesium.sampleTerrainMostDetailed(worldTerrain, cartos.slice(i, i + BAKE_CHUNK));
  } catch (e) {
    console.warn("terrain sampling failed — using height 0:", e);
    tracks.forEach((tr) => { tr.h = new Float32Array(tr.lng.length); delete tr.__anchors; });
    return;
  }
  // scatter sampled heights back to their tracks, then lerp between anchors
  let k = 0;
  tracks.forEach((tr) => {
    const n = tr.lng.length, anchors = tr.__anchors;
    const h = new Float32Array(n);
    const av = new Array(anchors.length);
    for (let a = 0; a < anchors.length; a++, k++) av[a] = cartos[k].height || 0;
    for (let a = 0; a < anchors.length - 1; a++) {
      const t0 = anchors[a], t1 = anchors[a + 1];
      for (let t = t0; t <= t1; t++) {
        const f = t1 === t0 ? 0 : (t - t0) / (t1 - t0);
        h[t] = av[a] + (av[a + 1] - av[a]) * f;
      }
    }
    h[n - 1] = av[av.length - 1];
    tr.h = h;
    delete tr.__anchors;
  });
}

// ---------------------------------------------------------------------------
// Fleet: clock + cars + peds + roads + HUD for one scene.
// ---------------------------------------------------------------------------

function addFleet(viewer, scene) {
  const { meta, cars, peds } = scene;
  const NF = meta.n_steps, DT = meta.dt;
  window.__DT = DT;                 // for the ?t=<frame> capture param
  const start = Cesium.JulianDate.now();
  const stop = Cesium.JulianDate.addSeconds(start, NF * DT, new Cesium.JulianDate());
  Object.assign(viewer.clock, {
    startTime: start.clone(), stopTime: stop.clone(), currentTime: start.clone(),
    clockRange: Cesium.ClockRange.LOOP_STOP, multiplier: 2.0, shouldAnimate: true,
  });
  if (viewer.timeline) viewer.timeline.zoomTo(start, stop);

  const timeAt = (t) => Cesium.JulianDate.addSeconds(start, t * DT, new Cesium.JulianDate());

  if (scene.roads && Q.get("roads") === "1") addRoads(scene.roads);

  const carEntities = cars.map((c, i) => addCar(c, i, { start, timeAt, NF, DT, meta }));

  // ?track=<i> -> chase one car (handy for eyeballing orientation up close).
  const track = parseInt(Q.get("track"), 10);
  if (!isNaN(track) && carEntities[track]) viewer.trackedEntity = carEntities[track];

  // Few peds -> full 3D figures (entities). Dense crowds -> a raw point
  // collection updated per frame: entity cylinders with staggered availability
  // force geometry re-batching every time a ped expires (~10 fps at 288 peds).
  if (peds.length <= PED_FIGURE_MAX) peds.forEach((p, i) => addPed(p, i, timeAt, NF));
  else pedFx = addPedsDense(viewer, peds, start, DT, NF);

  hudTeardown = setupHUD(viewer, scene, start, DT, NF);

  const tf = parseInt(Q.get("t"), 10);
  if (!isNaN(tf)) {
    viewer.clock.currentTime = Cesium.JulianDate.addSeconds(start, tf * DT, new Cesium.JulianDate());
    if (Q.get("pause") === "1") viewer.clock.shouldAnimate = false;
  }
}

// The env can RESPAWN a car (on goal/crash) at a new spot on its route. That's a
// teleport: interpolating across it streaks the car over the whole map. So find
// the car's LONGEST continuous run — break wherever it jumps more than a car
// could plausibly move in one step — and render that as the car's trip.
function longestTrip(c, NF, DT, vmax) {
  const JUMP = Math.max(25, (vmax || 16) * DT * 5);
  const pos = (t) => Cesium.Cartesian3.fromDegrees(c.lng[t], c.lat[t], 0);
  let best = null, s = 0;
  const push = (a, b) => { if (b > a && (!best || b - a > best[1] - best[0])) best = [a, b]; };
  for (let t = 1; t < NF; t++) {
    if (Cesium.Cartesian3.distance(pos(t - 1), pos(t)) > JUMP) { push(s, t - 1); s = t; }
  }
  push(s, NF - 1);
  return best;
}

const CAR_LIFT = 0.15;  // small lift so wheels sit on the terrain skin

function addCar(c, i, ctx) {
  const { timeAt, NF, DT, meta, start } = ctx;
  const trip = longestTrip(c, NF, DT, meta.vmax);
  if (!trip) return null;
  let [t0, t1] = trip;

  // first crash frame within the trip (crash flags are persistent/cumulative)
  let tCrash = -1;
  if (c.crash) for (let t = t0; t <= t1; t++) if (c.crash[t]) { tCrash = t; break; }
  // first arrived frame within the trip
  let tArr = -1;
  if (c.arr) for (let t = t0; t <= t1; t++) if (c.arr[t]) { tArr = t; break; }

  // A crashed car freezes at the crash point, lingers CRASH_LINGER_S, then is
  // removed — mirroring the env, where wrecks stop being obstacles.
  if (tCrash >= 0) t1 = Math.min(t1, tCrash);

  const pos = new Cesium.SampledPositionProperty();
  for (let t = t0; t <= t1; t++)
    pos.addSample(timeAt(t), Cesium.Cartesian3.fromDegrees(c.lng[t], c.lat[t], (c.h ? c.h[t] : 0) + CAR_LIFT));
  pos.setInterpolationOptions({ interpolationDegree: 1,
    interpolationAlgorithm: Cesium.LinearApproximation });
  pos.forwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
  pos.backwardExtrapolationType = Cesium.ExtrapolationType.HOLD;

  // Orientation from the exported heading, NOT velocity: VelocityOrientationProperty
  // goes undefined at zero speed, so a stopped car would snap to a default facing.
  // Cesium heading = -hdg (our hdg is CCW-from-east; Cesium heading is CW-from-north).
  const ori = new Cesium.SampledProperty(Cesium.Quaternion);
  for (let t = t0; t <= t1; t++) {
    const hpr = new Cesium.HeadingPitchRoll(-c.hdg[t], 0, 0);
    ori.addSample(timeAt(t), Cesium.Transforms.headingPitchRollQuaternion(
      Cesium.Cartesian3.fromDegrees(c.lng[t], c.lat[t], (c.h ? c.h[t] : 0) + CAR_LIFT), hpr));
  }
  ori.forwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
  ori.backwardExtrapolationType = Cesium.ExtrapolationType.HOLD;

  // State color, resolved per tick: red (crashed) > green (arrived; ghosts out
  // after ARRIVED_FADE_S — no longer an obstacle) > blue (en-route, brighter=faster).
  const frameAt = (time) => Math.max(t0, Math.min(t1,
    Math.round(Cesium.JulianDate.secondsDifference(time, start) / DT)));
  const color = new Cesium.CallbackProperty((time, result) => {
    const f = frameAt(time);
    if (tCrash >= 0 && f >= tCrash) return Cesium.Color.clone(CRASH_RED, result);
    if (tArr >= 0 && f >= tArr) {
      const since = Cesium.JulianDate.secondsDifference(time, timeAt(tArr));
      return Cesium.Color.clone(since > ARRIVED_FADE_S ? DONE_GREEN_GHOST : DONE_GREEN, result);
    }
    return Cesium.Color.clone(enRouteBlue((c.spd ? c.spd[f] : 0) / (meta.vmax || 16)), result);
  }, false);

  // Availability: crashed cars leave the world CRASH_LINGER_S after the crash.
  let availability;
  if (tCrash >= 0) {
    availability = new Cesium.TimeIntervalCollection([new Cesium.TimeInterval({
      start: timeAt(0),
      stop: Cesium.JulianDate.addSeconds(timeAt(tCrash), CRASH_LINGER_S, new Cesium.JulianDate()),
    })]);
  }

  // Distance LOD: a cheap dot always present + the 3D model only near the camera.
  // They CROSS-FADE (dot out exactly as the model comes in) so it reads as detail
  // resolving, not a spawn. Cesium frustum-culls off-screen entities for free.
  const MODEL_FAR = 600;
  return sceneSource.entities.add({
    position: pos,
    orientation: ori,
    availability,
    point: {
      pixelSize: 7, color,
      outlineColor: Cesium.Color.BLACK.withAlpha(0.4), outlineWidth: 1,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      translucencyByDistance: new Cesium.NearFarScalar(MODEL_FAR * 0.6, 0.0, MODEL_FAR, 1.0),
    },
    model: {
      uri: `./assets/${carBody(i)}.glb`,
      minimumPixelSize: 24, maximumScale: 12, scale: 1.0,
      color, colorBlendMode: Cesium.ColorBlendMode.MIX, colorBlendAmount: 0.8,
      distanceDisplayCondition: new Cesium.DistanceDisplayCondition(0.0, MODEL_FAR),
    },
  });
}

const PED_FIGURE_MAX = 40;   // above this, peds render as points, not 3D figures
const PED_AMBER = Cesium.Color.fromCssColorString("#f59e0b");
const PED_OUTLINE = Cesium.Color.BLACK.withAlpha(0.4);

// Dense crowds: one PointPrimitiveCollection, positions written per frame from
// the raw tracks (lerped between samples). No entities, no availability, no
// geometry re-batching — flat cost per ped. Finished peds hide after the same
// linger as crashed cars (they're no longer obstacles in the env).
function addPedsDense(viewer, peds, start, DT, NF) {
  const collection = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
  const items = peds.map((p) => {
    const len = Math.min(p.lng.length, NF);
    let lastMove = -1;
    for (let t = 1; t < len; t++) {
      if (Math.abs(p.lng[t] - p.lng[t - 1]) > 1e-7 ||
          Math.abs(p.lat[t] - p.lat[t - 1]) > 1e-7) lastMove = t;
    }
    const endF = (lastMove > 0 && lastMove < len - 1)
      ? lastMove + CRASH_LINGER_S / DT : Infinity;
    const point = collection.add({
      pixelSize: 5, color: PED_AMBER, outlineColor: PED_OUTLINE, outlineWidth: 1,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
    });
    return { p, len, endF, point };
  });
  const update = () => {
    const s = Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, start) / DT;
    for (const it of items) {
      if (s > it.endF) { it.point.show = false; continue; }
      it.point.show = true;
      const f = Math.max(0, Math.min(it.len - 1, s));
      const i0 = Math.floor(f), i1 = Math.min(it.len - 1, i0 + 1), fr = f - i0;
      const lng = it.p.lng[i0] + (it.p.lng[i1] - it.p.lng[i0]) * fr;
      const lat = it.p.lat[i0] + (it.p.lat[i1] - it.p.lat[i0]) * fr;
      const h = it.p.h ? it.p.h[i0] + (it.p.h[i1] - it.p.h[i0]) * fr : 0;
      it.point.position = Cesium.Cartesian3.fromDegrees(lng, lat, h + 1.0);
    }
  };
  viewer.scene.preUpdate.addEventListener(update);
  update();
  return { collection, update };
}

// ---- pedestrians as simple 3D characters (body cylinder + head sphere) ----
// A ped that finished its trip is REMOVED a few seconds later (mirrors the env,
// where finished peds stop being obstacles). "Finished" is inferred from the
// track: the last frame where it actually moved. Never-started peds stay put.
function addPed(p, i, timeAt, NF) {
  const len = Math.min(p.lng.length, NF);
  const look = pedLook(i);

  let lastMove = -1;
  for (let t = 1; t < len; t++) {
    if (Math.abs(p.lng[t] - p.lng[t - 1]) > 1e-7 ||
        Math.abs(p.lat[t] - p.lat[t - 1]) > 1e-7) lastMove = t;
  }
  let availability;
  if (lastMove > 0 && lastMove < len - 1) {  // finished mid-run -> leaves the world
    availability = new Cesium.TimeIntervalCollection([new Cesium.TimeInterval({
      start: timeAt(0),
      stop: Cesium.JulianDate.addSeconds(timeAt(lastMove), CRASH_LINGER_S, new Cesium.JulianDate()),
    })]);
  }
  const partPos = (dz) => {
    const pos = new Cesium.SampledPositionProperty();
    for (let t = 0; t < len; t++)
      pos.addSample(timeAt(t), Cesium.Cartesian3.fromDegrees(p.lng[t], p.lat[t], (p.h ? p.h[t] : 0) + dz));
    pos.setInterpolationOptions({ interpolationDegree: 1, interpolationAlgorithm: Cesium.LinearApproximation });
    pos.forwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    pos.backwardExtrapolationType = Cesium.ExtrapolationType.HOLD;
    return pos;
  };
  // body: a slightly tapered cylinder ~1.1 m tall (centre ~0.6 m up), plus an
  // amber dot that fades in at distance — a 1.4 m figure is subpixel from the
  // default viewing altitude, and the dot is what makes crossings readable.
  sceneSource.entities.add({
    position: partPos(0.6),
    availability,
    cylinder: { length: 1.1, topRadius: 0.17, bottomRadius: 0.24, material: look.shirt },
    point: {
      pixelSize: 5, color: Cesium.Color.fromCssColorString("#f59e0b"),
      outlineColor: Cesium.Color.BLACK.withAlpha(0.4), outlineWidth: 1,
      disableDepthTestDistance: Number.POSITIVE_INFINITY,
      translucencyByDistance: new Cesium.NearFarScalar(120, 0.0, 320, 1.0),
    },
  });
  // head: a small sphere (~1.38 m up)
  sceneSource.entities.add({
    position: partPos(1.38),
    availability,
    ellipsoid: { radii: new Cesium.Cartesian3(0.16, 0.16, 0.19), material: look.skin },
  });
}

// Synthesized ambient crowd for the legacy lane demo (it exports no peds):
// anchor each walker near a real car-route point, then random-walk slowly.
function synthPeds(cars, NF, DT) {
  if (!cars || !cars.length) return [];
  const n = Math.min(30, Math.max(16, Math.round(cars.length / 6)));
  const rng = mulberry32(0x9ed5eed);
  const peds = [];
  for (let i = 0; i < n; i++) {
    const c = cars[Math.floor(rng() * cars.length)];
    const t0 = Math.floor(rng() * Math.max(1, c.lng.length - 1));
    const lat0 = c.lat[t0], lng0 = c.lng[t0], cosl = Math.cos(lat0 * Math.PI / 180);
    const off = 6 + rng() * 9, a0 = rng() * Math.PI * 2;
    let x = Math.cos(a0) * off, y = Math.sin(a0) * off;
    let hd = rng() * Math.PI * 2; const spd = 0.7 + rng() * 0.9;
    const lng = [], lat = [];
    for (let t = 0; t < NF; t++) {
      hd += (rng() - 0.5) * 0.35;
      x += Math.cos(hd) * spd * DT; y += Math.sin(hd) * spd * DT;
      if (Math.hypot(x, y) > 32) hd += Math.PI;          // wander, but stay local
      lng.push(lng0 + x / (111320 * cosl));
      lat.push(lat0 + y / 111320);
    }
    peds.push({ lng, lat });
  }
  return peds;
}

// Road-graph overlay (?roads=1): the exact OSM edges the env drives on.
function addRoads(roads) {
  const mat = Cesium.Color.fromCssColorString("#22d3ee").withAlpha(0.3);
  roads.forEach((seg) => {
    sceneSource.entities.add({
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArray(
          [seg[0][0], seg[0][1], seg[1][0], seg[1][1]]),
        width: 2, material: mat, clampToGround: true,
      },
    });
  });
}

// Camera framing. URL params let an external capture harness frame a specific
// intersection: ?lon=&lat=&alt=&pitch=&heading=
function flyToScene(viewer, scene) {
  const center = scene.meta.center || [SF.lon, SF.lat];
  const camLon = parseFloat(Q.get("lon")), camLat = parseFloat(Q.get("lat"));
  const alt = parseFloat(Q.get("alt")) || 300;
  const pitch = parseFloat(Q.get("pitch")) || -32;
  const heading = parseFloat(Q.get("heading")) || 0;
  const dest = (!isNaN(camLon) && !isNaN(camLat))
    ? Cesium.Cartesian3.fromDegrees(camLon, camLat, alt)
    : Cesium.Cartesian3.fromDegrees(center[0], center[1] - 0.0019, alt);
  viewer.camera.flyTo({
    destination: dest,
    orientation: { heading: Cesium.Math.toRadians(heading),
                   pitch: Cesium.Math.toRadians(pitch), roll: 0 },
    duration: Q.has("lon") ? 0 : 1.5,
  });
}

// ---------------------------------------------------------------------------
// Live tracker: recompute fleet metrics for the CURRENT frame on each tick.
//   Trips  = cumulative arrivals. Moving = cars with speed > 0.5 m/s.
//   Crashes= cars whose crash flag is set this frame. Avg speed of the movers.
// Returns a teardown fn that unhooks the clock listener (for scene switches).
// ---------------------------------------------------------------------------

function setupHUD(viewer, scene, start, dt, nf) {
  const cars = scene.cars;
  const vmax = scene.meta.vmax || 9;
  const n = cars.length;
  const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  set("m-cars", n);
  set("m-peds", scene.peds.length);

  // ---- precompute per-frame series: moving / slowed / arrived / crashed / mean speed ----
  const mov = new Array(nf), jam = new Array(nf), cr = new Array(nf), ms = new Array(nf);
  for (let f = 0; f < nf; f++) {
    let m = 0, j = 0, c = 0, sp = 0, k = 0;
    for (const car of cars) {
      if (car.crash && car.crash[f]) { c++; continue; }
      if (car.arr && car.arr[f]) continue;   // arrived cars are out of the fleet
      const s = car.spd ? Math.max(0, car.spd[f]) : 0;
      if (s > 0.5) m++; else j++;
      sp += s; k++;
    }
    mov[f] = m; jam[f] = j; cr[f] = c; ms[f] = k ? sp / k : 0;
  }
  // trips: prefer the exported series, else accumulate from per-car arrival flags
  const trips = scene.tripsSeries || (() => {
    const a = new Array(nf);
    for (let f = 0; f < nf; f++) { let d = 0; for (const c of cars) if (c.arr && c.arr[f]) d++; a[f] = d; }
    return a;
  })();

  // ---- chart canvases (sized to their CSS box, DPR-aware) ----
  const ids = ["dc-trips", "dc-fleet", "dc-hist"];
  const ctx = {};
  function sizeCanvases() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    ids.forEach((id) => {
      const c = document.getElementById(id); if (!c) return;
      c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
      const x = c.getContext("2d"); x.setTransform(dpr, 0, 0, dpr, 0, 0); ctx[id] = x;
    });
  }
  sizeCanvases();
  window.addEventListener("resize", sizeCanvases);

  function drawLine(id, data, col, fill, f) {
    const x = ctx[id]; if (!x) return; const w = x.canvas.clientWidth, h = x.canvas.clientHeight;
    x.clearRect(0, 0, w, h);
    let mx = 1; for (const v of data) mx = Math.max(mx, v);
    x.strokeStyle = "rgba(255,255,255,.06)"; x.lineWidth = 1;
    for (let g = 0; g <= 2; g++) { const yy = 3 + g / 2 * (h - 9); x.beginPath(); x.moveTo(0, yy); x.lineTo(w, yy); x.stroke(); }
    const pt = (i) => [i / (nf - 1) * w, h - 3 - data[i] / mx * (h - 9)];
    if (fill) { x.beginPath(); for (let i = 0; i < nf; i++) { const [xx, yy] = pt(i); i ? x.lineTo(xx, yy) : x.moveTo(xx, yy); } x.lineTo(w, h); x.lineTo(0, h); x.closePath(); x.fillStyle = fill; x.fill(); }
    x.beginPath(); for (let i = 0; i < nf; i++) { const [xx, yy] = pt(i); i ? x.lineTo(xx, yy) : x.moveTo(xx, yy); } x.strokeStyle = col; x.lineWidth = 2; x.stroke();
    const px = f / (nf - 1) * w; x.strokeStyle = "rgba(52,211,153,.5)"; x.lineWidth = 1; x.beginPath(); x.moveTo(px, 0); x.lineTo(px, h); x.stroke();
  }
  function drawStack(id, f) {
    const x = ctx[id]; if (!x) return; const w = x.canvas.clientWidth, h = x.canvas.clientHeight;
    x.clearRect(0, 0, w, h);
    const mx = Math.max(1, n);
    const layer = (arr, base, col) => {
      x.beginPath();
      for (let i = 0; i < nf; i++) { const xx = i / (nf - 1) * w, yy = h - (base[i] + arr[i]) / mx * h; i ? x.lineTo(xx, yy) : x.moveTo(xx, yy); }
      for (let i = nf - 1; i >= 0; i--) { const xx = i / (nf - 1) * w, yy = h - base[i] / mx * h; x.lineTo(xx, yy); }
      x.closePath(); x.fillStyle = col; x.fill();
    };
    const b1 = mov.slice(), b2 = mov.map((m, i) => m + jam[i]);
    layer(mov, new Array(nf).fill(0), "rgba(52,211,153,.55)");
    layer(jam, b1, "rgba(245,158,11,.6)");
    layer(cr, b2, "rgba(239,68,68,.8)");
    const px = f / (nf - 1) * w; x.strokeStyle = "rgba(52,211,153,.5)"; x.lineWidth = 1; x.beginPath(); x.moveTo(px, 0); x.lineTo(px, h); x.stroke();
  }
  function drawHist(id, f) {
    const x = ctx[id]; if (!x) return; const w = x.canvas.clientWidth, h = x.canvas.clientHeight, B = 8, bins = new Array(B).fill(0);
    let cnt = 0;
    for (const car of cars) { if (car.crash && car.crash[f]) continue; if (car.arr && car.arr[f]) continue; const s = car.spd ? Math.max(0, car.spd[f]) : 0; bins[Math.min(B - 1, Math.floor(s / vmax * B))]++; cnt++; }
    const mx = Math.max(1, ...bins), bw = w / B;
    x.clearRect(0, 0, w, h);
    for (let i = 0; i < B; i++) { const bh = bins[i] / mx * (h - 3), xx = i * bw + 2; x.fillStyle = `hsl(${(i + 0.5) / B * 140},80%,52%)`; x.fillRect(xx, h - bh, bw - 4, bh); }
    set("dc-hn", cnt + " cars");
  }

  const update = () => {
    const f = Math.max(0, Math.min(nf - 1, Math.round(
      Cesium.JulianDate.secondsDifference(viewer.clock.currentTime, start) / dt)));
    set("m-trips", trips[f]);
    set("m-moving", mov[f]);
    set("m-crashes", cr[f]);
    set("m-speed", (ms[f] * 2.23694).toFixed(0) + " mph");
    set("m-time", (f * dt).toFixed(0) + "s");
    const bar = (id, lab, v) => { const el = document.getElementById(id); if (el) el.style.width = (v / n * 100) + "%"; set(lab, v); };
    bar("fb-mov", "fbn-mov", mov[f]); bar("fb-jam", "fbn-jam", jam[f]); bar("fb-crash", "fbn-crash", cr[f]);
    drawLine("dc-trips", trips, "#34d399", "rgba(52,211,153,.12)", f);
    drawStack("dc-fleet", f);
    drawHist("dc-hist", f);
  };
  viewer.clock.onTick.addEventListener(update);
  update();
  return () => {
    viewer.clock.onTick.removeEventListener(update);
    window.removeEventListener("resize", sizeCanvases);
  };
}

boot();
