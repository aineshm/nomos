/* SmoothRide Cesium viewer — replays a scene.json (schema v1) of meshed cars on
   3D San Francisco. Terrain + OSM buildings when an ion token is present; flat
   ellipsoid + our extruded GeoJSON buildings otherwise.

   Supports an optional public/manifest.json (written by the training-export
   script) that lists multiple scene snapshots keyed by training iteration.  When
   present the HUD exposes a dropdown so the user can switch between snapshots and
   watch the policy improve.  When absent the viewer falls back to the legacy
   single-file behaviour (public/scene.json). */

const CFG = window.SMOOTHRIDE_CONFIG || { cesiumIonToken: "" };
const WORLD = "trained";          // which world to animate
const CAR_L = 4.6, CAR_W = 2.0, CAR_H = 1.5;   // meters

// ---------------------------------------------------------------------------
// Viewer-level state (initialised once; scene state is reset per loadScene call)
// ---------------------------------------------------------------------------

let _viewer = null;          // single Cesium.Viewer instance reused across loads
let _sceneEntityIds = [];    // IDs of entities added for the current scene (cars, peds, roads, buildings)
let _clockTickListener = null;  // handle for the per-scene onTick subscription

// ---------------------------------------------------------------------------
// Bootstrap: create viewer once, then load from manifest or fallback
// ---------------------------------------------------------------------------

async function main() {
  const hasToken = !!CFG.cesiumIonToken;
  if (hasToken) Cesium.Ion.defaultAccessToken = CFG.cesiumIonToken;

  _viewer = new Cesium.Viewer("cesiumContainer", {
    terrainProvider: hasToken
      ? await Cesium.createWorldTerrainAsync()
      : new Cesium.EllipsoidTerrainProvider(),
    // baseLayer:false => no default Cesium-ion imagery (which 401s without a
    // token). Without a token we drape free OpenStreetMap tiles instead, so the
    // real SF street grid is visible offline; with a token, ion world imagery.
    baseLayer: hasToken ? undefined : false,
    animation: true, timeline: true, baseLayerPicker: false, geocoder: false,
  });
  _viewer.scene.globe.depthTestAgainstTerrain = true;
  window.viewer = _viewer;   // handy for console debugging (single-viewer app)

  if (!hasToken) {
    _viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({
      url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      maximumLevel: 19, credit: "© OpenStreetMap contributors",
    }));
  }

  // OSM Buildings (ion) are viewer-level, loaded once.
  if (hasToken) {
    try { _viewer.scene.primitives.add(await Cesium.createOsmBuildingsAsync()); }
    catch (e) { console.warn("OSM Buildings failed, using per-scene GeoJSON", e); }
  }

  // Try to load the manifest; fall back to the legacy single scene path.
  let manifest = null;
  try {
    const resp = await fetch("public/manifest.json", { cache: "no-store" });
    if (resp.ok) {
      const parsed = await resp.json();
      if (parsed && Array.isArray(parsed.scenes) && parsed.scenes.length > 0) {
        manifest = parsed;
      }
    }
  } catch (_) {
    // manifest absent or malformed — fall back silently
  }

  const select = document.getElementById("scene-select");
  const sceneRow = document.getElementById("scene-row");

  if (manifest) {
    // Populate dropdown with manifest entries (manifest is already sorted by iter).
    manifest.scenes.forEach((entry) => {
      const opt = document.createElement("option");
      opt.value = entry.file;
      opt.textContent = entry.label;
      select.appendChild(opt);
    });

    // Default to the LAST (most-trained) entry.
    select.selectedIndex = manifest.scenes.length - 1;
    sceneRow.style.display = "flex";

    select.addEventListener("change", () => {
      loadScene("public/" + select.value);
    });

    await loadScene("public/" + manifest.scenes[manifest.scenes.length - 1].file);
  } else {
    // Backward-compat: no manifest — load legacy scene.json, keep dropdown hidden.
    sceneRow.style.display = "none";
    await loadScene("public/scene.json");
  }
}

// ---------------------------------------------------------------------------
// Scene loader — fetches one scene file and renders it, replacing the prior scene
// ---------------------------------------------------------------------------

async function loadScene(path) {
  clearScene();

  const scene = await (await fetch(path, { cache: "no-store" })).json();
  if (scene.schema_version !== 1) throw new Error("unsupported schema " + scene.schema_version);
  const meta = scene.meta, world = scene.worlds[WORLD];

  // Per-scene GeoJSON buildings (only used when ion token is absent, since OSM
  // Buildings are a viewer-level primitive loaded once in main()).
  const hasToken = !!CFG.cesiumIonToken;
  if (!hasToken) addGeoJsonBuildings(_viewer, scene);

  drawRoads(_viewer, scene.roads);

  // Time window for playback.
  const start = Cesium.JulianDate.now();
  const stop = Cesium.JulianDate.addSeconds(start, meta.dt * (meta.n_steps - 1), new Cesium.JulianDate());
  _viewer.clock.startTime = start.clone();
  _viewer.clock.stopTime = stop.clone();
  _viewer.clock.currentTime = start.clone();
  _viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
  _viewer.clock.multiplier = 1.0;
  _viewer.timeline.zoomTo(start, stop);

  world.cars.forEach((car) => addCar(_viewer, car, start, meta));
  (world.peds || []).forEach((ped) => addPed(_viewer, ped, start, meta));

  // HUD — trips and crashed both update LIVE at the current frame (start at 0),
  // not the end-of-run totals (which made "crashed" read non-zero before playback).
  document.getElementById("cars").textContent = world.summary.cars;
  document.getElementById("trips").textContent = world.trips_series[0];
  document.getElementById("crashed").textContent =
    world.cars.reduce((n, c) => n + (c.crash[0] || 0), 0);

  _clockTickListener = () => {
    const frac = Cesium.JulianDate.secondsDifference(_viewer.clock.currentTime, start) / meta.dt;
    const f = Math.max(0, Math.min(world.trips_series.length - 1, Math.round(frac)));
    document.getElementById("trips").textContent = world.trips_series[f];
    document.getElementById("crashed").textContent =
      world.cars.reduce((n, c) => n + (c.crash[f] || 0), 0);
  };
  _viewer.clock.onTick.addEventListener(_clockTickListener);

  // Frame the city: fit the scene's bounding sphere with a fixed oblique tilt, so
  // the whole street grid fills the view at any bbox size (a fixed-altitude flyTo
  // leaves a small city as a speck under a horizon of the rest of the state).
  // bounds is [[wLon,sLat],[eLon,nLat]] (SW, NE).
  const [[wLon, sLat], [eLon, nLat]] = meta.bounds;
  const sphere = Cesium.BoundingSphere.fromPoints([
    Cesium.Cartesian3.fromDegrees(wLon, sLat),
    Cesium.Cartesian3.fromDegrees(eLon, nLat),
  ]);
  _viewer.camera.flyToBoundingSphere(sphere, {
    offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-45), sphere.radius * 2.4),
    duration: 0,
  });
}

// ---------------------------------------------------------------------------
// Scene cleanup — removes all per-scene entities and the clock-tick listener
// ---------------------------------------------------------------------------

function clearScene() {
  // Remove only the entities we added for the previous scene.
  _sceneEntityIds.forEach((id) => {
    const entity = _viewer.entities.getById(id);
    if (entity) _viewer.entities.remove(entity);
  });
  _sceneEntityIds = [];

  // Detach the previous onTick handler so it doesn't reference stale data.
  if (_clockTickListener) {
    _viewer.clock.onTick.removeEventListener(_clockTickListener);
    _clockTickListener = null;
  }
}

// ---------------------------------------------------------------------------
// Entity helpers — each wraps viewer.entities.add and records the returned id
// ---------------------------------------------------------------------------

function trackEntity(entity) {
  _sceneEntityIds.push(entity.id);
  return entity;
}

function sampledPosition(car, start, meta) {
  const p = new Cesium.SampledPositionProperty();
  for (let t = 0; t < car.lng.length; t++) {
    const when = Cesium.JulianDate.addSeconds(start, t * meta.dt, new Cesium.JulianDate());
    p.addSample(when, Cesium.Cartesian3.fromDegrees(car.lng[t], car.lat[t], (car.z[t] || 0) + CAR_H / 2));
  }
  return p;
}

// Pedestrians: small upright markers (amber), ~1.7 m tall, walking on the terrain.
const PED_H = 1.7;
function addPed(viewer, ped, start, meta) {
  const pos = new Cesium.SampledPositionProperty();
  for (let t = 0; t < ped.lng.length; t++) {
    const when = Cesium.JulianDate.addSeconds(start, t * meta.dt, new Cesium.JulianDate());
    pos.addSample(when, Cesium.Cartesian3.fromDegrees(ped.lng[t], ped.lat[t], (ped.z[t] || 0) + PED_H / 2));
  }
  trackEntity(viewer.entities.add({
    position: pos,
    cylinder: {
      length: PED_H, topRadius: 0.3, bottomRadius: 0.35,
      material: Cesium.Color.fromCssColorString("#f59e0b"),  // amber, distinct from cars
    },
  }));
}

// Per-car colour by state: red = crashed, green = arrived (trip complete),
// blue = en route (brighter the faster it's going). Crash takes priority — a car
// is at most one of these since arrival/crash are terminal (remove-on-arrival).
const CRASH_RED = Cesium.Color.fromCssColorString("#ef4444");
const DONE_GREEN = Cesium.Color.fromCssColorString("#22c55e");
function carColor(car, i, vmax) {
  if (car.crash[i]) return CRASH_RED;
  if (car.arr && car.arr[i]) return DONE_GREEN;
  const f = Math.max(0, Math.min(1, car.spd[i] / (vmax || 16)));
  return Cesium.Color.fromHsl(0.58, 0.85, 0.4 + 0.25 * f);   // en route: blue, brighter=faster
}

function frameIndex(car, start, time, meta) {
  const f = Cesium.JulianDate.secondsDifference(time, start) / meta.dt;
  return Math.max(0, Math.min(car.lng.length - 1, Math.round(f)));
}

function addCar(viewer, car, start, meta) {
  const pos = sampledPosition(car, start, meta);
  // Orient by the sim's OWN heading (car.hdg), not by velocity: hdg is the car's
  // true facing every step, so the box faces forward while driving and never
  // spins on a stop or a respawn-teleport (where velocity orientation goes wild).
  // Sim heading is radians CCW from east; Cesium heading is CW from north, hence
  // (pi/2 - hdg). Box dimensions.x (CAR_L, the length) lies along that heading.
  const orientation = new Cesium.CallbackProperty((time) => {
    const p = pos.getValue(time);
    if (!p) return undefined;
    // Box length (dimensions.x = body +X) must point along travel. Cesium heading
    // rotates about -Z (clockwise from East), and sim hdg is CCW from East, so the
    // angle that aligns +X with travel is -hdg (NOT pi/2-hdg, which faces sideways).
    const hdg = car.hdg[frameIndex(car, start, time, meta)];
    const hpr = new Cesium.HeadingPitchRoll(-hdg, 0, 0);
    return Cesium.Transforms.headingPitchRollQuaternion(p, hpr);
  }, false);
  trackEntity(viewer.entities.add({
    position: pos,
    orientation: orientation,
    box: {
      dimensions: new Cesium.Cartesian3(CAR_L, CAR_W, CAR_H),
      material: new Cesium.ColorMaterialProperty(new Cesium.CallbackProperty((time) => {
        return carColor(car, frameIndex(car, start, time, meta), meta.vmax);
      }, false)),
    },
  }));
}

function drawRoads(viewer, roads) {
  roads.forEach((seg) => {
    trackEntity(viewer.entities.add({
      polyline: {
        positions: Cesium.Cartesian3.fromDegreesArrayHeights(
          [seg[0][0], seg[0][1], seg[0][2], seg[1][0], seg[1][1], seg[1][2]]),
        width: 3, material: Cesium.Color.fromCssColorString("#22d3ee").withAlpha(0.9),
        clampToGround: false,
      },
    }));
  });
}

function addGeoJsonBuildings(viewer, scene) {
  const fc = scene.buildings;
  if (!fc || !fc.features) return;
  fc.features.forEach((ft) => {
    const ring = ft.geometry.coordinates[0];
    const flat = [];
    ring.forEach((p) => { flat.push(p[0], p[1]); });
    trackEntity(viewer.entities.add({
      polygon: {
        hierarchy: Cesium.Cartesian3.fromDegreesArray(flat),
        extrudedHeight: ft.properties.height || 8,
        material: Cesium.Color.fromCssColorString("#1a2230").withAlpha(0.9),
        outline: false,
      },
    }));
  });
}

main().catch((e) => { console.error(e); alert("Viewer error: " + e.message); });
