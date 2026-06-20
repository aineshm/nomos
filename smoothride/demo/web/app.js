// SmoothRide web viewer — deck.gl + Mapbox.
// Replays exported trajectories (smoothride.demo.export_web) on the real SF map:
// the TRAINED coordination policy (full opacity) overlaid on the UNTRAINED
// "today's traffic" shadow world (faint), with WheeledLab-style RC-car meshes.

// --- tuning you may need to touch ---------------------------------------
const HEADING_SIGN = 1;        // flip to -1 if cars drive backwards
const HEADING_OFFSET_DEG = 0;  // add 90/180 if the mesh faces the wrong way
const CAR_SIZE = 1.0;          // SimpleMeshLayer sizeScale (mesh is in meters)
const TRAIL_LEN = 14;          // frames of motion trail
const JUMP_DEG = 0.0003;       // split trails across respawn teleports (~33 m)
const CAR_GLB = null;          // set to "models/wheeledlab_car.glb" to use a real asset
// -----------------------------------------------------------------------

const TOKEN = (new URLSearchParams(location.search).get("token"))
  || (window.MAPBOX_TOKEN && window.MAPBOX_TOKEN !== "PASTE_YOUR_TOKEN_HERE"
        ? window.MAPBOX_TOKEN : null);

const WORLD_STYLE = {
  trained:   { opacity: 1.0,  z: 0.0 },
  untrained: { opacity: 0.30, z: 0.0 },
};

// RdYlGn-ish ramp: red (stopped) -> yellow -> green (fast)
function speedColor(frac) {
  const f = Math.max(0, Math.min(1, frac));
  let r, g, b;
  if (f < 0.5) { const t = f / 0.5; r = 230; g = 60 + 150 * t; b = 50; }
  else { const t = (f - 0.5) / 0.5; r = 230 - 180 * t; g = 210 - 30 * t; b = 60 + 20 * t; }
  return [r | 0, g | 0, b | 0];
}

function fail(html) {
  const el = document.getElementById("err");
  el.hidden = false;
  el.innerHTML = `<div class="box">${html}</div>`;
}

let DATA = null, NF = 0, frame = 0, playing = true, view = "both";
let showTrails = true, showRoads = true;
let deckgl = null, trips = { trained: [], untrained: [] };

async function boot() {
  if (typeof deck === "undefined") {
    return fail(`<h2>deck.gl failed to load</h2><p>The CDN script didn't load —
      check your network / the <code>&lt;script&gt;</code> tag in index.html.</p>`);
  }
  try {
    DATA = await (await fetch("public/trajectories.json")).json();
  } catch (e) {
    return fail(`<h2>No trajectory data</h2><p>Could not load
      <code>public/trajectories.json</code>. Generate it first:</p>
      <p><code>python -m smoothride.demo.export_web</code></p>
      <p>then serve this folder: <code>python -m http.server</code> and open
      <code>localhost:8000/smoothride/demo/web/</code>.</p>`);
  }
  try {
    NF = DATA.meta.n_steps;
    document.getElementById("scrub").max = NF - 1;
    buildTrips();
    initDeck();
    wireControls();
    render();                 // draw frame 0 immediately (don't wait for play)
    requestAnimationFrame(tick);
  } catch (e) {
    fail(`<h2>Viewer crashed during setup</h2>
      <p><code>${(e && e.message) || e}</code></p>
      <pre style="text-align:left;white-space:pre-wrap;font-size:11px;opacity:.7">${
        (e && e.stack) || ""}</pre>
      <p>Send me this message and I'll fix it.</p>`);
    console.error(e);
  }
}

// Precompute trip polylines per world, split at respawn teleports.
function buildTrips() {
  for (const w of ["trained", "untrained"]) {
    const out = [];
    for (const car of DATA.worlds[w].cars) {
      let path = [], times = [];
      const flush = () => { if (path.length > 1) out.push({ path, times }); path = []; times = []; };
      for (let t = 0; t < NF; t++) {
        const p = [car.lng[t], car.lat[t]];
        if (path.length) {
          const dx = p[0] - path[path.length - 1][0];
          const dy = p[1] - path[path.length - 1][1];
          if (Math.hypot(dx, dy) > JUMP_DEG) flush();
        }
        path.push(p); times.push(t);
      }
      flush();
    }
    trips[w] = out;
  }
}

function carData(world, t) {
  // one record per car at frame t (skip fully-padded cars that never move)
  return DATA.worlds[world].cars.map((c, i) => ({
    position: [c.lng[t], c.lat[t]],
    heading: c.hdg[t],
    speed: c.spd[t],
    crashed: !!c.crash[t],
    id: i,
  }));
}

function carLayers() {
  const layers = [];
  const worlds = view === "both" ? ["untrained", "trained"] : [view];
  for (const w of worlds) {
    const st = WORLD_STYLE[w];
    const common = {
      id: `cars-${w}`,
      data: carData(w, frame),
      getPosition: d => d.position,
      getColor: d => {
        const c = d.crashed ? [239, 68, 68] : speedColor(d.speed / DATA.meta.vmax);
        return [...c, Math.round(255 * st.opacity)];
      },
      getOrientation: d => [0, 0,
        HEADING_SIGN * (d.heading * 180 / Math.PI) + HEADING_OFFSET_DEG],
      sizeScale: CAR_SIZE,
      pickable: false,
      material: { ambient: 0.5, diffuse: 0.6, shininess: 32, specularColor: [255, 255, 255] },
      parameters: { depthTest: true },
      updateTriggers: { getColor: frame, getOrientation: frame },
    };
    if (CAR_GLB) {
      layers.push(new deck.ScenegraphLayer({ ...common, scenegraph: CAR_GLB,
        _animations: { "*": { speed: 1 } }, getTranslation: [0, 0, 0] }));
    } else {
      layers.push(new deck.SimpleMeshLayer({ ...common, mesh: SmoothRideMesh.car }));
    }
  }
  return layers;
}

function trailLayers() {
  if (!showTrails) return [];
  const worlds = view === "both" ? ["untrained", "trained"] : [view];
  return worlds.map(w => new deck.TripsLayer({
    id: `trips-${w}`,
    data: trips[w],
    getPath: d => d.path,
    getTimestamps: d => d.times,
    getColor: w === "trained" ? [52, 211, 153] : [245, 158, 11],
    opacity: w === "trained" ? 0.8 : 0.25,
    widthMinPixels: w === "trained" ? 2.5 : 1.5,
    jointRounded: true,
    capRounded: true,
    trailLength: TRAIL_LEN,
    currentTime: frame,
  }));
}

function roadLayer() {
  if (!showRoads) return [];
  return [new deck.PathLayer({
    id: "roads",
    data: DATA.roads,
    getPath: d => d,
    getColor: [120, 134, 158, 90],
    getWidth: 1.5,
    widthMinPixels: 1,
    capRounded: true,
  })];
}

function layers() {
  return [...roadLayer(), ...trailLayers(), ...carLayers()];
}

function initDeck() {
  const [lng, lat] = DATA.meta.center;
  const viewState = { longitude: lng, latitude: lat, zoom: DATA.meta.zoom,
    pitch: 52, bearing: -18 };

  if (TOKEN) {
    mapboxgl.accessToken = TOKEN;
    const map = new mapboxgl.Map({
      container: "map",
      style: "mapbox://styles/mapbox/dark-v11",
      center: [lng, lat], zoom: DATA.meta.zoom, pitch: 52, bearing: -18,
      antialias: true,
    });
    const overlay = new deck.MapboxOverlay({ interleaved: true, layers: layers() });
    map.on("load", () => map.addControl(overlay));
    deckgl = overlay;
  } else {
    // graceful fallback: no basemap tiles, roads give the SF context
    document.getElementById("map").style.background = "#0b0e13";
    const note = document.createElement("div");
    note.id = "title";
    note.style.bottom = "70px"; note.style.top = "auto"; note.style.opacity = ".7";
    note.innerHTML = `<div class="sub">⚠ No Mapbox token — running without the
      satellite basemap. Add one in <code>config.js</code> for the real SF map.</div>`;
    document.body.appendChild(note);
    deckgl = new deck.Deck({
      canvas: "deck-canvas",
      initialViewState: viewState,
      controller: true,
      layers: layers(),
    });
  }
}

function updateHUD() {
  for (const [pfx, w] of [["t", "trained"], ["u", "untrained"]]) {
    const cars = DATA.worlds[w].cars;
    let moving = 0, jam = 0, crash = 0;
    for (const c of cars) {
      if (c.crash[frame]) { crash++; continue; }
      if (c.spd[frame] > 1.0) moving++; else jam++;
    }
    document.getElementById(`${pfx}-moving`).textContent = moving;
    document.getElementById(`${pfx}-jam`).textContent = jam;
    document.getElementById(`${pfx}-crash`).textContent = crash;
    document.getElementById(`${pfx}-trips`).textContent =
      DATA.worlds[w].trips_series[frame];
  }
  document.getElementById("clock").textContent =
    `t=${(frame * DATA.meta.dt).toFixed(1)}s`;
  document.getElementById("scrub").value = frame;
}

function render() {
  deckgl.setProps({ layers: layers() });
  updateHUD();
}

let last = 0;
function tick(ts) {
  if (playing && ts - last > 60) {   // ~16 fps playback
    frame = (frame + 1) % NF;
    last = ts;
    render();
  }
  requestAnimationFrame(tick);
}

function wireControls() {
  const playBtn = document.getElementById("play");
  playBtn.onclick = () => {
    playing = !playing;
    playBtn.textContent = playing ? "⏸ Pause" : "▶ Play";
  };
  document.getElementById("scrub").oninput = e => {
    frame = +e.target.value; playing = false; playBtn.textContent = "▶ Play"; render();
  };
  document.getElementById("view").onchange = e => { view = e.target.value; render(); };
  document.getElementById("trails").onchange = e => { showTrails = e.target.checked; render(); };
  document.getElementById("roads").onchange = e => { showRoads = e.target.checked; render(); };
}

boot();
