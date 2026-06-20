# Car models

The viewer ships with a **procedural** WheeledLab-style RC-car mesh built in
`../carmesh.js` (a low chassis + four chunky exposed wheels), so it works with
no external assets.

To use a real 3D car asset instead:

1. Drop a glTF/GLB file here, e.g. `wheeledlab_car.glb`.
2. In `../app.js` set `CAR_GLB = "models/wheeledlab_car.glb"`.
   The viewer then renders it with deck.gl's `ScenegraphLayer` instead of the
   procedural `SimpleMeshLayer`.

Notes:
- Author the model pointing **+X (forward)**, Z up, in **meters**. If it faces
  the wrong way, adjust `HEADING_OFFSET_DEG` / `HEADING_SIGN` in `app.js`.
- WheeledLab's own car assets (MuSHR / HOUND) live in its USD/Isaac assets; a GLB
  export of one of those is the most "on-brand" choice if you have it.
