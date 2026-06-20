// Procedural WheeledLab-style RC car mesh for deck.gl SimpleMeshLayer.
//
// WheeledLab's platforms are ~1/10-scale RC cars (MuSHR/HOUND): a low flat
// chassis with four big exposed wheels. We build that silhouette out of boxes,
// authored in METERS, pointing +X (forward), Z up, centered at the origin — so
// SimpleMeshLayer can place it at a lng/lat and rotate it by heading.
//
// Swap in a real asset later: drop a .glb in models/ and use a ScenegraphLayer
// instead of SimpleMeshLayer (see app.js — CAR_GLB).
(function (global) {
  // one axis-aligned box -> {pos[72], nrm[72], idx[36]} (4 verts/face, 6 faces)
  function box(cx, cy, cz, sx, sy, sz, baseIndex) {
    const x = sx / 2, y = sy / 2, z = sz / 2;
    // 6 faces: +X,-X,+Y,-Y,+Z,-Z ; corners CCW
    const faces = [
      { n: [1, 0, 0], c: [[x, -y, -z], [x, y, -z], [x, y, z], [x, -y, z]] },
      { n: [-1, 0, 0], c: [[-x, y, -z], [-x, -y, -z], [-x, -y, z], [-x, y, z]] },
      { n: [0, 1, 0], c: [[x, y, -z], [-x, y, -z], [-x, y, z], [x, y, z]] },
      { n: [0, -1, 0], c: [[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z]] },
      { n: [0, 0, 1], c: [[x, -y, z], [x, y, z], [-x, y, z], [-x, -y, z]] },
      { n: [0, 0, -1], c: [[-x, -y, -z], [-x, y, -z], [x, y, -z], [x, -y, -z]] },
    ];
    const pos = [], nrm = [], idx = [];
    let v = baseIndex;
    for (const f of faces) {
      for (const corner of f.c) {
        pos.push(corner[0] + cx, corner[1] + cy, corner[2] + cz);
        nrm.push(f.n[0], f.n[1], f.n[2]);
      }
      idx.push(v, v + 1, v + 2, v, v + 2, v + 3);
      v += 4;
    }
    return { pos, nrm, idx };
  }

  function makeCarMesh() {
    // dimensions tuned to read as a real-scale car on the SF grid (footprint
    // matches the sim's CAR_L=4.6 / CAR_W=2.0) but styled RC-buggy.
    const parts = [];
    parts.push([0.0, 0.0, 0.45, 4.2, 1.8, 0.5]);   // chassis slab
    parts.push([-0.3, 0.0, 0.95, 2.2, 1.5, 0.6]);  // cabin (toward rear)
    const wx = 1.45, wy = 0.95, wr = 0.45;          // wheel placement / size
    for (const sx of [wx, -wx]) for (const sy of [wy, -wy]) {
      parts.push([sx, sy, 0.42, 0.9, 0.45, wr * 2]); // chunky exposed wheel
    }

    const pos = [], nrm = [], idx = [];
    let base = 0;
    for (const p of parts) {
      const b = box(p[0], p[1], p[2], p[3], p[4], p[5], base);
      pos.push(...b.pos); nrm.push(...b.nrm); idx.push(...b.idx);
      base += b.pos.length / 3;
    }
    return { pos, nrm, idx };
  }

  // A small upright box for pedestrians.
  function makePedMesh() {
    return box(0, 0, 0.9, 0.6, 0.6, 1.8, 0);  // {pos, nrm, idx}
  }

  // deck.gl 9 SimpleMeshLayer wants attributes as {value, size} descriptors and
  // reads positions via BOTH `POSITION` and `positions` depending on path, so we
  // provide both keys. Without the {value,size} wrapper it crashes in
  // getMeshBoundingBox (reading .length of an undefined `.value`).
  function toMesh(m) {
    const positions = new Float32Array(m.pos);
    const normals = new Float32Array(m.nrm);
    const indices = new Uint32Array(m.idx);
    const P = { value: positions, size: 3 };
    const N = { value: normals, size: 3 };
    return {
      attributes: { POSITION: P, NORMAL: N, positions: P, normals: N },
      indices: { value: indices, size: 1 },
    };
  }

  global.SmoothRideMesh = {
    car: toMesh(makeCarMesh()),
    ped: toMesh(makePedMesh()),
  };
})(window);
