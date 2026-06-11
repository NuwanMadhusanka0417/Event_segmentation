// Event-camera 3D directed-graph visualizer.
//
// Streams events forward in virtual time. For each new event we look back over
// the "active window" of recently-added events (those whose z-delta could
// possibly be within max distance), brute-force the Euclidean distance, take
// the k nearest, and draw directed edges existing -> new (color gradient
// dim -> bright shows direction without per-edge arrowheads).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const SPACE_SCALE = 100;          // target half-width of the scene in world units
const HARD_MAX_EDGES = 2_000_000; // cap to keep GPU memory in check

const COLOR_SOURCE = new THREE.Color(0x14283c); // dim end of each edge
const COLOR_POS_TARGET = new THREE.Color(0x3d8cc4); // bright end (positive polarity)
const COLOR_NEG_TARGET = new THREE.Color(0xc04a3d); // bright end (negative polarity)
const COLOR_POS_NODE = new THREE.Color(0x4f9bce);
const COLOR_NEG_NODE = new THREE.Color(0xb85a4e);

// Shaded-sphere impostor material for nodes. Each point sprite is treated as
// a unit disk; the fragment shader rebuilds the sphere normal from the sprite
// UV and runs Lambert + specular, giving true 3D-ball shading without the
// cost of instancing thousands of real sphere meshes.
const NODE_VERTEX_SHADER = /* glsl */ `
  attribute vec3 color;
  uniform float uSize;
  varying vec3 vColor;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize * (300.0 / -mv.z);
    gl_Position = projectionMatrix * mv;
  }
`;
const NODE_FRAGMENT_SHADER = /* glsl */ `
  precision mediump float;
  varying vec3 vColor;
  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;
    float z = sqrt(1.0 - r2);
    vec3 n = vec3(uv.x, -uv.y, z); // flip y to match screen
    vec3 L = normalize(vec3(0.45, 0.65, 0.85));
    float diff = max(dot(n, L), 0.0);
    vec3 R = reflect(-L, n);
    float spec = pow(max(R.z, 0.0), 28.0);
    vec3 col = vColor * (0.28 + 0.75 * diff) + vec3(spec * 0.55);
    gl_FragColor = vec4(col, 1.0);
  }
`;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  payload: null,        // raw event payload from the server
  positions: null,      // Float32Array length 3N — normalized scene coords (x, y, z)
  polarities: null,     // Int8Array length N
  N: 0,
  virtualTime: 0,       // in normalized z-units (same units as positions[i*3+2])
  vtMax: 0,
  nextIdx: 0,           // index of next event to inject
  running: false,
  // Parameters (kept in sync with sliders).
  k: 8,
  maxDist: 12,
  speed: 1,
  timeScale: 1,
  // FPS bookkeeping.
  lastFrame: 0,
  fpsAcc: 0,
  fpsFrames: 0,
};

// ---------------------------------------------------------------------------
// Three.js scene
// ---------------------------------------------------------------------------

const viewport = document.getElementById('viewport');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07090d);
scene.fog = new THREE.Fog(0x07090d, 350, 900);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 5000);
camera.position.set(160, 140, 220);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.85;

// Scene group — apply user-controlled time scale to the z axis only.
const graphGroup = new THREE.Group();
scene.add(graphGroup);

// Reference axes + grid in their own group so they don't get z-scaled.
const refs = new THREE.Group();
scene.add(refs);

(function buildReferences() {
  const grid = new THREE.GridHelper(SPACE_SCALE * 2, 20, 0x223040, 0x121820);
  grid.position.y = -SPACE_SCALE / 2;
  refs.add(grid);

  const axes = new THREE.AxesHelper(SPACE_SCALE * 0.6);
  axes.position.set(-SPACE_SCALE, -SPACE_SCALE / 2, -SPACE_SCALE);
  refs.add(axes);
})();

// Preallocated geometries — sized lazily when a file is loaded.
let nodesGeom = null;
let nodesPoints = null;
let edgesGeom = null;
let edgesLines = null;

function buildBuffers(N) {
  if (nodesPoints) graphGroup.remove(nodesPoints);
  if (edgesLines) graphGroup.remove(edgesLines);

  const maxEdges = Math.min(N * 32, HARD_MAX_EDGES);

  nodesGeom = new THREE.BufferGeometry();
  nodesGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  nodesGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  nodesGeom.setDrawRange(0, 0);
  const nodeMat = new THREE.ShaderMaterial({
    uniforms: { uSize: { value: 1.1 } },
    vertexShader: NODE_VERTEX_SHADER,
    fragmentShader: NODE_FRAGMENT_SHADER,
    transparent: false,
    depthWrite: true,
  });
  nodesPoints = new THREE.Points(nodesGeom, nodeMat);
  nodesPoints.frustumCulled = false;
  graphGroup.add(nodesPoints);

  edgesGeom = new THREE.BufferGeometry();
  edgesGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxEdges * 6), 3));
  edgesGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(maxEdges * 6), 3));
  edgesGeom.setDrawRange(0, 0);
  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.65,
  });
  edgesLines = new THREE.LineSegments(edgesGeom, edgeMat);
  edgesLines.frustumCulled = false;
  graphGroup.add(edgesLines);

  state.nodeCount = 0;     // number of inserted nodes so far
  state.edgeCount = 0;     // number of inserted edges so far
  state.edgeCap = maxEdges;
}

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------------------
// Event loading + normalization
// ---------------------------------------------------------------------------

async function fetchFiles() {
  const r = await fetch('/api/files');
  const j = await r.json();
  const sel = document.getElementById('file-select');
  sel.innerHTML = '';
  if (!j.files.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '(no files in events/ — upload one)';
    sel.appendChild(opt);
    return;
  }
  for (const name of j.files) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  }
}

async function loadEvents(name) {
  msg(`Loading ${name}…`, 'info');
  const r = await fetch(`/api/events?name=${encodeURIComponent(name)}`);
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.error || `Failed to load ${name}`);
  }
  const payload = await r.json();
  prepareStream(payload);
  msg(
    `Loaded ${payload.count.toLocaleString()} events` +
    (payload.downsampled_from ? ` (downsampled from ${payload.downsampled_from.toLocaleString()})` : ''),
    'info'
  );
}

function prepareStream(payload) {
  state.payload = payload;
  const N = payload.count;
  state.N = N;

  // Normalize so that the largest spatial extent maps to roughly 2 * SPACE_SCALE
  // in scene units; time gets the SAME scale so distance/k are in comparable units.
  const xMid = (payload.x_min + payload.x_max) / 2;
  const yMid = (payload.y_min + payload.y_max) / 2;
  const xRange = Math.max(payload.x_max - payload.x_min, 1e-6);
  const yRange = Math.max(payload.y_max - payload.y_min, 1e-6);
  const tRange = Math.max(payload.t_max - payload.t_min, 1e-6);

  const spatialSpan = Math.max(xRange, yRange);
  const s = SPACE_SCALE / spatialSpan; // unit conversion

  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    pos[i * 3 + 0] = (payload.x[i] - xMid) * s;
    pos[i * 3 + 1] = (payload.y[i] - yMid) * s;
    // Map t to z using the same per-unit scale, but normalize range so that
    // the full timeline starts at z=0 and ends at z = SPACE_SCALE * 2.
    pos[i * 3 + 2] = (payload.t[i] / tRange) * (SPACE_SCALE * 2);
  }

  state.positions = pos;
  state.polarities = Int8Array.from(payload.p);
  state.vtMax = (SPACE_SCALE * 2);
  state.virtualTime = 0;
  state.nextIdx = 0;

  buildBuffers(N);

  // Frame camera on the full extent.
  const box = new THREE.Box3(
    new THREE.Vector3(-SPACE_SCALE, -SPACE_SCALE / 2, 0),
    new THREE.Vector3(SPACE_SCALE, SPACE_SCALE / 2, SPACE_SCALE * 2)
  );
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.set(center.x + 220, center.y + 160, center.z + 320);
  controls.update();

  updateStats();
}

// ---------------------------------------------------------------------------
// Graph construction (called once per new event)
// ---------------------------------------------------------------------------

function addEvent(i) {
  const pos = state.positions;
  const N = state.N;
  const D = state.maxDist;
  const D2 = D * D;
  const k = state.k;

  const x = pos[i * 3 + 0];
  const y = pos[i * 3 + 1];
  const z = pos[i * 3 + 2];

  // ---- find candidate neighbours ----
  // Events are time-sorted, so we walk backwards from i-1 and stop as soon as
  // the z gap exceeds D. Within that window we brute-force the Euclidean check.
  const cand = []; // [distSq, idx]
  for (let j = i - 1; j >= 0; j--) {
    const dz = z - pos[j * 3 + 2];
    if (dz > D) break;
    const dx = x - pos[j * 3 + 0];
    const dy = y - pos[j * 3 + 1];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= D2) {
      cand.push([d2, j]);
    }
  }
  cand.sort((a, b) => a[0] - b[0]);
  const take = Math.min(k, cand.length);

  // ---- write the node ----
  const np = nodesGeom.attributes.position.array;
  const nc = nodesGeom.attributes.color.array;
  const ni = state.nodeCount;
  np[ni * 3] = x; np[ni * 3 + 1] = y; np[ni * 3 + 2] = z;
  const polCol = state.polarities[i] > 0 ? COLOR_POS_NODE : COLOR_NEG_NODE;
  nc[ni * 3] = polCol.r; nc[ni * 3 + 1] = polCol.g; nc[ni * 3 + 2] = polCol.b;
  state.nodeCount += 1;
  nodesGeom.setDrawRange(0, state.nodeCount);
  nodesGeom.attributes.position.needsUpdate = true;
  nodesGeom.attributes.color.needsUpdate = true;

  // ---- write the directed edges (existing -> new) ----
  const ep = edgesGeom.attributes.position.array;
  const ec = edgesGeom.attributes.color.array;
  const targetCol = state.polarities[i] > 0 ? COLOR_POS_TARGET : COLOR_NEG_TARGET;

  for (let t = 0; t < take; t++) {
    if (state.edgeCount >= state.edgeCap) break;
    const j = cand[t][1];
    const base = state.edgeCount * 6;
    // Source vertex (existing neighbour) — dim end.
    ep[base + 0] = pos[j * 3 + 0];
    ep[base + 1] = pos[j * 3 + 1];
    ep[base + 2] = pos[j * 3 + 2];
    ec[base + 0] = COLOR_SOURCE.r;
    ec[base + 1] = COLOR_SOURCE.g;
    ec[base + 2] = COLOR_SOURCE.b;
    // Target vertex (new event) — bright end.
    ep[base + 3] = x;
    ep[base + 4] = y;
    ep[base + 5] = z;
    ec[base + 3] = targetCol.r;
    ec[base + 4] = targetCol.g;
    ec[base + 5] = targetCol.b;
    state.edgeCount += 1;
  }
  edgesGeom.setDrawRange(0, state.edgeCount * 2);
  edgesGeom.attributes.position.needsUpdate = true;
  edgesGeom.attributes.color.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// Animation loop
// ---------------------------------------------------------------------------

function step(now) {
  const dtMs = state.lastFrame ? now - state.lastFrame : 16;
  state.lastFrame = now;

  if (state.running && state.payload) {
    // Advance virtual time. We chose vtMax to span the whole stream in
    // 1 / speed real-time seconds when speed = 1, so a speed of 1.0 means
    // ~10 s for the full stream at 60fps target. The slider then multiplies.
    const baseRate = state.vtMax / 10_000; // units per ms
    state.virtualTime += dtMs * baseRate * state.speed;
    if (state.virtualTime > state.vtMax) state.virtualTime = state.vtMax;

    const pos = state.positions;
    let added = 0;
    while (state.nextIdx < state.N) {
      const z = pos[state.nextIdx * 3 + 2];
      if (z > state.virtualTime) break;
      addEvent(state.nextIdx);
      state.nextIdx += 1;
      added += 1;
      // Avoid stalling the frame on big jumps; cap inserts per frame.
      if (added > 8000) break;
    }

    if (state.nextIdx >= state.N && state.virtualTime >= state.vtMax) {
      setRunning(false);
    }
    updateStats();
  }

  // FPS tracker.
  state.fpsAcc += dtMs;
  state.fpsFrames += 1;
  if (state.fpsAcc >= 500) {
    document.getElementById('stat-fps').textContent =
      Math.round((state.fpsFrames * 1000) / state.fpsAcc);
    state.fpsAcc = 0;
    state.fpsFrames = 0;
  }

  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(step);
}
requestAnimationFrame(step);

// ---------------------------------------------------------------------------
// UI plumbing
// ---------------------------------------------------------------------------

function $(id) { return document.getElementById(id); }

function msg(text, kind = 'err') {
  const el = $('msg');
  el.textContent = text || '';
  el.className = 'row msg ' + (kind === 'info' ? 'info' : '');
}

function updateStats() {
  $('stat-events').textContent = `${state.nextIdx.toLocaleString()} / ${state.N.toLocaleString()}`;
  $('stat-nodes').textContent = state.nodeCount.toLocaleString();
  $('stat-edges').textContent = state.edgeCount.toLocaleString();
  $('stat-time').textContent = state.vtMax
    ? (state.virtualTime / state.vtMax).toFixed(3)
    : '0.000';
}

function setRunning(on) {
  state.running = on;
  $('run').disabled = on || !state.payload;
  $('pause').disabled = !on;
}

function bindSlider(id, valId, key, fmt = (v) => v.toFixed(0)) {
  const slider = $(id);
  const label = $(valId);
  const apply = () => {
    const v = parseFloat(slider.value);
    state[key] = v;
    label.textContent = fmt(v);
    if (key === 'timeScale') {
      graphGroup.scale.z = v;
    }
  };
  slider.addEventListener('input', apply);
  apply();
}

bindSlider('k', 'k-val', 'k');
bindSlider('d', 'd-val', 'maxDist');
bindSlider('sp', 'sp-val', 'speed', (v) => v.toFixed(2));
bindSlider('ts', 'ts-val', 'timeScale', (v) => v.toFixed(2));

$('refresh').addEventListener('click', fetchFiles);

$('file-select').addEventListener('change', async (e) => {
  const name = e.target.value;
  if (!name) return;
  try { await loadEvents(name); } catch (err) { msg(err.message); }
});

$('file-upload').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  msg(`Uploading ${file.name}…`, 'info');
  const fd = new FormData();
  fd.append('file', file);
  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  const j = await r.json();
  if (!r.ok) { msg(j.error || 'Upload failed'); return; }
  await fetchFiles();
  $('file-select').value = j.name;
  await loadEvents(j.name).catch((err) => msg(err.message));
});

$('run').addEventListener('click', async () => {
  const sel = $('file-select').value;
  if (!sel) { msg('Select an event file first.'); return; }
  if (!state.payload || state.payload._name !== sel) {
    try {
      await loadEvents(sel);
      if (state.payload) state.payload._name = sel;
    } catch (e) { msg(e.message); return; }
  }
  setRunning(true);
});

$('pause').addEventListener('click', () => setRunning(false));

$('reset').addEventListener('click', () => {
  setRunning(false);
  if (state.payload) {
    state.virtualTime = 0;
    state.nextIdx = 0;
    state.nodeCount = 0;
    state.edgeCount = 0;
    if (nodesGeom) { nodesGeom.setDrawRange(0, 0); nodesGeom.attributes.position.needsUpdate = true; }
    if (edgesGeom) { edgesGeom.setDrawRange(0, 0); edgesGeom.attributes.position.needsUpdate = true; }
    updateStats();
  }
});

$('fullscreen').addEventListener('click', () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
});

$('collapse').addEventListener('click', () => {
  $('hud').classList.add('collapsed');
  $('show-hud').hidden = false;
});
$('show-hud').addEventListener('click', () => {
  $('hud').classList.remove('collapsed');
  $('show-hud').hidden = true;
});

// Kick things off.
fetchFiles();
