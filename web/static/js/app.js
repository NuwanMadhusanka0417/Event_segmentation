// Event-camera visualizer: 3D graph + 2D event video + space-time 3D video.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  VIDEO_WINDOW_FRAC_DEFAULT,
  EVENT_VIDEO_DOT_U_SIZE,
  EVENT_VIDEO_DOT_DEPTH_REF,
  getEventVideoDisplayScale,
  eventVideoDotPx,
} from './video-common.js';

const SPACE_SCALE = 100;
const ST_SCALE = 100;
const HARD_MAX_EDGES = 2_000_000;
const ST_WINDOW_FRAC_DEFAULT = 0.02;
const IMG_POS = [255, 0, 0];
const IMG_NEG = [0, 0, 255];
const ST_DOT_U_SIZE = EVENT_VIDEO_DOT_U_SIZE;
const ST_DOT_DEPTH_REF = EVENT_VIDEO_DOT_DEPTH_REF;

const COLOR_SOURCE = new THREE.Color(0x14283c);
const COLOR_POS_TARGET = new THREE.Color(0x3d8cc4);
const COLOR_NEG_TARGET = new THREE.Color(0xc04a3d);
const COLOR_POS_NODE = new THREE.Color(0x4f9bce);
const COLOR_NEG_NODE = new THREE.Color(0xb85a4e);
const ST_COLOR_POS = new THREE.Color(1, 0, 0);
const ST_COLOR_NEG = new THREE.Color(0, 0, 1);

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
    vec3 n = vec3(uv.x, -uv.y, z);
    vec3 L = normalize(vec3(0.45, 0.65, 0.85));
    float diff = max(dot(n, L), 0.0);
    vec3 R = reflect(-L, n);
    float spec = pow(max(R.z, 0.0), 28.0);
    vec3 col = vColor * (0.28 + 0.75 * diff) + vec3(spec * 0.55);
    gl_FragColor = vec4(col, 1.0);
  }
`;
const ST_FRAGMENT_SHADER = /* glsl */ `
  precision mediump float;
  varying vec3 vColor;
  void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    if (dot(uv, uv) > 1.0) discard;
    gl_FragColor = vec4(vColor, 1.0);
  }
`;

const state = {
  payload: null,
  positions: null,
  polarities: null,
  N: 0,
  virtualTime: 0,
  vtMax: 0,
  nextIdx: 0,
  running: false,
  k: 8,
  maxDist: 12,
  speed: 1,
  timeScale: 1,
  viewMode: 'graph',
  imgW: 0,
  imgH: 0,
  xOff: 0,
  yOff: 0,
  videoCtx: null,
  videoData: null,
  videoDispW: 0,
  videoDispH: 0,
  videoDispScale: 1,
  videoWindowFrac: VIDEO_WINDOW_FRAC_DEFAULT,
  videoWindowSize: 0,
  videoTime: 0,
  videoSpan: 0,
  videoTMin: 0,
  videoScrubbing: false,
  stWindowFrac: ST_WINDOW_FRAC_DEFAULT,
  stWindowSize: 0,
  stTime: 0,
  stSpan: 0,
  stTMin: 0,
  stScrubbing: false,
  stExtents: { lx: ST_SCALE, ly: ST_SCALE, lt: ST_SCALE },
  lastFrame: 0,
  fpsAcc: 0,
  fpsFrames: 0,
};

const viewport = document.getElementById('viewport');
const videoViewport = document.getElementById('video-viewport');
const videoCanvas = document.getElementById('video-canvas');
const videoControlBar = document.getElementById('video-control-bar');
const videoWindowSlider = document.getElementById('vw');
const videoWindowVal = document.getElementById('vw-val');
const videoScrub = document.getElementById('video-scrub');
const videoProgress = document.getElementById('video-progress');
const videoTimeRange = document.getElementById('video-time-range');
const stViewport = document.getElementById('st-viewport');
const stStage = document.getElementById('st-stage');
const stControlBar = document.getElementById('st-control-bar');
const stWindowSlider = document.getElementById('st-vw');
const stWindowVal = document.getElementById('st-vw-val');
const stScrub = document.getElementById('st-scrub');
const stProgress = document.getElementById('st-progress');
const stTimeRange = document.getElementById('st-time-range');

const VIZ_PANE_IDS = { graph: 'viewport', video: 'video-viewport', st3d: 'st-viewport', seg: 'seg-viewport' };

// ---- Graph 3D scene ----
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

const graphGroup = new THREE.Group();
scene.add(graphGroup);

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

let nodesGeom = null;
let nodesPoints = null;
let edgesGeom = null;
let edgesLines = null;

// ---- Space-time 3D scene ----
const stScene = new THREE.Scene();
stScene.background = new THREE.Color(0xe8e8e8);

const stCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 8000);
const stRenderer = new THREE.WebGLRenderer({ antialias: true });
stRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
stStage.appendChild(stRenderer.domElement);

const stControls = new OrbitControls(stCamera, stRenderer.domElement);
stControls.enableDamping = true;
stControls.dampingFactor = 0.08;
stControls.rotateSpeed = 0.85;

const stRoot = new THREE.Group();
stScene.add(stRoot);

let stGeom = null;
let stPoints = null;
let stAxesGroup = null;

function buildBuffers(N) {
  if (nodesPoints) graphGroup.remove(nodesPoints);
  if (edgesLines) graphGroup.remove(edgesLines);

  const maxEdges = Math.min(N * 32, HARD_MAX_EDGES);

  nodesGeom = new THREE.BufferGeometry();
  nodesGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  nodesGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(N * 3), 3));
  nodesGeom.setDrawRange(0, 0);
  nodesPoints = new THREE.Points(nodesGeom, new THREE.ShaderMaterial({
    uniforms: { uSize: { value: 1.1 } },
    vertexShader: NODE_VERTEX_SHADER,
    fragmentShader: NODE_FRAGMENT_SHADER,
    transparent: false,
    depthWrite: true,
  }));
  nodesPoints.frustumCulled = false;
  graphGroup.add(nodesPoints);

  edgesGeom = new THREE.BufferGeometry();
  edgesGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxEdges * 6), 3));
  edgesGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(maxEdges * 6), 3));
  edgesGeom.setDrawRange(0, 0);
  edgesLines = new THREE.LineSegments(edgesGeom, new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.65,
  }));
  edgesLines.frustumCulled = false;
  graphGroup.add(edgesLines);

  state.nodeCount = 0;
  state.edgeCount = 0;
  state.edgeCap = maxEdges;
}

function resizeGraph() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

function getVideoDisplayScale() {
  return getEventVideoDisplayScale(state.imgW, state.imgH, videoControlBar.hidden);
}

function videoDotPx() {
  return eventVideoDotPx();
}

function resizeVideoDisplay() {
  if (!state.imgW || !state.imgH || !videoViewport.classList.contains('has-video')) return;
  const scale = getVideoDisplayScale();
  const dispW = Math.max(1, Math.round(state.imgW * scale));
  const dispH = Math.max(1, Math.round(state.imgH * scale));
  state.videoDispScale = scale;
  state.videoDispW = dispW;
  state.videoDispH = dispH;
  videoCanvas.width = dispW;
  videoCanvas.height = dispH;
  videoCanvas.style.width = `${dispW}px`;
  videoCanvas.style.height = `${dispH}px`;
  if (state.videoCtx) {
    state.videoData = state.videoCtx.createImageData(dispW, dispH);
  }
}

function resizeSpaceTime() {
  const barH = stControlBar.hidden ? 0 : 190;
  const w = window.innerWidth;
  const h = Math.max(200, window.innerHeight - barH);
  stCamera.aspect = w / h;
  stCamera.updateProjectionMatrix();
  stRenderer.setSize(w, h);
}

function resize() {
  resizeGraph();
  if (state.viewMode === 'st3d') resizeSpaceTime();
  else if (state.viewMode === 'video') {
    resizeVideoDisplay();
    if (state.payload) renderVideoFrame();
  }
}

window.addEventListener('resize', resize);
resize();

async function fetchFiles() {
  const r = await fetch('/api/files');
  const j = await r.json();
  const sel = document.getElementById('file-select');
  sel.innerHTML = '';
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = j.files.length ? '(select a file)' : '(no files in events/ — upload one)';
  sel.appendChild(blank);
  for (const name of j.files) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  }
}

function clearLoadedFile() {
  state.payload = null;
  state.positions = null;
  state.polarities = null;
  state.N = 0;
  state.virtualTime = 0;
  state.vtMax = 0;
  state.nextIdx = 0;
  state.nodeCount = 0;
  state.edgeCount = 0;
  state.stTime = 0;
  state.stSpan = 0;
  state.stWindowSize = 0;
  state.stScrubbing = false;
  state.videoCtx = null;
  state.videoData = null;
  state.videoDispW = 0;
  state.videoDispH = 0;
  state.videoDispScale = 1;
  state.videoTime = 0;
  state.videoSpan = 0;
  state.videoWindowSize = 0;
  state.videoScrubbing = false;
  stViewport.classList.remove('has-data');
  videoViewport.classList.remove('has-video');
  stControlBar.hidden = true;
  videoControlBar.hidden = true;
  if (stPoints) {
    stRoot.remove(stPoints);
    stGeom?.dispose();
    stPoints = null;
    stGeom = null;
  }
  if (stAxesGroup) {
    stRoot.remove(stAxesGroup);
    stAxesGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose?.();
    });
    stAxesGroup = null;
  }
  setRunning(false);
  if (nodesGeom) {
    nodesGeom.setDrawRange(0, 0);
    nodesGeom.attributes.position.needsUpdate = true;
  }
  if (edgesGeom) {
    edgesGeom.setDrawRange(0, 0);
    edgesGeom.attributes.position.needsUpdate = true;
  }
  updateStats();
  msg('', 'info');
}

async function loadEvents(name) {
  msg(`Loading ${name}…`, 'info');
  const r = await fetch(`/api/events?name=${encodeURIComponent(name)}`);
  if (!r.ok) {
    const j = await r.json().catch(() => ({}));
    throw new Error(j.error || `Failed to load ${name}`);
  }
  prepareStream(await r.json());
  msg(
    `Loaded ${state.payload.count.toLocaleString()} events` +
    (state.payload.downsampled_from ? ` (downsampled from ${state.payload.downsampled_from.toLocaleString()})` : ''),
    'info'
  );
}

function prepareStream(payload) {
  state.payload = payload;
  const N = payload.count;
  state.N = N;

  const xMid = (payload.x_min + payload.x_max) / 2;
  const yMid = (payload.y_min + payload.y_max) / 2;
  const xRange = Math.max(payload.x_max - payload.x_min, 1e-6);
  const yRange = Math.max(payload.y_max - payload.y_min, 1e-6);
  const tRange = Math.max(payload.t_max - payload.t_min, 1e-6);
  const s = SPACE_SCALE / Math.max(xRange, yRange);

  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    pos[i * 3 + 0] = (payload.x[i] - xMid) * s;
    pos[i * 3 + 1] = (payload.y[i] - yMid) * s;
    pos[i * 3 + 2] = (payload.t[i] / tRange) * (SPACE_SCALE * 2);
  }

  state.positions = pos;
  state.polarities = Int8Array.from(payload.p);
  state.vtMax = SPACE_SCALE * 2;
  state.virtualTime = 0;
  state.nextIdx = 0;

  buildBuffers(N);

  const box = new THREE.Box3(
    new THREE.Vector3(-SPACE_SCALE, -SPACE_SCALE / 2, 0),
    new THREE.Vector3(SPACE_SCALE, SPACE_SCALE / 2, SPACE_SCALE * 2)
  );
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  camera.position.set(center.x + 220, center.y + 160, center.z + 320);
  controls.update();

  prepareSpaceTime(payload);
  prepareVideo(payload);
  updateStats();
}

// ---------------------------------------------------------------------------
// Space-time 3D — x, y, time volume with sliding window
// ---------------------------------------------------------------------------

function lowerBound(arr, value) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function formatTimeUnits(value) {
  if (!Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1e6 || (Math.abs(value) >= 1e3 && Number.isInteger(value))) {
    return `${(value / 1e3).toFixed(2)} ms`;
  }
  if (Math.abs(value) >= 1) return value.toFixed(2);
  return value.toFixed(4);
}

// ---------------------------------------------------------------------------
// Event video — 2D sliding time-window playback
// ---------------------------------------------------------------------------

function fillImageWhite(data) {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = 255;
    data[i + 1] = 255;
    data[i + 2] = 255;
    data[i + 3] = 255;
  }
}

function plotSharpDot(data, dispW, dispH, cx, cy, dotPx, rgb) {
  const half = Math.floor(dotPx / 2);
  for (let dy = 0; dy < dotPx; dy++) {
    for (let dx = 0; dx < dotPx; dx++) {
      const px = cx - half + dx;
      const py = cy - half + dy;
      if (px < 0 || py < 0 || px >= dispW || py >= dispH) continue;
      const i = (py * dispW + px) * 4;
      data[i] = rgb[0];
      data[i + 1] = rgb[1];
      data[i + 2] = rgb[2];
      data[i + 3] = 255;
    }
  }
}

function setImageExtents(payload) {
  const xMin = Math.floor(payload.x_min);
  const xMax = Math.ceil(payload.x_max);
  const yMin = Math.floor(payload.y_min);
  const yMax = Math.ceil(payload.y_max);
  state.xOff = xMin;
  state.yOff = yMin;
  state.imgW = Math.max(1, xMax - xMin + 1);
  state.imgH = Math.max(1, yMax - yMin + 1);
}

function recomputeVideoWindow() {
  if (!state.payload?.t?.length) {
    state.videoSpan = 1;
    state.videoWindowSize = 1;
    return;
  }
  const t = state.payload.t;
  state.videoTMin = t[0];
  state.videoSpan = Math.max(t[t.length - 1] - state.videoTMin, 1e-9);
  state.videoWindowSize = Math.max(state.videoSpan * state.videoWindowFrac, 1e-9);
  const maxStart = Math.max(0, state.videoSpan - state.videoWindowSize);
  if (state.videoTime > maxStart) state.videoTime = maxStart;
}

function updateVideoBarLabels() {
  const absBase = state.payload?.t_min ?? 0;
  const winStart = state.videoTMin + state.videoTime;
  const winEnd = winStart + state.videoWindowSize;
  const maxStart = Math.max(0, state.videoSpan - state.videoWindowSize);
  const pct = maxStart > 0 ? (state.videoTime / maxStart) * 100 : 100;
  videoWindowVal.textContent = `${(state.videoWindowFrac * 100).toFixed(1)}%`;
  videoProgress.textContent = `${pct.toFixed(0)}%`;
  if (!state.videoScrubbing) videoScrub.value = String(Math.round(pct * 10));

  const t = state.payload?.t;
  let nEvents = 0;
  if (t?.length) {
    nEvents = lowerBound(t, winEnd) - lowerBound(t, state.videoTime);
  }
  videoTimeRange.textContent =
    `${formatTimeUnits(absBase + winStart)} – ${formatTimeUnits(absBase + winEnd)} · ${nEvents.toLocaleString()} events`;
}

function syncVideoBarVisibility() {
  videoControlBar.hidden = !(state.viewMode === 'video' && state.payload);
  if (!videoControlBar.hidden) resizeVideoDisplay();
}

function prepareVideo(payload) {
  setImageExtents(payload);
  state.videoCtx = videoCanvas.getContext('2d', { alpha: false });
  state.videoTime = 0;
  state.videoWindowFrac = parseFloat(videoWindowSlider.value) / 100;
  recomputeVideoWindow();
  videoViewport.classList.add('has-video');
  resizeVideoDisplay();
  renderVideoFrame();
  syncVideoBarVisibility();
}

function renderVideoFrame() {
  if (!state.videoCtx || !state.videoData || !state.payload?.t?.length) return;
  const data = state.videoData.data;
  const dispW = state.videoDispW;
  const dispH = state.videoDispH;
  const scale = state.videoDispScale;
  const dotPx = videoDotPx();
  fillImageWhite(data);
  const payload = state.payload;
  const t = payload.t;
  const winEnd = state.videoTime + state.videoWindowSize;
  const start = lowerBound(t, state.videoTime);
  const end = lowerBound(t, winEnd);
  for (let i = start; i < end; i++) {
    const cx = Math.round((Math.round(payload.x[i]) - state.xOff) * scale);
    const cy = Math.round((Math.round(payload.y[i]) - state.yOff) * scale);
    const rgb = payload.p[i] > 0 ? IMG_POS : IMG_NEG;
    plotSharpDot(data, dispW, dispH, cx, cy, dotPx, rgb);
  }
  state.videoCtx.putImageData(state.videoData, 0, 0);
  updateVideoBarLabels();
}

function setVideoTimeFromScrub() {
  const maxStart = Math.max(0, state.videoSpan - state.videoWindowSize);
  state.videoTime = (parseInt(videoScrub.value, 10) / 1000) * maxStart;
  renderVideoFrame();
}

function advanceVideo(dtMs) {
  const maxStart = Math.max(0, state.videoSpan - state.videoWindowSize);
  const rate = state.videoSpan / 10_000;
  state.videoTime += dtMs * rate * state.speed;
  if (state.videoTime >= maxStart) {
    state.videoTime = maxStart;
    setRunning(false);
  }
  renderVideoFrame();
}

function resetVideo() {
  if (!state.payload) return;
  state.videoTime = 0;
  renderVideoFrame();
}

function buildSpaceTimeAxes(lx, ly, lt) {
  const group = new THREE.Group();
  const lineMat = new THREE.LineBasicMaterial({ color: 0x111111 });

  const corner = [
    [0, 0, 0], [lx, 0, 0],
    [0, 0, 0], [0, ly, 0],
    [0, 0, 0], [0, 0, lt],
    [lx, 0, 0], [lx, ly, 0],
    [lx, ly, 0], [lx, ly, lt],
    [0, ly, 0], [0, ly, lt],
    [0, 0, lt], [lx, 0, lt],
    [lx, 0, lt], [lx, ly, lt],
    [0, ly, lt], [0, 0, lt],
  ];
  const verts = new Float32Array(corner.flat());
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  group.add(new THREE.LineSegments(g, lineMat));

  return group;
}

function recomputeStWindow() {
  if (!state.payload?.t?.length) {
    state.stSpan = 1;
    state.stWindowSize = 1;
    return;
  }
  const t = state.payload.t;
  state.stTMin = t[0];
  state.stSpan = Math.max(t[t.length - 1] - state.stTMin, 1e-9);
  state.stWindowSize = Math.max(state.stSpan * state.stWindowFrac, 1e-9);
  const maxStart = Math.max(0, state.stSpan - state.stWindowSize);
  if (state.stTime > maxStart) state.stTime = maxStart;
}

function updateStBarLabels() {
  const absBase = state.payload?.t_min ?? 0;
  const winEnd = state.stTime + state.stWindowSize;
  const maxStart = Math.max(0, state.stSpan - state.stWindowSize);
  const pct = maxStart > 0 ? (state.stTime / maxStart) * 100 : 100;
  stWindowVal.textContent = `${(state.stWindowFrac * 100).toFixed(1)}%`;
  stProgress.textContent = `${pct.toFixed(0)}%`;
  if (!state.stScrubbing) stScrub.value = String(Math.round(pct * 10));

  const t = state.payload?.t;
  let nEvents = 0;
  if (t?.length) {
    nEvents = lowerBound(t, winEnd);
  }
  stTimeRange.textContent =
    `${formatTimeUnits(absBase)} – ${formatTimeUnits(absBase + winEnd)} · ${nEvents.toLocaleString()} events`;
}

function syncStBarVisibility() {
  stControlBar.hidden = !(state.viewMode === 'st3d' && state.payload);
  if (!stControlBar.hidden) resizeSpaceTime();
}

function prepareSpaceTime(payload) {
  if (stPoints) {
    stRoot.remove(stPoints);
    stGeom?.dispose();
  }
  if (stAxesGroup) {
    stRoot.remove(stAxesGroup);
    stAxesGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose?.();
    });
  }

  const N = payload.count;
  const xMin = payload.x_min;
  const xMax = payload.x_max;
  const yMin = payload.y_min;
  const yMax = payload.y_max;
  const t = payload.t;
  const t0 = t[0] || 0;
  const tEnd = t[t.length - 1] || t0;

  const lx = ST_SCALE;
  const ly = ST_SCALE;
  const lt = ST_SCALE;
  const sx = lx / Math.max(xMax - xMin, 1e-6);
  const sy = ly / Math.max(yMax - yMin, 1e-6);
  const st = lt / Math.max(tEnd - t0, 1e-9);
  state.stExtents = { lx, ly, lt };

  const positions = new Float32Array(N * 3);
  const colors = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    positions[i * 3] = (payload.x[i] - xMin) * sx;
    // Flip Y so image rows map with y increasing upward (Three.js convention).
    positions[i * 3 + 1] = (yMax - payload.y[i]) * sy;
    positions[i * 3 + 2] = (payload.t[i] - t0) * st;
    const c = payload.p[i] > 0 ? ST_COLOR_POS : ST_COLOR_NEG;
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }

  stGeom = new THREE.BufferGeometry();
  stGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  stGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  stGeom.setDrawRange(0, 0);

  stPoints = new THREE.Points(stGeom, new THREE.ShaderMaterial({
    uniforms: { uSize: { value: ST_DOT_U_SIZE } },
    vertexShader: NODE_VERTEX_SHADER,
    fragmentShader: ST_FRAGMENT_SHADER,
    transparent: false,
    depthWrite: true,
  }));
  stPoints.frustumCulled = false;
  stRoot.add(stPoints);

  stAxesGroup = buildSpaceTimeAxes(lx, ly, lt);
  stRoot.add(stAxesGroup);

  stCamera.position.set(-lx * 0.55, ly * 1.05, lt * 1.55);
  stControls.target.set(lx * 0.35, ly * 0.45, lt * 0.35);
  stControls.update();

  state.stTime = 0;
  state.stWindowFrac = parseFloat(stWindowSlider.value) / 100;
  recomputeStWindow();
  updateStWindow();
  stViewport.classList.add('has-data');
  syncStBarVisibility();
  resizeSpaceTime();
}

function updateStWindow() {
  if (!stGeom || !state.payload?.t?.length) return;
  // Cumulative space-time volume: each event at (x, y, t), reveal from t=0 to playhead.
  const winEnd = state.stTime + state.stWindowSize;
  const end = lowerBound(state.payload.t, winEnd);
  stGeom.setDrawRange(0, end);
  updateStBarLabels();
}

function setStTimeFromScrub() {
  const maxStart = Math.max(0, state.stSpan - state.stWindowSize);
  state.stTime = (parseInt(stScrub.value, 10) / 1000) * maxStart;
  updateStWindow();
}

function advanceSpaceTime(dtMs) {
  const maxStart = Math.max(0, state.stSpan - state.stWindowSize);
  const rate = state.stSpan / 10_000;
  state.stTime += dtMs * rate * state.speed;
  if (state.stTime >= maxStart) {
    state.stTime = maxStart;
    setRunning(false);
  }
  updateStWindow();
}

function resetSpaceTime() {
  if (!state.payload) return;
  state.stTime = 0;
  updateStWindow();
}

function addEvent(i) {
  const pos = state.positions;
  const D = state.maxDist;
  const D2 = D * D;
  const k = state.k;
  const x = pos[i * 3];
  const y = pos[i * 3 + 1];
  const z = pos[i * 3 + 2];

  const cand = [];
  for (let j = i - 1; j >= 0; j--) {
    const dz = z - pos[j * 3 + 2];
    if (dz > D) break;
    const dx = x - pos[j * 3];
    const dy = y - pos[j * 3 + 1];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 <= D2) cand.push([d2, j]);
  }
  cand.sort((a, b) => a[0] - b[0]);
  const take = Math.min(k, cand.length);

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

  const ep = edgesGeom.attributes.position.array;
  const ec = edgesGeom.attributes.color.array;
  const targetCol = state.polarities[i] > 0 ? COLOR_POS_TARGET : COLOR_NEG_TARGET;

  for (let t = 0; t < take; t++) {
    if (state.edgeCount >= state.edgeCap) break;
    const j = cand[t][1];
    const base = state.edgeCount * 6;
    ep[base] = pos[j * 3]; ep[base + 1] = pos[j * 3 + 1]; ep[base + 2] = pos[j * 3 + 2];
    ec[base] = COLOR_SOURCE.r; ec[base + 1] = COLOR_SOURCE.g; ec[base + 2] = COLOR_SOURCE.b;
    ep[base + 3] = x; ep[base + 4] = y; ep[base + 5] = z;
    ec[base + 3] = targetCol.r; ec[base + 4] = targetCol.g; ec[base + 5] = targetCol.b;
    state.edgeCount += 1;
  }
  edgesGeom.setDrawRange(0, state.edgeCount * 2);
  edgesGeom.attributes.position.needsUpdate = true;
  edgesGeom.attributes.color.needsUpdate = true;
}

function step(now) {
  const dtMs = state.lastFrame ? now - state.lastFrame : 16;
  state.lastFrame = now;

  if (state.running && state.payload) {
    if (state.viewMode === 'graph') {
      const baseRate = state.vtMax / 10_000;
      state.virtualTime += dtMs * baseRate * state.speed;
      if (state.virtualTime > state.vtMax) state.virtualTime = state.vtMax;

      let added = 0;
      while (state.nextIdx < state.N) {
        const z = state.positions[state.nextIdx * 3 + 2];
        if (z > state.virtualTime) break;
        addEvent(state.nextIdx);
        state.nextIdx += 1;
        if (++added > 8000) break;
      }
      if (state.nextIdx >= state.N && state.virtualTime >= state.vtMax) setRunning(false);
    } else if (state.viewMode === 'video') {
      advanceVideo(dtMs);
    } else if (state.viewMode === 'st3d') {
      advanceSpaceTime(dtMs);
    }
    updateStats();
  }

  state.fpsAcc += dtMs;
  state.fpsFrames += 1;
  if (state.fpsAcc >= 500) {
    document.getElementById('stat-fps').textContent =
      Math.round((state.fpsFrames * 1000) / state.fpsAcc);
    state.fpsAcc = 0;
    state.fpsFrames = 0;
  }

  if (state.viewMode === 'graph') {
    controls.update();
    renderer.render(scene, camera);
  } else if (state.viewMode === 'st3d') {
    stControls.update();
    stRenderer.render(stScene, stCamera);
  }

  requestAnimationFrame(step);
}
requestAnimationFrame(step);

function $(id) { return document.getElementById(id); }

function msg(text, kind = 'err') {
  const el = $('msg');
  el.textContent = text || '';
  el.className = 'row msg ' + (kind === 'info' ? 'info' : '');
}

function setViewMode(mode) {
  if (mode !== state.viewMode && state.running) setRunning(false);
  if (mode !== 'seg' && window.segOnViewLeave) window.segOnViewLeave();
  state.viewMode = mode;
  document.querySelectorAll('.viz-tab').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.view === mode);
  });
  document.querySelectorAll('.viz-pane').forEach((pane) => {
    pane.classList.toggle('active', pane.id === VIZ_PANE_IDS[mode]);
  });
  document.body.classList.toggle('view-graph', mode === 'graph');
  document.body.classList.toggle('view-video', mode === 'video');
  document.body.classList.toggle('view-st3d', mode === 'st3d');
  document.body.classList.toggle('view-seg', mode === 'seg');
  const hudGraph = $('hud-graph');
  const hudSeg = $('hud-seg');
  if (hudGraph) hudGraph.hidden = mode === 'seg';
  if (hudSeg) hudSeg.hidden = mode !== 'seg';
  syncVideoBarVisibility();
  syncStBarVisibility();
  if (mode === 'graph') resizeGraph();
  else if (mode === 'video') {
    resizeVideoDisplay();
    if (state.payload) renderVideoFrame();
  } else if (mode === 'st3d') {
    resizeSpaceTime();
    if (state.payload) updateStWindow();
  } else if (mode === 'seg' && window.segOnViewEnter) {
    window.segOnViewEnter();
  }
}

function updateStats() {
  if ((state.viewMode === 'video' || state.viewMode === 'st3d') && state.payload) {
    const isVideo = state.viewMode === 'video';
    const span = isVideo ? state.videoSpan : state.stSpan;
    const tMin = isVideo ? state.videoTMin : state.stTMin;
    const time = isVideo ? state.videoTime : state.stTime;
    const windowSize = isVideo ? state.videoWindowSize : state.stWindowSize;
    const maxStart = Math.max(0, span - windowSize);
    const frac = maxStart > 0 ? time / maxStart : 0;
    const winEnd = time + windowSize;
    $('stat-events').textContent = span
      ? (isVideo
        ? `${formatTimeUnits(tMin + time)} – ${formatTimeUnits(tMin + winEnd)}`
        : `${formatTimeUnits((state.payload?.t_min ?? 0))} – ${formatTimeUnits((state.payload?.t_min ?? 0) + winEnd)}`)
      : '—';
    $('stat-nodes').textContent = '—';
    $('stat-edges').textContent = '—';
    $('stat-time').textContent = frac.toFixed(3);
    return;
  }
  $('stat-events').textContent = `${state.nextIdx.toLocaleString()} / ${state.N.toLocaleString()}`;
  const graphStats = state.viewMode === 'graph';
  $('stat-nodes').textContent = graphStats ? state.nodeCount.toLocaleString() : '—';
  $('stat-edges').textContent = graphStats ? state.edgeCount.toLocaleString() : '—';
  $('stat-time').textContent = state.vtMax ? (state.virtualTime / state.vtMax).toFixed(3) : '0.000';
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
    if (key === 'timeScale') graphGroup.scale.z = v;
  };
  slider.addEventListener('input', apply);
  apply();
}

bindSlider('k', 'k-val', 'k');
bindSlider('d', 'd-val', 'maxDist');
bindSlider('sp', 'sp-val', 'speed', (v) => v.toFixed(2));
bindSlider('ts', 'ts-val', 'timeScale', (v) => v.toFixed(2));

$('refresh').addEventListener('click', async () => {
  const sel = $('file-select');
  sel.innerHTML = '';
  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = '(reloading…)';
  sel.appendChild(blank);
  clearLoadedFile();
  await fetchFiles();
});

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
    resetSpaceTime();
    resetVideo();
    updateStats();
  }
});

document.querySelectorAll('.viz-tab').forEach((btn) => {
  btn.addEventListener('click', () => setViewMode(btn.dataset.view));
});

stWindowSlider.addEventListener('input', () => {
  state.stWindowFrac = parseFloat(stWindowSlider.value) / 100;
  recomputeStWindow();
  updateStWindow();
});

videoWindowSlider.addEventListener('input', () => {
  state.videoWindowFrac = parseFloat(videoWindowSlider.value) / 100;
  recomputeVideoWindow();
  renderVideoFrame();
});

videoScrub.addEventListener('pointerdown', () => { state.videoScrubbing = true; });
videoScrub.addEventListener('pointerup', () => { state.videoScrubbing = false; });
videoScrub.addEventListener('input', () => {
  setRunning(false);
  setVideoTimeFromScrub();
});

stScrub.addEventListener('pointerdown', () => { state.stScrubbing = true; });
stScrub.addEventListener('pointerup', () => { state.stScrubbing = false; });
stScrub.addEventListener('input', () => {
  setRunning(false);
  setStTimeFromScrub();
});

$('fullscreen').addEventListener('click', () => {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
  else document.exitFullscreen?.();
});

$('collapse').addEventListener('click', () => {
  $('hud').classList.add('collapsed');
  $('show-hud').hidden = false;
});
$('show-hud').addEventListener('click', () => {
  $('hud').classList.remove('collapsed');
  $('show-hud').hidden = true;
});

fetchFiles();
