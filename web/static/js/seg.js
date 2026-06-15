// event_segmentation tab — run pipeline, canvas video playback, static plots.

import {
  VIDEO_WINDOW_FRAC_DEFAULT,
  getEventVideoDisplayScale,
  eventVideoDotPx,
} from './video-common.js';

const SEG_BG = [0, 0, 0];

const segState = {
  defaults: null,
  groups: null,
  jobId: null,
  pollTimer: null,
  payload: null,
  colorRgb: null,
  imgW: 0,
  imgH: 0,
  xOff: 0,
  yOff: 0,
  ctx: null,
  imageData: null,
  dispW: 0,
  dispH: 0,
  dispScale: 1,
  time: 0,
  span: 0,
  tMin: 0,
  windowFrac: VIDEO_WINDOW_FRAC_DEFAULT,
  windowSize: 0,
  scrubbing: false,
  running: false,
  speed: 1,
  lastFrame: 0,
};

const segCanvas = document.getElementById('seg-video-canvas');
const segViewport = document.getElementById('seg-viewport');
const segMainScroll = document.getElementById('seg-main-scroll');
const segControlBar = document.getElementById('seg-control-bar');
const segWindowSlider = document.getElementById('seg-vw');
const segWindowVal = document.getElementById('seg-vw-val');
const segScrub = document.getElementById('seg-scrub');
const segProgress = document.getElementById('seg-progress');
const segTimeRange = document.getElementById('seg-time-range');

function seg$(id) { return document.getElementById(id); }

function segMsg(text, kind = 'err') {
  const el = seg$('seg-msg');
  if (!el) return;
  el.textContent = text || '';
  el.className = 'row msg ' + (kind === 'info' ? 'info' : '');
}

function assetUrl(jobId, name) {
  return `/api/segment/assets/${jobId}/${name}?t=${Date.now()}`;
}

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

function segDotPx() {
  return eventVideoDotPx();
}

function fillImageBlack(data) {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = SEG_BG[0];
    data[i + 1] = SEG_BG[1];
    data[i + 2] = SEG_BG[2];
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

function buildColorLookup(payload) {
  const cmap = payload.colormap || {};
  const defaultGray = cmap['-1'] || [110, 110, 110];
  const ids = payload.instance_id;
  const out = new Array(ids.length);
  const cache = {};
  for (let i = 0; i < ids.length; i++) {
    const key = String(ids[i]);
    let rgb = cache[key];
    if (!rgb) {
      rgb = cmap[key] || defaultGray;
      cache[key] = rgb;
    }
    out[i] = rgb;
  }
  return out;
}

function setImageExtents(payload) {
  const xMin = Math.floor(payload.x_min);
  const xMax = Math.ceil(payload.x_max);
  const yMin = Math.floor(payload.y_min);
  const yMax = Math.ceil(payload.y_max);
  segState.xOff = xMin;
  segState.yOff = yMin;
  segState.imgW = Math.max(1, xMax - xMin + 1);
  segState.imgH = Math.max(1, yMax - yMin + 1);
}

function getSegHudInset() {
  const hud = seg$('hud');
  if (!hud || hud.classList.contains('collapsed')) return 0;
  return hud.offsetWidth + 32;
}

function layoutSegPanel() {
  const scroll = segMainScroll;
  if (!scroll) return;

  const inset = getSegHudInset();
  scroll.style.paddingLeft = inset ? `${inset}px` : '';
  scroll.style.paddingRight = inset ? '16px' : '';
}

function scrollToSegResults() {
  const results = seg$('seg-results-scroll');
  const scroll = segMainScroll;
  if (!results || results.hidden || !scroll) return;
  requestAnimationFrame(() => {
    layoutSegPanel();
    const top = results.offsetTop - 12;
    scroll.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
  });
}

function getSegDisplayScale() {
  return getEventVideoDisplayScale(
    segState.imgW,
    segState.imgH,
    segControlBar?.hidden ?? true,
    getSegHudInset(),
  );
}

function resizeSegVideo() {
  if (!segState.imgW || !segState.imgH || !segViewport?.classList.contains('has-preview')) return;
  layoutSegPanel();
  const scale = getSegDisplayScale();
  const dispW = Math.max(1, Math.round(segState.imgW * scale));
  const dispH = Math.max(1, Math.round(segState.imgH * scale));
  segState.dispScale = scale;
  segState.dispW = dispW;
  segState.dispH = dispH;
  if (segCanvas) {
    segCanvas.width = dispW;
    segCanvas.height = dispH;
    segCanvas.style.width = `${dispW}px`;
    segCanvas.style.height = `${dispH}px`;
  }
  if (segState.ctx) {
    segState.imageData = segState.ctx.createImageData(dispW, dispH);
  }
}

function recomputeSegWindow() {
  const t = segState.payload?.t;
  if (!t?.length) {
    segState.span = 1;
    segState.windowSize = 1;
    return;
  }
  segState.tMin = t[0];
  segState.span = Math.max(t[t.length - 1] - segState.tMin, 1e-9);
  segState.windowSize = Math.max(segState.span * segState.windowFrac, 1e-9);
  const maxStart = Math.max(0, segState.span - segState.windowSize);
  if (segState.time > maxStart) segState.time = maxStart;
}

function updateSegBarLabels() {
  const absBase = segState.payload?.t_min ?? 0;
  const winStart = segState.tMin + segState.time;
  const winEnd = winStart + segState.windowSize;
  const maxStart = Math.max(0, segState.span - segState.windowSize);
  const pct = maxStart > 0 ? (segState.time / maxStart) * 100 : 100;
  if (segWindowVal) segWindowVal.textContent = `${(segState.windowFrac * 100).toFixed(1)}%`;
  if (segProgress) segProgress.textContent = `${pct.toFixed(0)}%`;
  if (!segState.scrubbing && segScrub) segScrub.value = String(Math.round(pct * 10));

  const t = segState.payload?.t;
  const inst = segState.payload?.instance_id;
  let nEvents = 0;
  let nInst = 0;
  if (t?.length) {
    const start = lowerBound(t, segState.time);
    const end = lowerBound(t, winEnd);
    nEvents = end - start;
    if (inst && end > start) {
      const slice = inst.slice(start, end);
      const uniq = new Set(slice.filter((v) => v >= 0));
      nInst = uniq.size;
    }
  }
  if (segTimeRange) {
    segTimeRange.textContent =
      `${formatTimeUnits(absBase + winStart)} – ${formatTimeUnits(absBase + winEnd)} · `
      + `${nEvents.toLocaleString()} events · ${nInst} instances`;
  }
}

function renderSegFrame() {
  if (!segState.ctx || !segState.imageData || !segState.payload?.t?.length) return;
  const data = segState.imageData.data;
  const dispW = segState.dispW;
  const dispH = segState.dispH;
  const scale = segState.dispScale;
  const dotPx = segDotPx();
  fillImageBlack(data);
  const payload = segState.payload;
  const t = payload.t;
  const winEnd = segState.time + segState.windowSize;
  const start = lowerBound(t, segState.time);
  const end = lowerBound(t, winEnd);
  for (let i = start; i < end; i++) {
    const cx = Math.round((Math.round(payload.x[i]) - segState.xOff) * scale);
    const cy = Math.round((Math.round(payload.y[i]) - segState.yOff) * scale);
    plotSharpDot(data, dispW, dispH, cx, cy, dotPx, segState.colorRgb[i]);
  }
  segState.ctx.putImageData(segState.imageData, 0, 0);
  updateSegBarLabels();
}

function setSegTimeFromScrub() {
  const maxStart = Math.max(0, segState.span - segState.windowSize);
  segState.time = (parseInt(segScrub.value, 10) / 1000) * maxStart;
  renderSegFrame();
}

function advanceSegVideo(dtMs) {
  const maxStart = Math.max(0, segState.span - segState.windowSize);
  const rate = segState.span / 10_000;
  segState.time += dtMs * rate * segState.speed;
  if (segState.time >= maxStart) {
    segState.time = maxStart;
    setSegRunning(false);
  }
  renderSegFrame();
}

function resetSegVideo() {
  if (!segState.payload) return;
  segState.time = 0;
  renderSegFrame();
}

function setSegRunning(on) {
  segState.running = on;
  const hasData = Boolean(segState.payload);
  if (seg$('seg-play')) seg$('seg-play').disabled = on || !hasData;
  if (seg$('seg-pause')) seg$('seg-pause').disabled = !on;
  if (seg$('seg-reset')) seg$('seg-reset').disabled = !hasData;
}

function prepareSegVideo(payload) {
  segState.payload = payload;
  segState.colorRgb = buildColorLookup(payload);
  setImageExtents(payload);
  segState.ctx = segCanvas?.getContext('2d', { alpha: false });
  segState.time = 0;
  segState.windowFrac = parseFloat(segWindowSlider?.value || '2') / 100;
  recomputeSegWindow();
  segViewport?.classList.add('has-preview');
  if (segControlBar) segControlBar.hidden = false;
  if (seg$('seg-empty')) seg$('seg-empty').hidden = true;
  resizeSegVideo();
  renderSegFrame();
  setSegRunning(false);
}

async function loadSegEvents(jobId) {
  segMsg('Loading labeled events…', 'info');
  const r = await fetch(`/api/segment/events?job_id=${encodeURIComponent(jobId)}`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || 'Failed to load labeled events');
  prepareSegVideo(j);
  const note = j.downsampled_from
    ? ` (${j.count.toLocaleString()} shown, downsampled from ${j.downsampled_from.toLocaleString()})`
    : ` (${j.count.toLocaleString()} events)`;
  segMsg(`Ready for playback${note}. Scroll down for plots, or press Play.`, 'info');
}

function collectSegParams() {
  const params = {};
  if (!segState.groups) return params;
  for (const group of segState.groups) {
    for (const field of group.fields) {
      const el = seg$(`seg-param-${field.key}`);
      if (!el) continue;
      if (field.type === 'bool') {
        params[field.key] = el.checked;
      } else if (field.type === 'int') {
        params[field.key] = parseInt(el.value, 10);
      } else {
        params[field.key] = parseFloat(el.value);
      }
    }
  }
  return params;
}

function buildParamForm(defaults, groups) {
  const root = seg$('seg-params');
  if (!root) return;
  root.innerHTML = '';
  for (const group of groups) {
    const details = document.createElement('details');
    details.className = 'seg-param-group';
    details.open = group.id === 'io' || group.id === 'tracks';
    const summary = document.createElement('summary');
    summary.textContent = group.label;
    details.appendChild(summary);
    const inner = document.createElement('div');
    inner.className = 'seg-param-fields';
    for (const field of group.fields) {
      const label = document.createElement('label');
      label.className = 'seg-field';
      label.htmlFor = `seg-param-${field.key}`;
      const cap = document.createElement('span');
      cap.className = 'seg-field-label';
      cap.textContent = field.label;
      label.appendChild(cap);
      const defVal = defaults[field.key];
      if (field.type === 'bool') {
        const inp = document.createElement('input');
        inp.type = 'checkbox';
        inp.id = `seg-param-${field.key}`;
        inp.checked = Boolean(defVal);
        label.appendChild(inp);
      } else {
        const inp = document.createElement('input');
        inp.type = 'number';
        inp.id = `seg-param-${field.key}`;
        inp.step = field.type === 'int' ? '1' : 'any';
        inp.value = defVal;
        label.appendChild(inp);
      }
      inner.appendChild(label);
    }
    details.appendChild(inner);
    root.appendChild(details);
  }
}

async function loadSegDefaults() {
  const r = await fetch('/api/segment/defaults');
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || 'Failed to load defaults');
  segState.defaults = j.defaults;
  segState.groups = j.groups;
  buildParamForm(j.defaults, j.groups);
}

function setSegProgress(visible, pct = 0, message = '') {
  const row = seg$('seg-progress-row');
  const fill = seg$('seg-progress-fill');
  const msgEl = seg$('seg-progress-msg');
  if (row) row.hidden = !visible;
  if (fill) fill.style.width = `${Math.round(pct * 100)}%`;
  if (msgEl) msgEl.textContent = message || '—';
}

function renderSummary(summary) {
  const panel = seg$('seg-summary-panel');
  if (!panel || !summary) return;
  const rows = [
    ['Total events', summary.total_events?.toLocaleString?.() ?? summary.total_events],
    ['Instances', summary.n_confirmed_instances],
    ['Assigned', summary.events_assigned_to_instances?.toLocaleString?.() ?? summary.events_assigned_to_instances],
    ['Background', summary.events_background?.toLocaleString?.() ?? summary.events_background],
    ['Duration', summary.duration_s != null ? `${summary.duration_s} s` : '—'],
    ['Runtime', summary.wall_clock_s != null ? `${summary.wall_clock_s} s` : '—'],
    ['Throughput', summary.throughput_events_per_s != null ? `${summary.throughput_events_per_s} ev/s` : '—'],
    ['Sensor', summary.sensor_width && summary.sensor_height
      ? `${summary.sensor_width} × ${summary.sensor_height}` : '—'],
  ];
  panel.innerHTML = `
    <h2>Segmentation summary</h2>
    <dl class="seg-summary-grid">
      ${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v ?? '—'}</dd>`).join('')}
    </dl>`;
}

async function showSegResults(jobId, summary) {
  segState.jobId = jobId;
  const resultsScroll = seg$('seg-results-scroll');
  const results = seg$('seg-results');
  if (resultsScroll) resultsScroll.hidden = false;
  if (results) results.hidden = false;

  const traj = seg$('seg-trajectories');
  const speed = seg$('seg-speed');
  if (traj) traj.src = assetUrl(jobId, 'trajectories.png');
  if (speed) speed.src = assetUrl(jobId, 'speed_profiles.png');
  renderSummary(summary);

  await loadSegEvents(jobId);
  layoutSegPanel();
  resizeSegVideo();
  renderSegFrame();
  scrollToSegResults();
}

function stopPolling() {
  if (segState.pollTimer) {
    clearInterval(segState.pollTimer);
    segState.pollTimer = null;
  }
}

async function pollJob(jobId) {
  const r = await fetch(`/api/segment/status?job_id=${encodeURIComponent(jobId)}`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || 'Status check failed');
  setSegProgress(true, j.progress ?? 0, j.message || j.status);
  if (j.status === 'done') {
    stopPolling();
    seg$('seg-run').disabled = false;
    setSegProgress(false);
    segMsg('Segmentation complete.', 'info');
    await showSegResults(jobId, j.summary);
  } else if (j.status === 'error') {
    stopPolling();
    seg$('seg-run').disabled = false;
    setSegProgress(false);
    segMsg(j.error || j.message || 'Segmentation failed.');
  }
}

async function runSegmentation() {
  const name = seg$('file-select')?.value;
  if (!name) {
    segMsg('Select an event file first.');
    return;
  }
  stopPolling();
  setSegRunning(false);
  segState.payload = null;
  segState.colorRgb = null;
  segViewport?.classList.remove('has-preview');
  if (segControlBar) segControlBar.hidden = true;
  if (segMainScroll) segMainScroll.scrollTop = 0;
  if (seg$('seg-empty')) seg$('seg-empty').hidden = false;
  if (seg$('seg-results-scroll')) seg$('seg-results-scroll').hidden = true;
  if (seg$('seg-results')) seg$('seg-results').hidden = true;
  if (segCanvas) {
    const ctx = segCanvas.getContext('2d');
    ctx?.clearRect(0, 0, segCanvas.width, segCanvas.height);
  }

  segMsg('Starting segmentation…', 'info');
  seg$('seg-run').disabled = true;
  setSegProgress(true, 0, 'Queued…');

  const body = { input: name, params: collectSegParams() };
  const r = await fetch('/api/segment/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const j = await r.json();
  if (!r.ok) {
    seg$('seg-run').disabled = false;
    setSegProgress(false);
    segMsg(j.error || 'Failed to start job.');
    return;
  }

  segState.jobId = j.job_id;
  segMsg(`Processing ${j.n_events?.toLocaleString?.() ?? j.n_events} events…`, 'info');
  await pollJob(j.job_id);
  segState.pollTimer = setInterval(() => {
    pollJob(j.job_id).catch((err) => {
      stopPolling();
      seg$('seg-run').disabled = false;
      setSegProgress(false);
      segMsg(err.message);
    });
  }, 1500);
}

function segStep(now) {
  const dtMs = segState.lastFrame ? now - segState.lastFrame : 16;
  segState.lastFrame = now;
  if (segState.running && segState.payload && document.body.classList.contains('view-seg')) {
    advanceSegVideo(dtMs);
  }
  requestAnimationFrame(segStep);
}
requestAnimationFrame(segStep);

window.segOnViewEnter = () => {
  layoutSegPanel();
  resizeSegVideo();
  if (segState.payload) renderSegFrame();
};

window.segOnViewLeave = () => {
  setSegRunning(false);
};

window.addEventListener('resize', () => {
  if (document.body.classList.contains('view-seg')) {
    layoutSegPanel();
    resizeSegVideo();
    if (segState.payload) renderSegFrame();
  }
});

seg$('collapse')?.addEventListener('click', () => {
  requestAnimationFrame(() => {
    if (document.body.classList.contains('view-seg')) {
      layoutSegPanel();
      resizeSegVideo();
      if (segState.payload) renderSegFrame();
    }
  });
});
seg$('show-hud')?.addEventListener('click', () => {
  requestAnimationFrame(() => {
    if (document.body.classList.contains('view-seg')) {
      layoutSegPanel();
      resizeSegVideo();
      if (segState.payload) renderSegFrame();
    }
  });
});

const segHudObserver = seg$('hud-seg')
  ? new ResizeObserver(() => {
      if (document.body.classList.contains('view-seg')) layoutSegPanel();
    })
  : null;
if (segHudObserver && seg$('hud')) segHudObserver.observe(seg$('hud'));

segWindowSlider?.addEventListener('input', () => {
  segState.windowFrac = parseFloat(segWindowSlider.value) / 100;
  recomputeSegWindow();
  renderSegFrame();
});

segScrub?.addEventListener('pointerdown', () => { segState.scrubbing = true; });
segScrub?.addEventListener('pointerup', () => { segState.scrubbing = false; });
segScrub?.addEventListener('input', () => {
  setSegRunning(false);
  setSegTimeFromScrub();
});

seg$('seg-sp')?.addEventListener('input', () => {
  segState.speed = parseFloat(seg$('seg-sp').value);
  seg$('seg-sp-val').textContent = segState.speed.toFixed(1);
});
segState.speed = parseFloat(seg$('seg-sp')?.value || '1');

seg$('seg-run')?.addEventListener('click', () => {
  runSegmentation().catch((err) => {
    seg$('seg-run').disabled = false;
    setSegProgress(false);
    segMsg(err.message);
  });
});

seg$('seg-play')?.addEventListener('click', () => setSegRunning(true));
seg$('seg-pause')?.addEventListener('click', () => setSegRunning(false));
seg$('seg-reset')?.addEventListener('click', () => {
  setSegRunning(false);
  resetSegVideo();
});

loadSegDefaults().catch((err) => segMsg(err.message));
