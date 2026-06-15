/** Shared 2D event-video canvas sizing (Event video + event_segmentation tabs). */

export const VIDEO_WINDOW_FRAC_DEFAULT = 0.02;
export const EVENT_VIDEO_DOT_U_SIZE = 0.85;
export const EVENT_VIDEO_DOT_DEPTH_REF = 150;

export function getEventVideoDisplayScale(imgW, imgH, controlBarHidden, contentInsetLeft = 0) {
  if (!imgW || !imgH) return 1;
  const pad = 48;
  const barH = controlBarHidden ? 0 : 180;
  const maxW = window.innerWidth - pad - contentInsetLeft;
  const maxH = window.innerHeight - pad - barH;
  return Math.min(maxW / imgW, maxH / imgH, 16);
}

export function eventVideoDotPx() {
  return Math.max(1, Math.round(EVENT_VIDEO_DOT_U_SIZE * (300 / EVENT_VIDEO_DOT_DEPTH_REF)));
}
