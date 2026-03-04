import { app } from "../../scripts/app.js";

// --- Bezier approximations for preset curve previews ---
const EASING_BEZIER = {
  linear:          [0.00, 0.00, 1.00, 1.00],
  easeInQuad:      [0.55, 0.085, 0.68, 0.53],
  easeOutQuad:     [0.25, 0.46, 0.45, 0.94],
  easeInOutQuad:   [0.455, 0.03, 0.515, 0.955],
  easeInCubic:     [0.55, 0.055, 0.675, 0.19],
  easeOutCubic:    [0.215, 0.61, 0.355, 1.0],
  easeInOutCubic:  [0.645, 0.045, 0.355, 1.0],
  easeInQuart:     [0.895, 0.03, 0.685, 0.22],
  easeOutQuart:    [0.165, 0.84, 0.44, 1.0],
  easeInOutQuart:  [0.77, 0.0, 0.175, 1.0],
  easeInQuint:     [0.755, 0.05, 0.855, 0.06],
  easeOutQuint:    [0.23, 1.0, 0.32, 1.0],
  easeInOutQuint:  [0.86, 0.0, 0.07, 1.0],
  easeInSine:      [0.47, 0.0, 0.745, 0.715],
  easeOutSine:     [0.39, 0.575, 0.565, 1.0],
  easeInOutSine:   [0.445, 0.05, 0.55, 0.95],
  easeInExpo:      [0.95, 0.05, 0.795, 0.035],
  easeOutExpo:     [0.19, 1.0, 0.22, 1.0],
  easeInOutExpo:   [1.0, 0.0, 0.0, 1.0],
  easeInCirc:      [0.6, 0.04, 0.98, 0.335],
  easeOutCirc:     [0.075, 0.82, 0.165, 1.0],
  easeInOutCirc:   [0.785, 0.135, 0.15, 0.86],
  easeInBack:      [0.6, -0.28, 0.735, 0.045],
  easeOutBack:     [0.175, 0.885, 0.32, 1.275],
  easeInOutBack:   [0.68, -0.55, 0.265, 1.55],
  easeInElastic:   [0.5, -0.5, 0.75, -0.5],
  easeOutElastic:  [0.25, 1.5, 0.5, 1.5],
  easeInOutElastic:[0.5, -0.5, 0.5, 1.5],
  easeInBounce:    [0.5, -0.3, 0.7, -0.3],
  easeOutBounce:   [0.3, 1.3, 0.5, 1.3],
  easeInOutBounce: [0.5, -0.3, 0.5, 1.3],
};

// --- Cubic bezier evaluator ---
function evalBezierComponent(t, p1, p2) {
  return 3 * (1 - t) * (1 - t) * t * p1 + 3 * (1 - t) * t * t * p2 + t * t * t;
}

function evalBezierDerivative(t, p1, p2) {
  return 3 * (1 - t) * (1 - t) * p1 + 6 * (1 - t) * t * (p2 - p1) + 3 * t * t * (1 - p2);
}

function solveBezierX(x, p1x, p2x) {
  let t = x;
  for (let i = 0; i < 8; i++) {
    const err = evalBezierComponent(t, p1x, p2x) - x;
    const d = evalBezierDerivative(t, p1x, p2x);
    if (Math.abs(d) < 1e-7) break;
    t -= err / d;
    t = Math.max(0, Math.min(1, t));
  }
  if (Math.abs(evalBezierComponent(t, p1x, p2x) - x) > 1e-4) {
    let lo = 0, hi = 1;
    for (let i = 0; i < 20; i++) {
      const mid = (lo + hi) / 2;
      if (evalBezierComponent(mid, p1x, p2x) < x) lo = mid; else hi = mid;
    }
    t = (lo + hi) / 2;
  }
  return t;
}

function evalBezierCurve(x, p1x, p1y, p2x, p2y) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const t = solveBezierX(x, p1x, p2x);
  return evalBezierComponent(t, p1y, p2y);
}

// --- Canvas drawing ---
function drawCurve(canvas, preset, x1, y1, x2, y2) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const pad = 16;
  const cw = w - pad * 2;
  const ch = h - pad * 2;

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = "#1a1a2e";
  ctx.beginPath();
  ctx.roundRect(0, 0, w, h, 6);
  ctx.fill();

  // Grid
  ctx.strokeStyle = "#2a2a4a";
  ctx.lineWidth = 0.5;
  for (let i = 1; i <= 3; i++) {
    const gx = pad + (i / 4) * cw;
    const gy = pad + (i / 4) * ch;
    ctx.beginPath(); ctx.moveTo(gx, pad); ctx.lineTo(gx, pad + ch); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(pad + cw, gy); ctx.stroke();
  }

  // Border
  ctx.strokeStyle = "#3a3a5a";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, cw, ch);

  // Diagonal reference
  ctx.strokeStyle = "#3a3a5a";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad, pad + ch);
  ctx.lineTo(pad + cw, pad);
  ctx.stroke();
  ctx.setLineDash([]);

  // Get bezier control points
  let bx1, by1, bx2, by2;
  if (preset === "custom") {
    bx1 = x1; by1 = y1; bx2 = x2; by2 = y2;
  } else {
    const b = EASING_BEZIER[preset] || [0, 0, 1, 1];
    bx1 = b[0]; by1 = b[1]; bx2 = b[2]; by2 = b[3];
  }

  // Draw easing curve as polyline
  ctx.strokeStyle = "#66ccff";
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  const steps = 80;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    const val = evalBezierCurve(t, bx1, by1, bx2, by2);
    const px = pad + t * cw;
    const py = pad + (1 - val) * ch;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Control point tangent lines (custom mode)
  if (preset === "custom") {
    ctx.strokeStyle = "#ff8800";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.globalAlpha = 0.7;

    // P0 -> P1
    ctx.beginPath();
    ctx.moveTo(pad, pad + ch);
    ctx.lineTo(pad + bx1 * cw, pad + (1 - by1) * ch);
    ctx.stroke();

    // P3 -> P2
    ctx.beginPath();
    ctx.moveTo(pad + cw, pad);
    ctx.lineTo(pad + bx2 * cw, pad + (1 - by2) * ch);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.globalAlpha = 1;

    // Control point handles
    [
      [pad + bx1 * cw, pad + (1 - by1) * ch],
      [pad + bx2 * cw, pad + (1 - by2) * ch],
    ].forEach(([cx, cy]) => {
      ctx.fillStyle = "#ff8800";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }

  // Start/end dots
  ctx.fillStyle = "#66ccff";
  ctx.beginPath(); ctx.arc(pad, pad + ch, 3, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(pad + cw, pad, 3, 0, Math.PI * 2); ctx.fill();

  // Label
  ctx.fillStyle = "#8899aa";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  const label = preset === "custom"
    ? `custom (${bx1.toFixed(2)}, ${by1.toFixed(2)}, ${bx2.toFixed(2)}, ${by2.toFixed(2)})`
    : preset;
  ctx.fillText(label, w / 2, 12);
}

// --- Extension registration ---
const EASING_NODE_CLASSES = ["EaseCurve", "ApplyEasingToFloats"];

app.registerExtension({
  name: "nodesweet.easeCurve",

  nodeCreated(node) {
    if (!EASING_NODE_CLASSES.includes(node.comfyClass)) return;

    const presetWidget = node.widgets?.find((w) => w.name === "preset");
    if (!presetWidget) return;

    // Create a canvas element for the curve preview
    const canvas = document.createElement("canvas");
    canvas.width = 260;
    canvas.height = 200;
    canvas.style.width = "100%";
    canvas.style.maxWidth = "260px";
    canvas.style.borderRadius = "6px";

    // Helper to get current widget values
    function getWidgetValue(name) {
      const w = node.widgets?.find((w) => w.name === name);
      return w ? w.value : undefined;
    }

    function redraw() {
      const preset = getWidgetValue("preset") || "linear";
      const x1 = getWidgetValue("x1") ?? 0.42;
      const y1 = getWidgetValue("y1") ?? 0.0;
      const x2 = getWidgetValue("x2") ?? 0.58;
      const y2 = getWidgetValue("y2") ?? 1.0;
      drawCurve(canvas, preset, x1, y1, x2, y2);
    }

    // Initial draw
    redraw();

    // Add as DOM widget
    const widget = node.addDOMWidget("curve_preview", "canvas", canvas, {
      serialize: false,
    });
    widget.computeSize = () => [260, 210];

    // Redraw on any widget change
    const origCallback = node.onWidgetChanged;
    node.onWidgetChanged = function (name, value) {
      if (origCallback) origCallback.call(this, name, value);
      redraw();
    };

    // Also poll for changes (some widgets don't fire onWidgetChanged)
    let lastState = "";
    setInterval(() => {
      const state = [
        getWidgetValue("preset"),
        getWidgetValue("x1"),
        getWidgetValue("y1"),
        getWidgetValue("x2"),
        getWidgetValue("y2"),
      ].join(",");
      if (state !== lastState) {
        lastState = state;
        redraw();
      }
    }, 200);

    // Resize node
    node.setSize([
      Math.max(node.size[0], 300),
      node.computeSize()[1],
    ]);
  },
});
