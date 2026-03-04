<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from "vue"

const props = defineProps<{
  node: any
  presetWidget: any
}>()

// --- Easing bezier approximations (for SVG preview of presets) ---
const EASING_BEZIER: Record<string, number[]> = {
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
}

const PRESET_NAMES = Object.keys(EASING_BEZIER)

// --- Reactive state ---
const currentPreset = ref<string>(props.presetWidget.value || "linear")
const showGrid = ref(false)
const dragging = ref<"p1" | "p2" | null>(null)

// Custom bezier control points
const cp = ref({ x1: 0.42, y1: 0.0, x2: 0.58, y2: 1.0 })

const isCustom = computed(() => currentPreset.value === "custom")

// SVG viewbox: 200x200, with padding. Curve area is 160x160 with 20px padding.
const PAD = 20
const SIZE = 160

function toSvgX(t: number): number {
  return PAD + t * SIZE
}

function toSvgY(t: number): number {
  return PAD + (1 - t) * SIZE // flip Y axis
}

// --- Bezier curve path for SVG ---
function bezierPath(x1: number, y1: number, x2: number, y2: number): string {
  return `M ${toSvgX(0)} ${toSvgY(0)} C ${toSvgX(x1)} ${toSvgY(y1)}, ${toSvgX(x2)} ${toSvgY(y2)}, ${toSvgX(1)} ${toSvgY(1)}`
}

const mainCurvePath = computed(() => {
  if (isCustom.value) {
    return bezierPath(cp.value.x1, cp.value.y1, cp.value.x2, cp.value.y2)
  }
  const b = EASING_BEZIER[currentPreset.value] || [0, 0, 1, 1]
  return bezierPath(b[0], b[1], b[2], b[3])
})

// Control point positions for custom mode
const p1 = computed(() => ({ x: toSvgX(cp.value.x1), y: toSvgY(cp.value.y1) }))
const p2 = computed(() => ({ x: toSvgX(cp.value.x2), y: toSvgY(cp.value.y2) }))

// --- Sync with ComfyUI widgets ---
function findWidget(name: string) {
  return props.node.widgets?.find((w: any) => w.name === name)
}

function syncFromWidgets() {
  currentPreset.value = props.presetWidget.value || "linear"
  const w_x1 = findWidget("x1")
  const w_y1 = findWidget("y1")
  const w_x2 = findWidget("x2")
  const w_y2 = findWidget("y2")
  if (w_x1) cp.value.x1 = w_x1.value
  if (w_y1) cp.value.y1 = w_y1.value
  if (w_x2) cp.value.x2 = w_x2.value
  if (w_y2) cp.value.y2 = w_y2.value
}

function pushToWidgets() {
  const w_x1 = findWidget("x1")
  const w_y1 = findWidget("y1")
  const w_x2 = findWidget("x2")
  const w_y2 = findWidget("y2")
  if (w_x1) w_x1.value = cp.value.x1
  if (w_y1) w_y1.value = cp.value.y1
  if (w_x2) w_x2.value = cp.value.x2
  if (w_y2) w_y2.value = cp.value.y2
}

function selectPreset(name: string) {
  props.presetWidget.value = name
  currentPreset.value = name
  // Update the bezier control point widgets to match the preset (for visual consistency)
  if (name !== "custom") {
    const b = EASING_BEZIER[name]
    if (b) {
      cp.value = { x1: b[0], y1: b[1], x2: b[2], y2: b[3] }
      pushToWidgets()
    }
  }
  props.node.setDirtyCanvas?.(true, true)
}

// --- Drag handling for custom bezier control points ---
const svgRef = ref<SVGSVGElement | null>(null)

function getSvgPoint(e: MouseEvent | PointerEvent) {
  const svg = svgRef.value
  if (!svg) return { x: 0, y: 0 }
  const rect = svg.getBoundingClientRect()
  const scaleX = 200 / rect.width
  const scaleY = 200 / rect.height
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  }
}

function fromSvgX(sx: number): number {
  return Math.max(0, Math.min(1, (sx - PAD) / SIZE))
}

function fromSvgY(sy: number): number {
  return Math.max(-1, Math.min(2, 1 - (sy - PAD) / SIZE))
}

function onPointerDown(which: "p1" | "p2", e: PointerEvent) {
  if (!isCustom.value) return
  e.preventDefault()
  e.stopPropagation()
  dragging.value = which
  ;(e.target as Element)?.setPointerCapture?.(e.pointerId)
}

function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return
  e.preventDefault()
  e.stopPropagation()
  const pt = getSvgPoint(e)
  const x = fromSvgX(pt.x)
  const y = fromSvgY(pt.y)
  if (dragging.value === "p1") {
    cp.value.x1 = Math.round(x * 100) / 100
    cp.value.y1 = Math.round(y * 100) / 100
  } else {
    cp.value.x2 = Math.round(x * 100) / 100
    cp.value.y2 = Math.round(y * 100) / 100
  }
  pushToWidgets()
}

function onPointerUp() {
  dragging.value = null
  props.node.setDirtyCanvas?.(true, true)
}

// --- Thumbnail bezier path for preset grid ---
function thumbPath(name: string): string {
  const b = EASING_BEZIER[name] || [0, 0, 1, 1]
  const s = 30 // thumbnail size
  const p = 3  // thumbnail padding
  const sz = s - 2 * p
  const tx = (t: number) => p + t * sz
  const ty = (t: number) => p + (1 - t) * sz
  return `M ${tx(0)} ${ty(0)} C ${tx(b[0])} ${ty(b[1])}, ${tx(b[2])} ${ty(b[3])}, ${tx(1)} ${ty(1)}`
}

// --- Poll for widget changes ---
let pollInterval: number | undefined

onMounted(() => {
  syncFromWidgets()
  pollInterval = window.setInterval(syncFromWidgets, 300)
})

onUnmounted(() => {
  if (pollInterval !== undefined) window.clearInterval(pollInterval)
})
</script>

<template>
  <div class="ease-curve-root">
    <!-- Main curve display -->
    <svg
      ref="svgRef"
      viewBox="0 0 200 200"
      class="curve-svg"
      @pointermove="onPointerMove"
      @pointerup="onPointerUp"
      @pointerleave="onPointerUp"
    >
      <!-- Background -->
      <rect x="0" y="0" width="200" height="200" fill="#1a1a2e" rx="6" />

      <!-- Grid lines -->
      <line
        v-for="i in 3"
        :key="'gv' + i"
        :x1="PAD + (i * SIZE) / 4"
        :y1="PAD"
        :x2="PAD + (i * SIZE) / 4"
        :y2="PAD + SIZE"
        stroke="#2a2a4a"
        stroke-width="0.5"
      />
      <line
        v-for="i in 3"
        :key="'gh' + i"
        :x1="PAD"
        :y1="PAD + (i * SIZE) / 4"
        :x2="PAD + SIZE"
        :y2="PAD + (i * SIZE) / 4"
        stroke="#2a2a4a"
        stroke-width="0.5"
      />

      <!-- Curve area border -->
      <rect
        :x="PAD"
        :y="PAD"
        :width="SIZE"
        :height="SIZE"
        fill="none"
        stroke="#3a3a5a"
        stroke-width="1"
      />

      <!-- Linear reference line (diagonal) -->
      <line
        :x1="toSvgX(0)"
        :y1="toSvgY(0)"
        :x2="toSvgX(1)"
        :y2="toSvgY(1)"
        stroke="#3a3a5a"
        stroke-width="1"
        stroke-dasharray="4 4"
      />

      <!-- Easing curve -->
      <path
        :d="mainCurvePath"
        fill="none"
        stroke="#6cf"
        stroke-width="2.5"
        stroke-linecap="round"
      />

      <!-- Control point tangent lines (custom mode only) -->
      <template v-if="isCustom">
        <line
          :x1="toSvgX(0)"
          :y1="toSvgY(0)"
          :x2="p1.x"
          :y2="p1.y"
          stroke="#f80"
          stroke-width="1"
          stroke-dasharray="3 3"
          opacity="0.7"
        />
        <line
          :x1="toSvgX(1)"
          :y1="toSvgY(1)"
          :x2="p2.x"
          :y2="p2.y"
          stroke="#f80"
          stroke-width="1"
          stroke-dasharray="3 3"
          opacity="0.7"
        />

        <!-- Draggable control points -->
        <circle
          :cx="p1.x"
          :cy="p1.y"
          r="6"
          fill="#f80"
          stroke="#fff"
          stroke-width="1.5"
          class="handle"
          @pointerdown="onPointerDown('p1', $event)"
        />
        <circle
          :cx="p2.x"
          :cy="p2.y"
          r="6"
          fill="#f80"
          stroke="#fff"
          stroke-width="1.5"
          class="handle"
          @pointerdown="onPointerDown('p2', $event)"
        />
      </template>

      <!-- Start/end points -->
      <circle :cx="toSvgX(0)" :cy="toSvgY(0)" r="3" fill="#6cf" />
      <circle :cx="toSvgX(1)" :cy="toSvgY(1)" r="3" fill="#6cf" />

      <!-- Label -->
      <text x="100" y="14" text-anchor="middle" fill="#889" font-size="10" font-family="sans-serif">
        {{ isCustom ? `custom (${cp.x1}, ${cp.y1}, ${cp.x2}, ${cp.y2})` : currentPreset }}
      </text>
    </svg>

    <!-- Preset grid toggle -->
    <button class="grid-toggle" @click="showGrid = !showGrid">
      {{ showGrid ? "Hide presets" : "Show presets" }}
    </button>

    <!-- Preset thumbnail grid -->
    <div v-if="showGrid" class="preset-grid">
      <div
        v-for="name in PRESET_NAMES"
        :key="name"
        class="preset-thumb"
        :class="{ active: currentPreset === name }"
        @click="selectPreset(name)"
        :title="name"
      >
        <svg viewBox="0 0 30 30" class="thumb-svg">
          <rect x="0" y="0" width="30" height="30" fill="none" />
          <line x1="3" y1="27" x2="27" y2="3" stroke="#2a2a4a" stroke-width="0.5" stroke-dasharray="2 2" />
          <path :d="thumbPath(name)" fill="none" stroke="#6cf" stroke-width="1.5" stroke-linecap="round" />
        </svg>
        <span class="thumb-label">{{ name.replace(/ease/i, '').replace(/In|Out/g, m => m + ' ').trim() || 'linear' }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.ease-curve-root {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 4px;
  font-family: sans-serif;
}

.curve-svg {
  width: 100%;
  max-width: 260px;
  border-radius: 6px;
  user-select: none;
  touch-action: none;
}

.handle {
  cursor: grab;
}
.handle:active {
  cursor: grabbing;
}

.grid-toggle {
  background: #2a2a4a;
  color: #aab;
  border: 1px solid #3a3a5a;
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 10px;
  cursor: pointer;
  width: 100%;
  max-width: 260px;
}
.grid-toggle:hover {
  background: #3a3a5a;
  color: #dde;
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 3px;
  width: 100%;
  max-width: 260px;
}

.preset-thumb {
  display: flex;
  flex-direction: column;
  align-items: center;
  cursor: pointer;
  border: 1px solid #2a2a4a;
  border-radius: 4px;
  padding: 2px;
  background: #1a1a2e;
  transition: border-color 0.15s;
}
.preset-thumb:hover {
  border-color: #6cf;
}
.preset-thumb.active {
  border-color: #6cf;
  background: #222244;
}

.thumb-svg {
  width: 100%;
  aspect-ratio: 1;
}

.thumb-label {
  font-size: 6px;
  color: #778;
  text-align: center;
  line-height: 1.1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  width: 100%;
}
</style>
