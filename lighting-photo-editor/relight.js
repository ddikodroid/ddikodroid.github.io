/*
 * APICORA LIGHT — local relighting engine (no API key, runs fully in the browser).
 *
 * Pipeline:
 *   1. Monocular depth estimation with Depth Anything V2 (small, int8) via transformers.js
 *      (model is downloaded once from the Hugging Face CDN and cached by the browser).
 *   2. Surface normals derived from the depth map.
 *   3. Per-preset virtual light rig (key / fill / rim / on-axis flash with distance falloff)
 *      evaluated against those normals, plus depth-aware background exposure.
 *   4. Photographic grade: exposure, tone shoulder, temperature, split-toning, contrast,
 *      highlights / shadows, saturation, fade, glow / halation, vignette and grain.
 *
 * Exposes window.LocalRelight.run(originalFullRes, preset, adjustments, onStatus)
 * returning { blob, mimeType, width, height } — same shape as the Gemini path.
 */
(function () {
  'use strict';

  const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1';
  const DEPTH_MODEL = 'onnx-community/depth-anything-v2-small';
  const DEPTH_INPUT_EDGE = 640;
  const WORK_MAX_EDGE = 3072;

  /* ------------------------------------------------------------------ utils */
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const smooth = (e0, e1, x) => {
    const t = clamp((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
  };
  const norm3 = (v) => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
  };

  // Approximate black-body colour (Tanner Helland), normalised so max channel = 1.
  function kelvin(k) {
    const t = k / 100;
    let r, g, b;
    if (t <= 66) {
      r = 255;
      g = 99.4708025861 * Math.log(t) - 161.1195681661;
      b = t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
    } else {
      r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
      g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
      b = 255;
    }
    const c = [clamp(r, 0, 255) / 255, clamp(g, 0, 255) / 255, clamp(b, 0, 255) / 255];
    const m = Math.max(c[0], c[1], c[2]);
    return [c[0] / m, c[1] / m, c[2] / m];
  }
  const WHITE = [1, 1, 1];
  const N = kelvin(5600); // reference "neutral" light
  // colour relative to neutral daylight, so 5600K renders as white
  const kc = (k) => {
    const c = kelvin(k);
    return [c[0] / N[0], c[1] / N[1], c[2] / N[2]].map((v, _, a) => v / Math.max(a[0], a[1], a[2]));
  };

  /* ---------------------------------------------------------- light rigs */
  // Camera space: +x right, +y up, +z toward the viewer.
  const key = (dir, power = 1, color = WHITE, soft = 0) => ({ type: 'dir', dir: norm3(dir), power, color, soft });
  const flash = (power = 1, color = WHITE, soft = 0, offset = [0.04, 0.08]) => ({
    type: 'flash', dir: norm3([offset[0], offset[1], 1]), power, color, soft,
  });
  const rim = (dir, power = 1, color = WHITE) => ({ type: 'rim', dir: norm3(dir), power, color });

  const NEON_BLUE = [0.25, 0.55, 1];
  const NEON_PINK = [1, 0.25, 0.72];
  const NEON_CYAN = [0.15, 0.95, 1];

  // ambient: base fill level; amb: ambient colour; bg: how strongly the background
  // exposure control applies (negative = background is brightened instead).
  // grade: temp, sat, contrast, fade, grain, vignette, glow, glowTint, exposure,
  //        shadowTint, highlightTint, highlights, shadows.
  const CATEGORY_DEFAULTS = {
    flash: { lights: [flash(1.1)], ambient: 0.3, bg: 0.7, grade: { contrast: 0.12 } },
    outdoor_ccd: { lights: [key([0.3, 0.8, 0.5], 0.9, kc(5500))], ambient: 0.55, bg: 0.1, grade: {} },
    clean_glow: { lights: [key([-0.3, 0.45, 0.85], 0.8, WHITE, 0.4)], ambient: 0.7, bg: 0, grade: { glow: 0.2, contrast: -0.08 } },
    sunset: { lights: [key([0.85, 0.2, 0.45], 1.1, kc(3000))], ambient: 0.45, amb: kc(6500), bg: 0.2, grade: { temp: 0.25, glow: 0.3 } },
    studio: { lights: [key([-0.55, 0.45, 0.7], 1.2)], ambient: 0.22, bg: 0.7, grade: {} },
    retro_analog: { lights: [key([0, 0.4, 0.9], 0.5, WHITE, 0.4)], ambient: 0.8, bg: 0, grade: { grain: 0.22, fade: 0.06 } },
    neon_malam: { lights: [key([-0.9, 0.15, 0.35], 1.4, NEON_BLUE)], ambient: 0.15, bg: 0.6, grade: { sat: 0.2, contrast: 0.15 } },
  };

  const PRESET_RIGS = {
    // FLASH
    'iphone-direct-flash': { lights: [flash(1.25, kc(5800))], ambient: 0.25, bg: 0.9, grade: { contrast: 0.15, temp: -0.05 } },
    'iphone-soft-flash': { lights: [flash(0.8, kc(5400), 0.45)], ambient: 0.55, bg: 0.35, grade: { temp: 0.05 } },
    'iphone-night-flash': { lights: [flash(1.35, kc(6000))], ambient: 0.12, bg: 1, grade: { contrast: 0.2, temp: -0.1, vignette: 0.2 } },
    'iphone-indoor-flash': { lights: [flash(1.0, kc(5600))], ambient: 0.4, amb: kc(3200), bg: 0.6, grade: { temp: 0.08 } },
    'iphone-mirror-selfie-flash': { lights: [flash(1.1, WHITE, 0.1, [0, 0])], ambient: 0.35, bg: 0.6, grade: { vignette: 0.35, glow: 0.25 } },
    'dslr-direct-flash': { lights: [flash(1.3, kc(5500), 0, [0, 0.18])], ambient: 0.2, bg: 0.9, grade: { contrast: 0.2 } },
    'external-speedlight-flash': { lights: [key([0, 0.55, 0.85], 1.2, kc(5500))], ambient: 0.25, bg: 0.7, grade: { contrast: 0.12 } },
    'bounce-flash': { lights: [key([0.2, 0.9, 0.45], 0.9, WHITE, 0.6)], ambient: 0.55, bg: 0.25, grade: {} },
    'off-camera-flash': { lights: [key([-0.75, 0.35, 0.55], 1.3)], ambient: 0.2, bg: 0.7, grade: { contrast: 0.2 } },
    'disposable-camera-90s': { lights: [flash(1.2, kc(5200))], ambient: 0.2, bg: 0.8, grade: { grain: 0.35, vignette: 0.3, temp: 0.15, sat: 0.1, fade: 0.08, contrast: 0.15 } },
    'red-eye-vintage-flash': { lights: [flash(1.25, kc(5000), 0, [0, 0])], ambient: 0.2, bg: 0.85, grade: { grain: 0.3, temp: 0.1, fade: 0.1, vignette: 0.35, sat: -0.1 } },
    'fashion-hard-flash': { lights: [flash(1.4, kc(5800))], ambient: 0.15, bg: 0.7, grade: { contrast: 0.3, sat: 0.1 } },
    'fill-flash': { lights: [flash(0.5, WHITE, 0.5)], ambient: 0.85, bg: 0, grade: { shadows: 0.3 } },

    // OUTDOOR & CCD
    'bright-daylight': { lights: [key([0.3, 0.85, 0.4], 1.0, kc(5500))], ambient: 0.5, amb: kc(7500), bg: 0, grade: { contrast: 0.15, sat: 0.1 } },
    'overcast-soft-light': { lights: [key([0, 1, 0.3], 0.6, WHITE, 0.7)], ambient: 0.75, amb: kc(6800), bg: 0, grade: { contrast: -0.15, sat: -0.1 } },
    'open-shade': { lights: [key([0, 0.4, 0.9], 0.6, kc(7000), 0.6)], ambient: 0.7, amb: kc(7500), bg: 0, grade: { temp: -0.1, contrast: -0.05 } },
    'backlit-outdoor': { lights: [rim([0, 0.5, -0.85], 1.2, kc(5000))], ambient: 0.65, bg: -0.4, grade: { glow: 0.3, fade: 0.05, contrast: -0.1 } },
    'classic-ccd-camera': { lights: [flash(0.6)], ambient: 0.55, bg: 0.2, grade: { contrast: 0.2, sat: 0.15, temp: -0.1, grain: 0.1, highlights: 0.2 } },
    'fujifilm-ccd': { lights: [flash(0.55)], ambient: 0.6, bg: 0.15, grade: { contrast: 0.15, sat: 0.05, temp: -0.05, grain: 0.1, shadowTint: [-0.02, 0.03, 0.01] } },
    'kodak-ccd': { lights: [flash(0.55)], ambient: 0.6, bg: 0.15, grade: { contrast: 0.15, sat: 0.2, temp: 0.15, grain: 0.1 } },
    '2000s-digital-snapshot': { lights: [flash(0.9)], ambient: 0.4, bg: 0.5, grade: { contrast: 0.2, sat: 0.15, temp: -0.05, vignette: 0.1 } },

    // CLEAN & GLOW
    'natural-clean-light': { lights: [key([-0.4, 0.4, 0.8], 0.7, WHITE, 0.4)], ambient: 0.7, bg: 0, grade: {} },
    'soft-glow': { lights: [key([0, 0.5, 0.85], 0.7, WHITE, 0.5)], ambient: 0.7, bg: 0, grade: { glow: 0.45, contrast: -0.15, fade: 0.04 } },
    'window-glow': { lights: [key([-0.9, 0.15, 0.4], 1.1, kc(5800), 0.45)], ambient: 0.4, bg: 0.2, grade: { glow: 0.25 } },
    'beauty-glow': { lights: [key([0, 0.45, 0.9], 0.9, WHITE, 0.5)], ambient: 0.6, bg: 0.1, grade: { glow: 0.35, temp: 0.05, contrast: -0.1 } },
    'creamy-skin-light': { lights: [key([0, 0.35, 0.9], 0.8, kc(5000), 0.6)], ambient: 0.7, bg: 0, grade: { temp: 0.1, sat: -0.05, glow: 0.2, contrast: -0.15 } },
    'high-key-clean': { lights: [key([0, 0.3, 0.95], 0.8, WHITE, 0.6)], ambient: 0.9, bg: -0.6, grade: { exposure: 0.35, contrast: -0.2 } },
    'fresh-morning-light': { lights: [key([0.7, 0.4, 0.55], 0.9, kc(4800), 0.4)], ambient: 0.6, amb: kc(7000), bg: 0, grade: { glow: 0.2 } },

    // SUNSET
    'golden-hour': { lights: [key([0.85, 0.2, 0.45], 1.1, kc(3200))], ambient: 0.45, amb: kc(6500), bg: 0.15, grade: { glow: 0.3, temp: 0.25 } },
    'orange-sunset': { lights: [key([0.85, 0.1, 0.4], 1.2, kc(2600))], ambient: 0.35, amb: kc(6000), bg: 0.2, grade: { temp: 0.35, sat: 0.15, glow: 0.3 } },
    'soft-sunset': { lights: [key([0.7, 0.2, 0.6], 0.8, kc(3300), 0.5)], ambient: 0.6, bg: 0, grade: { temp: 0.2, glow: 0.35, contrast: -0.1, fade: 0.05 } },
    'backlit-golden-hour': { lights: [rim([0.2, 0.3, -0.9], 1.5, kc(3000))], ambient: 0.55, amb: kc(5500), bg: -0.4, grade: { glow: 0.5, temp: 0.2, fade: 0.05 } },
    'blue-hour': { lights: [key([0, 0.6, 0.7], 0.5, kc(9000), 0.5)], ambient: 0.5, amb: kc(10000), bg: 0.3, grade: { temp: -0.35, exposure: -0.2 } },
    'sunset-silhouette': { lights: [rim([0, 0.2, -1], 1.6, kc(2600))], ambient: 0.1, bg: -1.2, grade: { exposure: -0.4, contrast: 0.3, temp: 0.35, glow: 0.4 } },
    'warm-evening-street': { lights: [key([-0.6, 0.3, 0.7], 0.9, kc(2800))], ambient: 0.35, amb: kc(3200), bg: 0.5, grade: { temp: 0.25, vignette: 0.25 } },

    // STUDIO
    'rembrandt-lighting': { lights: [key([-0.7, 0.55, 0.45], 1.3)], ambient: 0.15, bg: 0.8, grade: {} },
    'butterfly-lighting': { lights: [key([0, 0.8, 0.6], 1.2)], ambient: 0.25, bg: 0.7, grade: {} },
    'loop-lighting': { lights: [key([-0.45, 0.5, 0.75], 1.2)], ambient: 0.25, bg: 0.7, grade: {} },
    'split-lighting': { lights: [key([-1, 0.05, 0.08], 1.4)], ambient: 0.08, bg: 0.85, grade: { contrast: 0.1 } },
    'clamshell-lighting': { lights: [key([0, 0.7, 0.7], 1.0, WHITE, 0.4), key([0, -0.6, 0.8], 0.5, WHITE, 0.5)], ambient: 0.35, bg: 0.5, grade: {} },
    'broad-lighting': { lights: [key([0.6, 0.4, 0.7], 1.1)], ambient: 0.3, bg: 0.6, grade: {} },
    'short-lighting': { lights: [key([-0.7, 0.4, 0.6], 1.2)], ambient: 0.18, bg: 0.75, grade: {} },
    'softbox-portrait': { lights: [key([-0.55, 0.45, 0.7], 1.1, WHITE, 0.5)], ambient: 0.35, bg: 0.55, grade: {} },
    'beauty-dish': { lights: [key([0, 0.6, 0.8], 1.2, WHITE, 0.2)], ambient: 0.3, bg: 0.6, grade: { contrast: 0.1 } },
    'ring-light': { lights: [flash(0.9, WHITE, 0.4, [0, 0])], ambient: 0.4, bg: 0.4, grade: { glow: 0.1 } },
    'three-point-lighting': { lights: [key([-0.6, 0.45, 0.65], 1.1), key([0.6, 0.2, 0.75], 0.4, WHITE, 0.5), rim([0.5, 0.4, -0.8], 0.9)], ambient: 0.2, bg: 0.6, grade: {} },
    'rim-light': { lights: [rim([-0.6, 0.3, -0.75], 1.0), rim([0.6, 0.3, -0.75], 1.0)], ambient: 0.3, bg: 0.8, grade: {} },
    'low-key-studio': { lights: [key([-0.7, 0.5, 0.5], 1.3)], ambient: 0.07, bg: 1, grade: { contrast: 0.2, exposure: -0.2, vignette: 0.3 } },
    'high-key-studio': { lights: [key([0, 0.3, 0.95], 0.9, WHITE, 0.6)], ambient: 0.85, bg: -1, grade: { exposure: 0.4, contrast: -0.15 } },

    // RETRO & ANALOG
    'kodak-gold-200': { grade: { temp: 0.25, sat: 0.15, contrast: 0.1, grain: 0.25, fade: 0.06, highlightTint: [0.03, 0.01, -0.03] } },
    'kodak-portra-400': { grade: { temp: 0.12, sat: -0.05, contrast: -0.1, grain: 0.2, fade: 0.05 } },
    'kodak-portra-800': { grade: { temp: 0.15, grain: 0.32, fade: 0.06, sat: 0.05 } },
    'kodak-ektar-100': { grade: { sat: 0.35, contrast: 0.2, temp: 0.05, grain: 0.1 } },
    'fujifilm-superia-400': { grade: { temp: -0.05, sat: 0.15, grain: 0.25, shadowTint: [-0.02, 0.035, 0.0] } },
    'fujifilm-pro-400h': { grade: { temp: -0.08, sat: -0.1, contrast: -0.15, fade: 0.08, grain: 0.2, shadowTint: [-0.02, 0.01, 0.03] } },
    'cinestill-800t': { lights: [key([0.3, 0.5, 0.8], 0.6, kc(3200), 0.4)], ambient: 0.6, bg: 0.3, grade: { temp: -0.35, glow: 0.5, glowTint: [1, 0.25, 0.15], grain: 0.3, contrast: 0.1 } },
    'disposable-film-camera': { lights: [flash(1.1, kc(5200))], ambient: 0.3, bg: 0.7, grade: { grain: 0.4, vignette: 0.35, temp: 0.15, fade: 0.1 } },
    'polaroid-instant-film': { grade: { fade: 0.15, contrast: -0.15, temp: 0.1, sat: -0.15, vignette: 0.3, grain: 0.15, shadowTint: [-0.02, 0.01, 0.03] } },

    // NEON & MALAM
    'neon-blue': { lights: [key([-0.9, 0.15, 0.35], 1.5, NEON_BLUE)], ambient: 0.15, amb: [0.6, 0.7, 1], bg: 0.6, grade: { sat: 0.2, contrast: 0.15 } },
    'neon-pink': { lights: [key([0.9, 0.15, 0.35], 1.5, NEON_PINK)], ambient: 0.15, amb: [1, 0.7, 0.9], bg: 0.6, grade: { sat: 0.2, contrast: 0.15 } },
    'cyberpunk-night': { lights: [key([-1, 0.1, 0.25], 1.5, NEON_PINK), key([1, 0.1, 0.25], 1.5, NEON_CYAN)], ambient: 0.08, amb: [0.5, 0.4, 0.9], bg: 0.8, grade: { contrast: 0.2, glow: 0.3, sat: 0.2 } },
    'night-street-flash': { lights: [flash(1.2, kc(5800))], ambient: 0.15, amb: kc(3000), bg: 0.85, grade: { grain: 0.15, vignette: 0.2 } },
    'city-night-ambient': { lights: [key([0.3, 0.6, 0.7], 0.6, kc(3200), 0.5)], ambient: 0.35, amb: kc(3500), bg: 0.4, grade: { exposure: -0.25, glow: 0.35, vignette: 0.2, temp: 0.05 } },
    'neon-rim-light': { lights: [rim([-0.7, 0.2, -0.7], 1.3, NEON_CYAN), rim([0.7, 0.2, -0.7], 1.3, NEON_PINK)], ambient: 0.25, bg: 0.7, grade: { contrast: 0.15 } },
    'moody-night-portrait': { lights: [key([-0.7, 0.4, 0.6], 1.0, kc(3000))], ambient: 0.1, amb: kc(9000), bg: 0.9, grade: { exposure: -0.3, vignette: 0.35, contrast: 0.15 } },
  };

  function getRig(preset) {
    const base = CATEGORY_DEFAULTS[preset.category] || CATEGORY_DEFAULTS.studio;
    const own = PRESET_RIGS[preset.id] || {};
    return {
      lights: own.lights || base.lights,
      ambient: own.ambient != null ? own.ambient : base.ambient,
      amb: own.amb || base.amb || WHITE,
      bg: own.bg != null ? own.bg : base.bg,
      grade: Object.assign({}, base.grade, own.grade),
    };
  }

  /* --------------------------------------------------------- depth model */
  let pipePromise = null;
  function getDepthPipeline(onStatus) {
    if (!pipePromise) {
      pipePromise = (async () => {
        onStatus && onStatus('Loading on-device AI…');
        const tf = await import(TRANSFORMERS_URL);
        tf.env.allowLocalModels = false;
        const progress_callback = (p) => {
          if (p && p.status === 'progress' && typeof p.progress === 'number' && /onnx/.test(p.file || '')) {
            onStatus && onStatus(`Downloading depth model (first run only)… ${Math.round(p.progress)}%`);
          }
        };
        let device = 'wasm';
        try {
          if (navigator.gpu && (await navigator.gpu.requestAdapter())) device = 'webgpu';
        } catch (_) { /* wasm fallback */ }
        try {
          return await tf.pipeline('depth-estimation', DEPTH_MODEL, {
            device,
            dtype: device === 'webgpu' ? 'fp16' : 'q8',
            progress_callback,
          });
        } catch (err) {
          if (device === 'wasm') throw err;
          return await tf.pipeline('depth-estimation', DEPTH_MODEL, { device: 'wasm', dtype: 'q8', progress_callback });
        }
      })().catch((err) => {
        pipePromise = null;
        throw err;
      });
    }
    return pipePromise;
  }

  function canvasToBlob(canvas, type, quality) {
    return new Promise((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('Canvas encoding failed.'))), type, quality)
    );
  }

  async function estimateDepth(drawable, w, h, onStatus) {
    const pipe = await getDepthPipeline(onStatus);
    onStatus && onStatus('Estimating scene depth…');
    const s = Math.min(1, DEPTH_INPUT_EDGE / Math.max(w, h));
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w * s));
    c.height = Math.max(1, Math.round(h * s));
    c.getContext('2d').drawImage(drawable, 0, 0, c.width, c.height);
    const url = URL.createObjectURL(await canvasToBlob(c, 'image/png'));
    let out;
    try {
      out = await pipe(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    if (Array.isArray(out)) out = out[0];

    let data, dw, dh;
    const t = out.predicted_depth;
    if (t && t.dims && t.data) {
      dh = t.dims[t.dims.length - 2];
      dw = t.dims[t.dims.length - 1];
      data = Float32Array.from(t.data);
    } else {
      const img = out.depth;
      dw = img.width;
      dh = img.height;
      data = new Float32Array(dw * dh);
      const ch = img.channels || 1;
      for (let i = 0; i < dw * dh; i++) data[i] = img.data[i * ch];
    }

    // Robust normalisation to 0 (far) .. 1 (near) using 2nd / 98th percentiles.
    const sample = [];
    const step = Math.max(1, Math.floor(data.length / 20000));
    for (let i = 0; i < data.length; i += step) sample.push(data[i]);
    sample.sort((a, b) => a - b);
    const lo = sample[Math.floor(sample.length * 0.02)];
    const hi = sample[Math.floor(sample.length * 0.98)];
    const range = hi - lo || 1;
    for (let i = 0; i < data.length; i++) data[i] = clamp((data[i] - lo) / range, 0, 1);
    return { data, w: dw, h: dh };
  }

  function boxBlur(src, w, h, r) {
    if (r < 1) return src;
    const tmp = new Float32Array(src.length);
    const out = new Float32Array(src.length);
    const pass = (a, b, horiz) => {
      const len = horiz ? w : h;
      const lines = horiz ? h : w;
      for (let l = 0; l < lines; l++) {
        let acc = 0;
        const idx = (i) => (horiz ? l * w + i : i * w + l);
        for (let i = -r; i <= r; i++) acc += a[idx(clamp(i, 0, len - 1))];
        for (let i = 0; i < len; i++) {
          b[idx(i)] = acc / (2 * r + 1);
          acc += a[idx(clamp(i + r + 1, 0, len - 1))] - a[idx(clamp(i - r, 0, len - 1))];
        }
      }
    };
    pass(src, tmp, true);
    pass(tmp, out, false);
    return out;
  }

  /* --------------------------------------------------------- light map */
  function buildLightMap(depth, rig, adj) {
    const { w, h } = depth;
    const style = adj.shadowStyle || 'Balanced';
    const softness = style === 'Diffused' ? 1 : style === 'Crisp Hard' ? 0 : 0.5;
    const blurR = Math.round((3 + softness * 6) * Math.max(w, h) / 640);
    const d = boxBlur(boxBlur(depth.data, w, h, blurR), w, h, Math.max(1, blurR >> 1));

    const intensity = clamp((adj.lightingIntensity != null ? adj.lightingIntensity : 75) / 75, 0, 1.5);
    const shadowI = adj.shadowIntensity != null ? adj.shadowIntensity : 50;
    const ambient = clamp(rig.ambient * (1.6 - (shadowI / 100) * 1.2), 0.02, 1.2);
    const bgEV = (adj.backgroundDarkness != null ? adj.backgroundDarkness : -1) * rig.bg;
    const k = Math.max(w, h) * 0.6; // normal strength

    const L = new Float32Array(w * h * 3);
    const fg = new Float32Array(w * h);
    let sumLum = 0;
    let sumW = 0;

    for (let y = 0; y < h; y++) {
      const y0 = y > 0 ? y - 1 : y;
      const y1 = y < h - 1 ? y + 1 : y;
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        const x0 = x > 0 ? x - 1 : x;
        const x1 = x < w - 1 ? x + 1 : x;
        const dx = (d[y * w + x1] - d[y * w + x0]) / (x1 - x0 || 1);
        const dy = (d[y1 * w + x] - d[y0 * w + x]) / (y1 - y0 || 1);
        let nx = -dx * k;
        let ny = dy * k; // image y points down, camera y points up
        let nz = 1;
        const nl = Math.hypot(nx, ny, nz);
        nx /= nl; ny /= nl; nz /= nl;

        const z = d[i];
        const near = smooth(0.3, 0.7, z);
        fg[i] = near;

        let r = 0, g = 0, b = 0;
        for (let j = 0; j < rig.lights.length; j++) {
          const lt = rig.lights[j];
          const [lx, ly, lz] = lt.dir;
          let s;
          if (lt.type === 'rim') {
            const side = nx * lx + ny * ly;
            s = Math.max(0, side + 0.15) * Math.pow(1 - nz, 1.3) * 2.2 * (0.35 + 0.65 * near);
          } else {
            const wrap = 0.1 + 0.45 * softness + (lt.soft || 0) * 0.5;
            const ndl = nx * lx + ny * ly + nz * lz;
            s = Math.max(0, (ndl + wrap) / (1 + wrap));
            s = Math.pow(s, 1.6 - 0.8 * softness);
            if (lt.type === 'flash') s *= 0.12 + 0.88 * Math.pow(z, 1.6); // distance falloff
          }
          const p = lt.power * s * intensity;
          r += lt.color[0] * p;
          g += lt.color[1] * p;
          b += lt.color[2] * p;
        }
        r += ambient * rig.amb[0];
        g += ambient * rig.amb[1];
        b += ambient * rig.amb[2];

        L[i * 3] = r; L[i * 3 + 1] = g; L[i * 3 + 2] = b;
        const wgt = 0.15 + near;
        sumLum += (0.2126 * r + 0.7152 * g + 0.0722 * b) * wgt;
        sumW += wgt;
      }
    }

    // Keep the subject's overall exposure stable, then shape background exposure by depth.
    const norm = sumW > 0 && sumLum > 0 ? sumW / sumLum : 1;
    const mix = clamp(intensity, 0, 1);
    for (let i = 0; i < w * h; i++) {
      const bgMul = Math.pow(2, bgEV * (1 - fg[i]));
      for (let c = 0; c < 3; c++) {
        const v = L[i * 3 + c] * norm;
        L[i * 3 + c] = clamp((1 + (v - 1) * (0.35 + 0.65 * mix) + 0) * bgMul, 0, 4);
      }
    }
    return L;
  }

  /* ----------------------------------------------------------- render */
  const S2L = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    S2L[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  }
  const l2s = (v) => (v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055);
  const shoulder = (v) => (v > 0.8 ? 0.8 + 0.2 * (1 - Math.exp(-(v - 0.8) / 0.2)) : v);

  function hash(x, y) {
    let n = (x * 374761393 + y * 668265263) | 0;
    n = Math.imul(n ^ (n >>> 13), 1274126177);
    return ((n ^ (n >>> 16)) >>> 0) / 4294967296;
  }

  async function run(originalFullRes, preset, adjustments, onStatus) {
    const adj = Object.assign({}, adjustments || {});
    const rig = getRig(preset);
    const g = rig.grade;

    const decoded = await originalFullRes.drawableSource();
    try {
      const depth = await estimateDepth(decoded.drawable, decoded.width, decoded.height, onStatus);
      onStatus && onStatus('Relighting with virtual light rig…');
      await new Promise((r) => setTimeout(r, 30));

      const L = buildLightMap(depth, rig, adj);

      const s = Math.min(1, WORK_MAX_EDGE / Math.max(decoded.width, decoded.height));
      const W = Math.max(1, Math.round(decoded.width * s));
      const H = Math.max(1, Math.round(decoded.height * s));
      const canvas = document.createElement('canvas');
      canvas.width = W;
      canvas.height = H;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(decoded.drawable, 0, 0, W, H);
      const img = ctx.getImageData(0, 0, W, H);
      const px = img.data;

      const exposure = Math.pow(2, (adj.exposure || 0) + (g.exposure || 0));
      const temp = (g.temp || 0) + (adj.colorTemperature || 0) / 100;
      const sat = (g.sat || 0) + (adj.saturation || 0) / 100;
      const contrast = (g.contrast || 0) + (adj.contrast || 0) / 200;
      const hiAdj = (g.highlights || 0) + (adj.highlights || 0) / 100;
      const shAdj = (g.shadows || 0) + (adj.shadows || 0) / 100;
      const fade = g.fade || 0;
      const grain = g.grain || 0;
      const vig = g.vignette || 0;
      const sT = g.shadowTint || [0, 0, 0];
      const hT = g.highlightTint || [0, 0, 0];
      const tr = 1 + 0.12 * temp;
      const tg = 1 + 0.02 * temp;
      const tb = 1 - 0.14 * temp;

      const { w: lw, h: lh } = depth;
      const sx = lw / W;
      const sy = lh / H;
      const cx = W / 2;
      const cy = H / 2;
      const rMax = Math.hypot(cx, cy);

      for (let y = 0; y < H; y++) {
        const fy = clamp((y + 0.5) * sy - 0.5, 0, lh - 1);
        const y0 = Math.floor(fy);
        const y1 = Math.min(lh - 1, y0 + 1);
        const ty = fy - y0;
        for (let x = 0; x < W; x++) {
          const fx = clamp((x + 0.5) * sx - 0.5, 0, lw - 1);
          const x0 = Math.floor(fx);
          const x1 = Math.min(lw - 1, x0 + 1);
          const tx = fx - x0;
          const a = (y0 * lw + x0) * 3, b = (y0 * lw + x1) * 3, c = (y1 * lw + x0) * 3, e = (y1 * lw + x1) * 3;
          const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;

          const o = (y * W + x) * 4;
          let r = S2L[px[o]] * (L[a] * w00 + L[b] * w01 + L[c] * w10 + L[e] * w11) * exposure;
          let gg = S2L[px[o + 1]] * (L[a + 1] * w00 + L[b + 1] * w01 + L[c + 1] * w10 + L[e + 1] * w11) * exposure;
          let bb = S2L[px[o + 2]] * (L[a + 2] * w00 + L[b + 2] * w01 + L[c + 2] * w10 + L[e + 2] * w11) * exposure;

          r = l2s(shoulder(r)) * tr;
          gg = l2s(shoulder(gg)) * tg;
          bb = l2s(shoulder(bb)) * tb;

          let lum = 0.2126 * r + 0.7152 * gg + 0.0722 * bb;
          const hiW = smooth(0.45, 1, lum);
          const shW = 1 - smooth(0, 0.5, lum);
          const delta = hiAdj * 0.3 * hiW + shAdj * 0.3 * shW;
          r += delta + sT[0] * shW + hT[0] * hiW;
          gg += delta + sT[1] * shW + hT[1] * hiW;
          bb += delta + sT[2] * shW + hT[2] * hiW;

          if (contrast) {
            r = (r - 0.5) * (1 + contrast) + 0.5;
            gg = (gg - 0.5) * (1 + contrast) + 0.5;
            bb = (bb - 0.5) * (1 + contrast) + 0.5;
          }
          if (sat) {
            lum = 0.2126 * r + 0.7152 * gg + 0.0722 * bb;
            r = lum + (r - lum) * (1 + sat);
            gg = lum + (gg - lum) * (1 + sat);
            bb = lum + (bb - lum) * (1 + sat);
          }
          if (fade) {
            r = fade + r * (1 - fade);
            gg = fade + gg * (1 - fade);
            bb = fade + bb * (1 - fade);
          }
          if (vig) {
            const rr = Math.hypot(x - cx, y - cy) / rMax;
            const m = 1 - vig * rr * rr * 0.9;
            r *= m; gg *= m; bb *= m;
          }
          if (grain) {
            const n = (hash(x, y) - 0.5) * grain * 0.12;
            r += n; gg += n; bb += n;
          }
          px[o] = r * 255;
          px[o + 1] = gg * 255;
          px[o + 2] = bb * 255;
        }
      }
      ctx.putImageData(img, 0, 0);

      const glow = g.glow || 0;
      if (glow > 0) {
        const gc = document.createElement('canvas');
        const gs = 0.25;
        gc.width = Math.max(1, Math.round(W * gs));
        gc.height = Math.max(1, Math.round(H * gs));
        const gx = gc.getContext('2d');
        gx.filter = `blur(${Math.max(2, Math.round(Math.max(gc.width, gc.height) / 60))}px) brightness(0.85) contrast(1.8)`;
        gx.drawImage(canvas, 0, 0, gc.width, gc.height);
        gx.filter = 'none';
        if (g.glowTint) {
          gx.globalCompositeOperation = 'multiply';
          gx.fillStyle = `rgb(${g.glowTint.map((v) => Math.round(v * 255)).join(',')})`;
          gx.fillRect(0, 0, gc.width, gc.height);
        }
        ctx.save();
        ctx.globalCompositeOperation = 'screen';
        ctx.globalAlpha = clamp(glow * 0.6, 0, 0.6);
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(gc, 0, 0, W, H);
        ctx.restore();
      }

      const blob = await canvasToBlob(canvas, 'image/jpeg', 0.94);
      canvas.width = 1;
      canvas.height = 1;
      return { blob, mimeType: 'image/jpeg', width: W, height: H };
    } finally {
      decoded.cleanup();
    }
  }

  window.LocalRelight = { run, getRig, preload: getDepthPipeline };
})();
