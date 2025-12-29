import React, { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Download, RefreshCcw, Sparkles, Droplets, Info } from "lucide-react";

/**
 * Seeded SVG Splotch Generator
 *
 * - Deterministically simulates many small fluid "packets" launched towards a surface
 * - Uses ballistic + drag + restitution + surface-flow advection + radial smear to deposit paint
 * - Rasterizes deposits into a density field
 * - Extracts vector boundary using marching squares and smooths to SVG path
 * - Exports an SVG file containing the resulting blob shape
 *
 * Manual placement always:
 * - Output can be clipped by the export viewBox. Use Pan/Scale controls to frame it.
 */

// ------------------------- Seeded RNG ----------------------------------------

function xmur3(str: string) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function () {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return h >>> 0;
  };
}

function sfc32(a: number, b: number, c: number, d: number) {
  return function () {
    a >>>= 0;
    b >>>= 0;
    c >>>= 0;
    d >>>= 0;
    let t = (a + b) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    d = (d + 1) | 0;
    t = (t + d) | 0;
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
}

function makeRng(seed: string) {
  const seedFn = xmur3(seed);
  const a = seedFn();
  const b = seedFn();
  const c = seedFn();
  const d = seedFn();
  const rand = sfc32(a, b, c, d);
  const float = () => rand();
  const range = (min: number, max: number) => min + (max - min) * float();
  const normal = () => {
    // Box–Muller
    let u = 0,
      v = 0;
    while (u === 0) u = float();
    while (v === 0) v = float();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };
  return { float, range, normal };
}

// ------------------------- Math helpers --------------------------------------

type Vec2 = { x: number; y: number };

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const length = (v: Vec2) => Math.hypot(v.x, v.y);
const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y });
const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
const mul = (v: Vec2, s: number): Vec2 => ({ x: v.x * s, y: v.y * s });
const norm = (v: Vec2): Vec2 => {
  const L = length(v);
  if (L < 1e-9) return { x: 0, y: 0 };
  return { x: v.x / L, y: v.y / L };
};
const rot = (v: Vec2, a: number): Vec2 => ({
  x: v.x * Math.cos(a) - v.y * Math.sin(a),
  y: v.x * Math.sin(a) + v.y * Math.cos(a),
});

// ------------------------- Density field -------------------------------------

class Field {
  w: number;
  h: number;
  data: Float32Array;
  constructor(w: number, h: number) {
    this.w = w;
    this.h = h;
    this.data = new Float32Array(w * h);
  }
  idx(x: number, y: number) {
    return y * this.w + x;
  }
  get(x: number, y: number) {
    x = clamp(x, 0, this.w - 1);
    y = clamp(y, 0, this.h - 1);
    return this.data[this.idx(x, y)];
  }
  add(x: number, y: number, v: number) {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return;
    this.data[this.idx(x, y)] += v;
  }
  max() {
    let m = 0;
    for (let i = 0; i < this.data.length; i++) m = Math.max(m, this.data[i]);
    return m;
  }
}

function depositGaussian(field: Field, p: Vec2, radius: number, amount: number) {
  const r = Math.max(1, Math.floor(radius));
  const x0 = Math.floor(p.x);
  const y0 = Math.floor(p.y);
  const sigma2 = (radius * radius) / 2;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const x = x0 + dx;
      const y = y0 + dy;
      const dd = dx * dx + dy * dy;
      const w = Math.exp(-dd / Math.max(1e-6, sigma2));
      field.add(x, y, amount * w);
    }
  }
}

function depositAniso(field: Field, p: Vec2, major: number, minor: number, angle: number, amount: number) {
  // Elliptical gaussian aligned with angle.
  const R = Math.max(1, Math.floor(Math.max(major, minor)));
  const x0 = Math.floor(p.x);
  const y0 = Math.floor(p.y);
  const ca = Math.cos(angle);
  const sa = Math.sin(angle);
  const sMajor2 = (major * major) / 2;
  const sMinor2 = (minor * minor) / 2;
  for (let dy = -R; dy <= R; dy++) {
    for (let dx = -R; dx <= R; dx++) {
      const x = x0 + dx;
      const y = y0 + dy;
      const xr = dx * ca + dy * sa;
      const yr = -dx * sa + dy * ca;
      const e = (xr * xr) / Math.max(1e-6, sMajor2) + (yr * yr) / Math.max(1e-6, sMinor2);
      const w = Math.exp(-e);
      field.add(x, y, amount * w);
    }
  }
}

function blurField(field: Field, passes: number) {
  const w = field.w,
    h = field.h;
  const tmp = new Float32Array(w * h);
  const k = [1, 2, 1];
  const ksum = 4;
  for (let pass = 0; pass < passes; pass++) {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let i = -1; i <= 1; i++) {
          const xx = clamp(x + i, 0, w - 1);
          s += field.data[y * w + xx] * k[i + 1];
        }
        tmp[y * w + x] = s / ksum;
      }
    }
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let s = 0;
        for (let i = -1; i <= 1; i++) {
          const yy = clamp(y + i, 0, h - 1);
          s += tmp[yy * w + x] * k[i + 1];
        }
        field.data[y * w + x] = s / ksum;
      }
    }
  }
}

// ------------------------- Marching squares ----------------------------------

type Poly = Vec2[];

function marchingSquares(field: Field, threshold: number): Poly[] {
  const w = field.w,
    h = field.h;

  type Seg = { a: Vec2; b: Vec2 };
  const segs: Seg[] = [];

  function interp(p1: Vec2, p2: Vec2, v1: number, v2: number) {
    const t = (threshold - v1) / (v2 - v1 + 1e-12);
    return { x: lerp(p1.x, p2.x, t), y: lerp(p1.y, p2.y, t) };
  }

  for (let y = 0; y < h - 1; y++) {
    for (let x = 0; x < w - 1; x++) {
      const v00 = field.get(x, y);
      const v10 = field.get(x + 1, y);
      const v11 = field.get(x + 1, y + 1);
      const v01 = field.get(x, y + 1);

      let idx = 0;
      if (v00 >= threshold) idx |= 1;
      if (v10 >= threshold) idx |= 2;
      if (v11 >= threshold) idx |= 4;
      if (v01 >= threshold) idx |= 8;
      if (idx === 0 || idx === 15) continue;

      const p00 = { x, y };
      const p10 = { x: x + 1, y };
      const p11 = { x: x + 1, y: y + 1 };
      const p01 = { x, y: y + 1 };

      const e0 = interp(p00, p10, v00, v10);
      const e1 = interp(p10, p11, v10, v11);
      const e2 = interp(p01, p11, v01, v11);
      const e3 = interp(p00, p01, v00, v01);

      const center = (v00 + v10 + v11 + v01) / 4;
      const connectA = center >= threshold;

      const addSeg = (a: Vec2, b: Vec2) => segs.push({ a, b });

      switch (idx) {
        case 1:
        case 14:
          addSeg(e3, e0);
          break;
        case 2:
        case 13:
          addSeg(e0, e1);
          break;
        case 3:
        case 12:
          addSeg(e3, e1);
          break;
        case 4:
        case 11:
          addSeg(e1, e2);
          break;
        case 6:
        case 9:
          addSeg(e0, e2);
          break;
        case 7:
        case 8:
          addSeg(e3, e2);
          break;
        case 5:
          if (connectA) {
            addSeg(e3, e2);
            addSeg(e0, e1);
          } else {
            addSeg(e3, e0);
            addSeg(e2, e1);
          }
          break;
        case 10:
          if (connectA) {
            addSeg(e3, e0);
            addSeg(e2, e1);
          } else {
            addSeg(e3, e2);
            addSeg(e0, e1);
          }
          break;
        default:
          break;
      }
    }
  }

  const eps = 1e-3;
  const key = (p: Vec2) => `${Math.round(p.x / eps)}:${Math.round(p.y / eps)}`;
  const segMap = new Map<string, Seg[]>();

  for (const s of segs) {
    const ka = key(s.a);
    const kb = key(s.b);
    if (!segMap.has(ka)) segMap.set(ka, []);
    if (!segMap.has(kb)) segMap.set(kb, []);
    segMap.get(ka)!.push(s);
    segMap.get(kb)!.push(s);
  }

  const used = new Set<Seg>();
  const polys: Poly[] = [];

  function nextFrom(p: Vec2): Seg | null {
    const kp = key(p);
    const list = segMap.get(kp) || [];
    for (const s of list) {
      if (used.has(s)) continue;
      const da = Math.hypot(s.a.x - p.x, s.a.y - p.y);
      const db = Math.hypot(s.b.x - p.x, s.b.y - p.y);
      if (da < 2 * eps || db < 2 * eps) return s;
    }
    return null;
  }

  function otherEnd(seg: Seg, p: Vec2): Vec2 {
    const da = Math.hypot(seg.a.x - p.x, seg.a.y - p.y);
    const db = Math.hypot(seg.b.x - p.x, seg.b.y - p.y);
    return da < db ? seg.b : seg.a;
  }

  for (const s0 of segs) {
    if (used.has(s0)) continue;
    used.add(s0);

    const poly: Vec2[] = [s0.a, s0.b];
    let cur = s0.b;

    for (let guard = 0; guard < 20000; guard++) {
      const nxt = nextFrom(cur);
      if (!nxt) break;
      used.add(nxt);
      const other = otherEnd(nxt, cur);
      poly.push(other);
      cur = other;
      if (Math.hypot(cur.x - poly[0].x, cur.y - poly[0].y) < 2 * eps) {
        poly[poly.length - 1] = { ...poly[0] };
        break;
      }
    }

    if (poly.length > 10) polys.push(poly);
  }

  return polys;
}

function chaikinSmooth(poly: Vec2[], iterations: number) {
  const closed = Math.hypot(poly[0].x - poly[poly.length - 1].x, poly[0].y - poly[poly.length - 1].y) < 1e-6;
  let pts = poly.slice(0, closed ? poly.length - 1 : poly.length);
  for (let it = 0; it < iterations; it++) {
    const out: Vec2[] = [];
    for (let i = 0; i < pts.length; i++) {
      const p0 = pts[i];
      const p1 = pts[(i + 1) % pts.length];
      const q = { x: lerp(p0.x, p1.x, 0.25), y: lerp(p0.y, p1.y, 0.25) };
      const r = { x: lerp(p0.x, p1.x, 0.75), y: lerp(p0.y, p1.y, 0.75) };
      out.push(q, r);
    }
    pts = out;
  }
  if (closed) pts.push({ ...pts[0] });
  return pts;
}

function polyArea(poly: Vec2[]) {
  let a = 0;
  for (let i = 0; i < poly.length - 1; i++) {
    a += poly[i].x * poly[i + 1].y - poly[i + 1].x * poly[i].y;
  }
  return a / 2;
}

function polyCentroid(poly: Vec2[]) {
  let cx = 0,
    cy = 0;
  let a = 0;
  for (let i = 0; i < poly.length - 1; i++) {
    const x0 = poly[i].x,
      y0 = poly[i].y;
    const x1 = poly[i + 1].x,
      y1 = poly[i + 1].y;
    const cross = x0 * y1 - x1 * y0;
    a += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  a *= 0.5;
  if (Math.abs(a) < 1e-9) {
    let sx = 0,
      sy = 0;
    const n = Math.max(1, poly.length - 1);
    for (let i = 0; i < n; i++) {
      sx += poly[i].x;
      sy += poly[i].y;
    }
    return { x: sx / n, y: sy / n };
  }
  return { x: cx / (6 * a), y: cy / (6 * a) };
}

function pointInPoly(p: Vec2, poly: Vec2[]) {
  let inside = false;
  for (let i = 0, j = poly.length - 2; i < poly.length - 1; j = i++) {
    const xi = poly[i].x,
      yi = poly[i].y;
    const xj = poly[j].x,
      yj = poly[j].y;
    const intersect = yi > p.y !== yj > p.y && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi + 1e-12) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function toSvgPath(poly: Vec2[], scale: number, offset: Vec2) {
  const pts = poly.map((p) => ({ x: p.x * scale + offset.x, y: p.y * scale + offset.y }));
  const cmds = [`M ${pts[0].x.toFixed(2)} ${pts[0].y.toFixed(2)}`];
  for (let i = 1; i < pts.length; i++) cmds.push(`L ${pts[i].x.toFixed(2)} ${pts[i].y.toFixed(2)}`);
  cmds.push("Z");
  return cmds.join(" ");
}

// ------------------------- Splat simulation ----------------------------------

type Geometry = "circle" | "line" | "spray" | "fling";

type Params = {
  seed: string;
  svgSize: number;
  fieldSize: number;
  geometry: Geometry;

  packets: number;
  baseRadius: number;
  radiusJitter: number;

  viscosity: number;
  restitution: number;
  drag: number;
  impactSpread: number;
  smear: number;
  noise: number;

  blur: number;
  threshold: number;
  smooth: number;

  sprayAngleDeg: number;
  sprayMagnitude: number;
  sprayCovariance: number;

  strokes: number;
  flingPower: number;
  directionality: number;
  anisotropy: number;
  tail: number;
  tailDroplets: number;

  // Manual placement (always enabled)
  panX: number;
  panY: number;
  userScale: number;

  invertY: boolean;
};

function simulate(params: Params) {
  const rng = makeRng(params.seed);

  // Use a larger internal field for simulation to prevent clipping.
  const displayW = params.fieldSize;
  const workingW = displayW * 7; 
  const field = new Field(workingW, workingW);
  
  // Physics units are based on displayW, so the splotch stays "normal size"
  // inside the huge workingW buffer.
  const physW = displayW;
  
  // For spray geometry, offset the center opposite to spray direction
  // to compensate for the drift caused by spray magnitude
  let center = { x: workingW / 2, y: workingW / 2 };
  if (params.geometry === "spray" && params.sprayMagnitude > 0) {
    const ang = (params.sprayAngleDeg * Math.PI) / 180;
    // Offset opposite to spray direction, scaled by magnitude and physics size
    const offsetAmount = params.sprayMagnitude * physW * 0.35;
    center = {
      x: workingW / 2 - Math.cos(ang) * offsetAmount,
      y: workingW / 2 - Math.sin(ang) * offsetAmount
    };
  } 

  function sampleSource(): { p: Vec2; v: Vec2; vz: number } {
    const speed = rng.range(0.6, 1.2);
    const vz = rng.range(1.0, 1.8);

    if (params.geometry === "circle") {
      const ang = rng.range(0, Math.PI * 2);
      const r = Math.abs(rng.normal()) * (physW * 0.06);
      const p = add(center, { x: Math.cos(ang) * r, y: Math.sin(ang) * r });
      const vdir = rot({ x: 1, y: 0 }, rng.range(0, Math.PI * 2));
      const v = mul(vdir, speed);
      return { p, v, vz };
    }

    if (params.geometry === "line") {
      const t = rng.range(-0.5, 0.5);
      const lineLen = physW * 0.22;
      const p = add(center, { x: t * lineLen, y: rng.normal() * (physW * 0.02) });
      const v = { x: speed * rng.range(0.6, 1.3), y: rng.normal() * 0.2 };
      return { p, v, vz };
    }

    if (params.geometry === "spray") {
      const baseStd = physW * 0.085;
      const ang = (params.sprayAngleDeg * Math.PI) / 180;
      const dir = { x: Math.cos(ang), y: Math.sin(ang) };
      const perp = { x: -dir.y, y: dir.x };
      const cov = clamp(params.sprayCovariance, 0, 1);

      const alongStd = baseStd * (1 + 2.2 * cov);
      const perpStd = baseStd * (1 - 0.55 * cov);
      const meanShift = params.sprayMagnitude * (physW * 0.14);

      const along = rng.normal() * alongStd + meanShift * (0.25 + 0.75 * rng.float());
      const across = rng.normal() * perpStd;
      const p = add(center, add(mul(dir, along), mul(perp, across)));

      const drift = params.sprayMagnitude * 2.2;
      const v = add(
        mul(dir, drift + rng.normal() * speed * (0.55 + 0.65 * cov)),
        mul(perp, rng.normal() * speed * (1.05 - 0.7 * cov))
      );
      return { p, v, vz };
    }

    // fling fallback (real fling logic below)
    const p = add(center, { x: rng.normal() * (physW * 0.05), y: rng.normal() * (physW * 0.05) });
    const v = { x: rng.normal() * speed * 2.0, y: rng.normal() * speed * 2.0 };
    return { p, v, vz };
  }

  const flowAngle = rng.range(0, Math.PI * 2);
  const flowDir = { x: Math.cos(flowAngle), y: Math.sin(flowAngle) };
  const swirl = rng.range(-0.8, 0.8);

  function surfaceFlow(p: Vec2): Vec2 {
    const u = (p.x - center.x) / (physW * 0.5);
    const v = (p.y - center.y) / (physW * 0.5);
    const radial = norm({ x: u, y: v });
    const tang = { x: -radial.y, y: radial.x };
    const f1 = mul(flowDir, 0.65);
    const f2 = mul(tang, swirl * 0.55);
    const f3 = mul(radial, -0.25);
    return add(add(f1, f2), f3);
  }

  const steps = params.geometry === "fling" ? 34 : 22;
  const dt = 1 / steps;

  const strokeCount = params.geometry === "fling" ? Math.max(1, Math.floor(params.strokes)) : 1;
  const packetsPerStroke = Math.max(1, Math.floor(params.packets / strokeCount));

  function sampleAngle(mu: number) {
    const k = clamp(params.directionality, 0, 1);
    if (k < 1e-6) return rng.range(0, Math.PI * 2);
    const sigma = lerp(1.35, 0.08, k);
    return mu + rng.normal() * sigma;
  }

  function sampleBrushOrigin(dir: Vec2) {
    const perp = { x: -dir.y, y: dir.x };
    const brushWidth = physW * 0.10;
    const along = physW * 0.10;
    const u = rng.range(-0.5, 0.5);
    const v = rng.range(-0.5, 0.5);
    return add(center, add(mul(perp, u * brushWidth), mul(dir, v * along)));
  }

  for (let i = 0; i < params.packets; i++) {
    const strokeIdx = params.geometry === "fling" ? Math.floor(i / packetsPerStroke) : 0;

    let src = sampleSource();
    if (params.geometry === "fling") {
      const baseAng = (xmur3(`${params.seed}::stroke::${strokeIdx}`)() / 4294967296) * Math.PI * 2;
      const ang = sampleAngle(baseAng);
      const dir = norm({ x: Math.cos(ang), y: Math.sin(ang) });
      const perp = { x: -dir.y, y: dir.x };

      const heavy = Math.abs(rng.normal());
      const power = params.flingPower * (0.75 + 0.65 * rng.float()) * (1 + 0.9 * heavy * heavy);

      const along = power * (0.9 + 0.6 * rng.float());
      const across = power * (0.08 + 0.22 * (1 - params.directionality)) * rng.normal();

      const v = add(mul(dir, along / (physW * 0.02)), mul(perp, across / (physW * 0.02)));
      const p0 = sampleBrushOrigin(dir);
      const jitter = add(mul(perp, rng.normal() * (physW * 0.01)), mul(dir, rng.normal() * (physW * 0.01)));
      const p = add(p0, jitter);

      const vz = lerp(1.2, 2.6, clamp(heavy * 0.45, 0, 1));
      src = { p, v, vz };
    }

    let p = { x: src.p.x, y: src.p.y };
    let z = rng.range(0.9, 1.4);
    let v = { x: src.v.x, y: src.v.y };
    let vz = -src.vz;

    const packetMass = clamp(0.5 + 0.5 * rng.float(), 0.35, 1.0);

    v = add(v, { x: rng.normal() * 0.08, y: rng.normal() * 0.08 });

    let impacted = false;
    for (let s = 0; s < steps; s++) {
      const dragGain = params.geometry === "spray" ? 2.4 : 1.0;
      v = mul(v, 1 - params.drag * dragGain * 0.25 * dt);
      vz *= 1 - params.drag * dragGain * 0.35 * dt;

      vz -= 1.4 * dt;

      p = add(p, mul(v, physW * 0.015 * dt));
      z += vz * dt;

      // Don't clamp - let packets move freely, deposit functions handle bounds

      if (z <= 0) {
        impacted = true;
        break;
      }
    }

    if (!impacted) continue;

    const sp = length(v);
    const tangent = norm(v);
    const sens = params.geometry === "spray" ? 1.55 : 1.0;
    const impactEnergy = clamp((sp + Math.abs(vz)) * 0.6 * sens, 0.15, 3.2);

    const baseR = params.baseRadius + rng.normal() * params.radiusJitter;
    const coreR = clamp(baseR * (0.75 + 0.45 * impactEnergy), 1.5, physW * 0.12);

    const coreAmt = packetMass * (0.9 + 0.5 * rng.float());
    const ang = Math.atan2(tangent.y, tangent.x);
    const aniso = clamp(params.anisotropy, 1, 8);

    if (params.geometry === "fling") {
      const maj = coreR * lerp(1.0, 2.6, clamp((aniso - 1) / 7, 0, 1)) * (0.9 + 0.4 * sp);
      const min = coreR * lerp(1.0, 0.65, clamp((aniso - 1) / 7, 0, 1));
      depositAniso(field, p, maj, min, ang, coreAmt);
    } else {
      depositGaussian(field, p, coreR, coreAmt);
    }

    const streaks = (params.geometry === "fling" ? 10 : 4) + Math.floor(rng.range(0, params.geometry === "fling" ? 16 : 6));
    for (let k = 0; k < streaks; k++) {
      const t = rng.range(0.1, 1.0);
      const skew = rng.normal() * 0.25;
      const side = rot(tangent, skew);
      const spreadGain = params.geometry === "spray" ? 2.1 : 1.0;
      const dist = (coreR * 0.35 + t * params.impactSpread * spreadGain * coreR * impactEnergy) * (0.6 + 0.8 * rng.float());
      const pk = add(p, mul(side, dist));
      const rk = clamp(coreR * rng.range(0.18, 0.42), 1.0, coreR * 0.8);
      const ak = packetMass * rng.range(0.08, 0.22) * (1 + sp);
      if (params.geometry === "fling") {
        const maj = rk * (1.0 + 1.4 * clamp((params.anisotropy - 1) / 7, 0, 1)) * (0.6 + 0.6 * sp);
        const min = rk * 0.55;
        depositAniso(field, pk, maj, min, ang, ak);
      } else {
        depositGaussian(field, pk, rk, ak);
      }
    }

    const slideSteps = params.geometry === "fling" ? Math.floor(lerp(12, 38, clamp(params.tail, 0, 1.6) / 1.6)) : 9;
    let ps = { ...p };
    const restGain = params.geometry === "spray" ? 2.2 : 1.0;
    let vv = mul(tangent, sp * clamp(params.restitution * restGain, 0, 0.95));

    for (let t = 0; t < slideSteps; t++) {
      const f = surfaceFlow(ps);
      const visc = clamp(params.viscosity, 0, 1);
      vv = add(mul(vv, 1 - 0.45 * dt), mul(f, physW * 0.004 * (1 - visc)));
      ps = add(ps, vv);

      const rad = norm(sub(ps, center));
      const smearGain = params.geometry === "spray" ? 3.0 : 1.0;
      ps = add(ps, mul(rad, params.smear * smearGain * (0.35 + 0.65 * rng.float())));

      // Don't clamp - let packets move freely, deposit functions handle bounds

      const rr = clamp(coreR * rng.range(0.12, params.geometry === "fling" ? 0.28 : 0.35), 0.7, coreR);
      const aa = packetMass * rng.range(0.04, params.geometry === "fling" ? 0.10 : 0.12);

      if (params.geometry === "fling") {
        const decay = Math.exp(-t / Math.max(1, slideSteps * 0.65));
        const maj = rr * (1.2 + 2.4 * decay) * (0.7 + 0.8 * sp);
        const min = rr * (0.45 + 0.25 * decay);
        depositAniso(field, ps, maj, min, ang, aa * decay);
      } else {
        depositGaussian(field, ps, rr, aa);
      }

      const splatP = 0.12 + 0.08 * (1 - params.viscosity) + (params.geometry === "fling" ? 0.10 * clamp(params.tailDroplets, 0, 2) : 0);
      if (rng.float() < splatP) {
        const a2 = rng.range(0, Math.PI * 2);
        const dist2 =
          coreR *
          rng.range(params.geometry === "fling" ? 0.9 : 0.6, params.geometry === "fling" ? 5.2 : 2.2) *
          impactEnergy *
          (params.geometry === "fling" ? 0.7 + 0.6 * sp : 1);
        const ps2 = add(ps, { x: Math.cos(a2) * dist2, y: Math.sin(a2) * dist2 });
        const rr2 = clamp(coreR * rng.range(params.geometry === "fling" ? 0.03 : 0.05, params.geometry === "fling" ? 0.12 : 0.18), 0.5, 6);
        const aa2 = packetMass * rng.range(params.geometry === "fling" ? 0.015 : 0.02, params.geometry === "fling" ? 0.05 : 0.06);
        if (params.geometry === "fling") {
          const maj = rr2 * (1.0 + 1.2 * rng.float());
          const min = rr2 * (0.55 + 0.15 * rng.float());
          depositAniso(field, ps2, maj, min, a2, aa2);
        } else {
          depositGaussian(field, ps2, rr2, aa2);
        }
      }
    }

    const bounces = params.restitution > 0.05 ? (rng.float() < 0.2 ? 1 : 0) : 0;
    for (let b = 0; b < bounces; b++) {
      v = mul(v, params.restitution);
      vz = -vz * params.restitution;
      p = add(p, mul(v, physW * 0.01 * (0.6 + 0.4 * rng.float())));
      // Don't clamp - let packets move freely, deposit functions handle bounds
      depositGaussian(field, p, clamp(coreR * rng.range(0.25, 0.5), 1, coreR), packetMass * rng.range(0.12, 0.25));
    }
  }

  if (params.noise > 0) {
    for (let y = 0; y < workingW; y++) {
      for (let x = 0; x < workingW; x++) {
        const n = rng.normal() * params.noise;
        field.data[y * workingW + x] = Math.max(0, field.data[y * workingW + x] + n);
      }
    }
  }

  if (params.blur > 0) blurField(field, params.blur);

  const m = field.max() || 1;
  for (let i = 0; i < field.data.length; i++) field.data[i] /= m;

  // ---------------- Placement (applied to the underlying field) ----------------
  // Convert pan (SVG px) into field units, and warp the raster BEFORE contouring.
  // Camera semantics: panning right (+panX) moves viewport right, showing content from the left.
  const size = params.svgSize;
  const S0 = size / displayW; // svg px per field unit (without userScale)
  const userScale = clamp(params.userScale, 0.2, 3);
  const tField = { x: params.panX / S0, y: params.panY / S0 }; // translation in display field units

  function sampleBilinear(src, x, y) {
    // Clamp to valid range to extend edges (prevents hard clipping)
    const sx = clamp(x, 0, src.w - 1.001);
    const sy = clamp(y, 0, src.h - 1.001);
    
    const x0 = Math.floor(sx);
    const y0 = Math.floor(sy);
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const tx = sx - x0;
    const ty = sy - y0;

    const v00 = src.get(x0, y0);
    const v10 = src.get(x1, y0);
    const v01 = src.get(x0, y1);
    const v11 = src.get(x1, y1);

    const a = lerp(v00, v10, tx);
    const b = lerp(v01, v11, tx);
    return lerp(a, b, ty);
  }

  // Warp: output coord q -> input coord p = (q - tField) / userScale
  // We align the center of the display grid with the center of the working grid.
  const placed = new Field(displayW, displayW);
  for (let y = 0; y < displayW; y++) {
    for (let x = 0; x < displayW; x++) {
      // Map display coordinate (0..displayW) to working coordinate centered at workingW/2
      const px = (x - displayW / 2) / userScale + workingW / 2 - tField.x;
      const py = (y - displayW / 2) / userScale + workingW / 2 - tField.y;
      placed.data[y * displayW + x] = sampleBilinear(field, px, py);
    }
  }

  const polys = marchingSquares(placed, clamp(params.threshold, 0.02, 0.98));

  const cleaned = polys
    .map((p) => chaikinSmooth(p, params.smooth))
    .filter((p) => p.length > 30);

  const sorted = cleaned
    .map((p) => ({ p, a: polyArea(p) }))
    .sort((A, B) => Math.abs(B.a) - Math.abs(A.a));

  const candidates = sorted.slice(0, Math.min(sorted.length, 6));
  let outer = candidates[0]?.p ?? [];
  let outArea = candidates[0]?.a ?? 0;
  let bestScore = -1;

  for (const c of candidates) {
    if (!c.p.length) continue;
    let contains = 0;
    for (const o of sorted) {
      if (o.p === c.p || !o.p.length) continue;
      const oc = polyCentroid(o.p);
      if (pointInPoly(oc, c.p)) contains++;
    }
    const score = contains * 10 + Math.abs(c.a);
    if (score > bestScore) {
      bestScore = score;
      outer = c.p;
      outArea = c.a;
    }
  }

  const holes: Vec2[][] = [];
  const outerAbs = Math.abs(outArea) || 1;
  for (const item of sorted) {
    if (item.p === outer) continue;
    const aAbs = Math.abs(item.a);
    if (aAbs < outerAbs * 0.01) continue;
    if (aAbs > outerAbs * 0.22) continue;
    const c = polyCentroid(item.p);
    if (!pointInPoly(c, outer)) continue;
    holes.push(item.p);
  }

  // Placement already applied to field above; just map field coords to SVG coords.
  const S = size / displayW;
  const O = { x: 0, y: 0 };

  function ensureWinding(poly: Vec2[], wantClockwiseYDown: boolean) {
    const a = polyArea(poly);
    const isClockwise = a < 0;
    if (isClockwise !== wantClockwiseYDown) return poly.slice().reverse();
    return poly;
  }

  let outerFixed = outer;
  if (outerFixed.length) outerFixed = ensureWinding(outerFixed, true);
  const holeFixed = holes.map((h) => ensureWinding(h, false));

  const dOuter = outerFixed.length ? toSvgPath(outerFixed, S, O) : "";
  const dHoles = holeFixed.map((h) => toSvgPath(h, S, O)).join(" ");
  const d = `${dOuter} ${dHoles}`.trim();

  return { d, preview: placed, polysCount: polys.length };
}

// ------------------------- UI ------------------------------------------------

const DEFAULT: Params = {
  seed: "mind-the-math",
  svgSize: 900,
  fieldSize: 220,
  geometry: "spray",

  packets: 1000,
  baseRadius: 2.6,
  radiusJitter: 1.1,

  viscosity: 0.3,
  restitution: 0.1,
  drag: 0.22,
  impactSpread: 1.35,
  smear: 0.07,
  noise: 0.02,

  blur: 1,
  threshold: 0.24,
  smooth: 2,

  sprayAngleDeg: 20,
  sprayMagnitude: 0.0,
  sprayCovariance: 0.0,

  strokes: 4,
  flingPower: 22,
  directionality: 0.78,
  anisotropy: 4.6,
  tail: 0.95,
  tailDroplets: 0.8,

  panX: 0,
  panY: 0,
  userScale: 1,

  invertY: false,
};

function useDebounced<T>(value: T, delay = 250) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return v;
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: "image/svg+xml;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function PreviewCanvas({ field }: { field: Field | null }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const c = ref.current;
    if (!c || !field) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    const W = field.w;
    const H = field.h;
    c.width = W;
    c.height = H;

    const img = ctx.createImageData(W, H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const v = clamp(field.data[y * W + x], 0, 1);
        const idx = (y * W + x) * 4;
        const g = Math.round(255 * (1 - v));
        img.data[idx + 0] = g;
        img.data[idx + 1] = g;
        img.data[idx + 2] = g;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [field]);

  return (
    <div className="rounded-2xl border bg-white shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/40">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Droplets className="h-4 w-4" />
          <span>Density preview</span>
        </div>
        <div className="text-xs text-muted-foreground">(raster field)</div>
      </div>
      <div className="p-3">
        <canvas ref={ref} className="w-full h-auto" style={{ imageRendering: "pixelated" }} />
      </div>
    </div>
  );
}

function SvgPreview({ d, size }: { d: string; size: number }) {
  return (
    <div className="rounded-2xl border bg-white shadow-sm overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/40">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Sparkles className="h-4 w-4" />
          <span>SVG preview</span>
        </div>
        <div className="text-xs text-muted-foreground">
          {size}×{size}
        </div>
      </div>
      <div className="p-3">
        <svg viewBox={`0 0 ${size} ${size}`} width="100%" height="auto" className="rounded-xl border bg-white">
          <rect x={0} y={0} width={size} height={size} fill="white" />
          {d ? (
            <path d={d} fill="black" fillRule="evenodd" />
          ) : (
            <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" fontSize="16" fill="#666">
              (no contour — tweak threshold)
            </text>
          )}
        </svg>
      </div>
    </div>
  );
}

function HelpIcon({ text }: { text: string }) {
  return (
    <span
      className="inline-flex items-center justify-center h-5 w-5 rounded-full text-muted-foreground hover:text-foreground cursor-help"
      title={text}
      aria-label={text}
    >
      <Info className="h-4 w-4" />
    </span>
  );
}

function Row({ label, help, children }: { label: string; help: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-12 items-center gap-3">
      <div className="col-span-5">
        <div className="flex items-center gap-2">
          <Label className="text-sm text-muted-foreground">{label}</Label>
          <HelpIcon text={help} />
        </div>
      </div>
      <div className="col-span-7">{children}</div>
    </div>
  );
}

function runSelfTests() {
  // Lightweight sanity checks (no external test runner required).
  // These run once in the browser (dev-friendly).
  const base: Params = { ...DEFAULT, seed: "test-seed", geometry: "spray", packets: 600, fieldSize: 160 };
  const r1 = simulate(base);
  const r2 = simulate(base);
  console.assert(r1.d === r2.d, "Determinism test failed: same seed+params should match");

  const r3 = simulate({ ...base, seed: "test-seed-2" });
  console.assert(r1.d !== r3.d, "Seed sensitivity test failed: different seed should differ");

  console.assert(typeof r1.d === "string", "SVG path should be a string");
  console.assert(r1.d.length === 0 || r1.d.startsWith("M "), "SVG path should start with 'M ' when present");
}

if (typeof window !== "undefined") {
  const w = window as any;
  if (!w.__SEED_SVG_SPLOTCH_TESTED__) {
    w.__SEED_SVG_SPLOTCH_TESTED__ = true;
    try {
      runSelfTests();
    } catch {
      // ignore
    }
  }
}

export default function App() {
  const [p, setP] = useState<Params>(DEFAULT);
  const dp = useDebounced(p, 250);

  const result = useMemo(() => {
    try {
      return simulate(dp);
    } catch {
      return { d: "", preview: null as any, polysCount: 0 };
    }
  }, [dp]);

  const svgText = useMemo(() => {
    const size = p.svgSize;
    const d = result.d;

    const meta = {
      seed: p.seed,
      geometry: p.geometry,
      packets: p.packets,
      fieldSize: p.fieldSize,
      threshold: p.threshold,
      smooth: p.smooth,
      blur: p.blur,
      viscosity: p.viscosity,
      restitution: p.restitution,
      drag: p.drag,
      impactSpread: p.impactSpread,
      smear: p.smear,
      noise: p.noise,
      baseRadius: p.baseRadius,
      radiusJitter: p.radiusJitter,
      sprayAngleDeg: p.sprayAngleDeg,
      sprayMagnitude: p.sprayMagnitude,
      sprayCovariance: p.sprayCovariance,
      strokes: p.strokes,
      flingPower: p.flingPower,
      directionality: p.directionality,
      anisotropy: p.anisotropy,
      tail: p.tail,
      tailDroplets: p.tailDroplets,
      panX: p.panX,
      panY: p.panY,
      userScale: p.userScale,
    };

    const desc = `Generated by Seeded SVG Splotch Generator. Params: ${JSON.stringify(meta)}`;

    const flip = p.invertY ? ` transform=\"translate(0 ${size}) scale(1 -1)\"` : "";

    return (
      `<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n` +
      `<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"${size}\" height=\"${size}\" viewBox=\"0 0 ${size} ${size}\">\n` +
      `  <desc>${desc.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</desc>\n` +
      `  <g${flip}>\n` +
      `    <path d=\"${d}\" fill=\"#000\" fill-rule=\"evenodd\"/>\n` +
      `  </g>\n` +
      `</svg>\n`
    );
  }, [p, result.d]);

  const filename = useMemo(() => {
    const safe = p.seed.trim().replace(/[^a-zA-Z0-9_-]+/g, "-").slice(0, 40) || "splotch";
    return `splotch_${safe}_${p.geometry}.svg`;
  }, [p.seed, p.geometry]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-50 to-white">
      <div className="max-w-6xl mx-auto px-4 py-6">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="flex items-start justify-between gap-4"
        >
          <div>
            <div className="flex items-center gap-2">
              <div className="h-10 w-10 rounded-2xl bg-black text-white flex items-center justify-center shadow">
                <Droplets className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">Seeded SVG Splotch Generator</h1>
                <p className="text-sm text-muted-foreground">Deterministic, seed-driven paint splats → SVG export.</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => setP((pp) => ({ ...pp, seed: `${pp.seed}` }))}
              className="rounded-2xl"
              title="Re-run (deterministic, same seed)"
            >
              <RefreshCcw className="h-4 w-4 mr-2" />
              Re-run
            </Button>
            <Button onClick={() => downloadText(filename, svgText)} className="rounded-2xl" disabled={!result.d}>
              <Download className="h-4 w-4 mr-2" />
              Export SVG
            </Button>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 mt-6">
          <div className="lg:col-span-5">
            <Card className="rounded-2xl shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Controls</CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                <Row label="Seed" help="Deterministic seed string. Same seed + settings = same SVG.">
                  <Input
                    value={p.seed}
                    onChange={(e) => setP({ ...p, seed: e.target.value })}
                    className="rounded-xl"
                    placeholder="any string"
                  />
                </Row>

                <Row
                  label="Geometry"
                  help="How paint packets are emitted (spray cloud, line strike, circle source, or correlated fling strokes)."
                >
                  <Select
                    value={p.geometry}
                    onValueChange={(v) => {
                      const g = v as Geometry;
                      if (g === "spray") {
                        setP({
                          ...p,
                          geometry: g,
                          packets: 1000,
                          fieldSize: Math.min(p.fieldSize, 240),
                        });
                        return;
                      }
                      if (g === "fling") {
                        setP({
                          ...p,
                          geometry: g,
                          packets: Math.min(Math.max(p.packets, 800), 1200),
                          fieldSize: Math.min(p.fieldSize, 220),
                          strokes: Math.min(Math.max(p.strokes, 3), 6),
                        });
                        return;
                      }
                      setP({ ...p, geometry: g });
                    }}
                  >
                    <SelectTrigger className="rounded-xl">
                      <SelectValue placeholder="Select" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fling">Fling</SelectItem>
                      <SelectItem value="spray">Spray</SelectItem>
                      <SelectItem value="circle">Circle source</SelectItem>
                      <SelectItem value="line">Line strike</SelectItem>
                    </SelectContent>
                  </Select>
                </Row>

                <Row label={`Packets (${p.packets})`} help="Number of paint packets simulated. More = richer detail but slower.">
                  <Slider value={[p.packets]} min={200} max={2600} step={50} onValueChange={([v]) => setP({ ...p, packets: v })} />
                </Row>

                <Row label={`SVG size (${p.svgSize}px)`} help="Export canvas size in pixels (viewBox and width/height).">
                  <Slider value={[p.svgSize]} min={256} max={2000} step={16} onValueChange={([v]) => setP({ ...p, svgSize: v })} />
                </Row>

                <Row label={`Field res (${p.fieldSize})`} help="Internal simulation grid resolution. Higher = finer edges but slower.">
                  <Slider value={[p.fieldSize]} min={128} max={420} step={4} onValueChange={([v]) => setP({ ...p, fieldSize: v })} />
                </Row>

                <div className="pt-2 border-t" />

                {p.geometry === "spray" && (
                  <>
                    <Row
                      label={`Spray direction (${Math.round(p.sprayAngleDeg)}°)`}
                      help="Direction of the spray bias (adds a dominant direction without expensive fling simulation)."
                    >
                      <Slider value={[p.sprayAngleDeg]} min={0} max={360} step={1} onValueChange={([v]) => setP({ ...p, sprayAngleDeg: v })} />
                    </Row>

                    <Row
                      label={`Spray magnitude (${p.sprayMagnitude.toFixed(2)})`}
                      help="Strength of the directional drift and mean shift. Higher = more fling-like throw."
                    >
                      <Slider value={[p.sprayMagnitude]} min={0} max={1.6} step={0.01} onValueChange={([v]) => setP({ ...p, sprayMagnitude: v })} />
                    </Row>

                    <Row
                      label={`Spray covariance (${p.sprayCovariance.toFixed(2)})`}
                      help="Anisotropy of the spray cloud. Higher = stretched distribution along the spray direction."
                    >
                      <Slider value={[p.sprayCovariance]} min={0} max={1} step={0.01} onValueChange={([v]) => setP({ ...p, sprayCovariance: v })} />
                    </Row>

                    <div className="pt-2 border-t" />
                  </>
                )}

                {p.geometry === "fling" && (
                  <>
                    <Row label={`Strokes (${p.strokes})`} help="How many correlated bursts make up the fling. More strokes = more varied directions.">
                      <Slider value={[p.strokes]} min={1} max={14} step={1} onValueChange={([v]) => setP({ ...p, strokes: v })} />
                    </Row>

                    <Row label={`Fling power (${p.flingPower.toFixed(0)})`} help="How hard the paint is thrown. Higher = longer streaks and more energetic splatter.">
                      <Slider value={[p.flingPower]} min={8} max={30} step={1} onValueChange={([v]) => setP({ ...p, flingPower: v })} />
                    </Row>

                    <Row
                      label={`Directionality (${p.directionality.toFixed(2)})`}
                      help="How tightly packets align to each stroke's main direction. Higher = cleaner, more coherent streaks."
                    >
                      <Slider value={[p.directionality]} min={0} max={1} step={0.01} onValueChange={([v]) => setP({ ...p, directionality: v })} />
                    </Row>

                    <Row label={`Anisotropy (${p.anisotropy.toFixed(1)})`} help="Elongation of deposits along velocity. Higher = more streak-like marks.">
                      <Slider value={[p.anisotropy]} min={1} max={8} step={0.1} onValueChange={([v]) => setP({ ...p, anisotropy: v })} />
                    </Row>

                    <Row label={`Tail (${p.tail.toFixed(2)})`} help="How long surface streaking continues after impact. Higher = longer tails.">
                      <Slider value={[p.tail]} min={0} max={1.6} step={0.01} onValueChange={([v]) => setP({ ...p, tail: v })} />
                    </Row>

                    <Row label={`Tail droplets (${p.tailDroplets.toFixed(2)})`} help="Probability/amount of far droplets in fling mode.">
                      <Slider value={[p.tailDroplets]} min={0} max={2} step={0.01} onValueChange={([v]) => setP({ ...p, tailDroplets: v })} />
                    </Row>

                    <div className="pt-2 border-t" />
                  </>
                )}

                <Row label={`Base radius (${p.baseRadius.toFixed(1)})`} help="Typical droplet radius. Bigger = thicker blobs and fewer fine spikes.">
                  <Slider value={[p.baseRadius]} min={1.0} max={10} step={0.1} onValueChange={([v]) => setP({ ...p, baseRadius: v })} />
                </Row>

                <Row label={`Radius jitter (${p.radiusJitter.toFixed(1)})`} help="Random variation in droplet size. Higher = more texture and variety.">
                  <Slider value={[p.radiusJitter]} min={0} max={6} step={0.1} onValueChange={([v]) => setP({ ...p, radiusJitter: v })} />
                </Row>

                <Row label={`Viscosity (${p.viscosity.toFixed(2)})`} help="Higher viscosity resists flow and reduces streaking/slide distance.">
                  <Slider value={[p.viscosity]} min={0} max={1} step={0.01} onValueChange={([v]) => setP({ ...p, viscosity: v })} />
                </Row>

                <Row label={`Impact spread (${p.impactSpread.toFixed(2)})`} help="How far micro-droplets travel along the impact tangent. Higher = spikier splats.">
                  <Slider value={[p.impactSpread]} min={0.2} max={6} step={0.01} onValueChange={([v]) => setP({ ...p, impactSpread: v })} />
                </Row>

                <Row label={`Smear (${p.smear.toFixed(2)})`} help="Radial outward smear during surface slide. Higher = more outward pull and streaking.">
                  <Slider value={[p.smear]} min={0} max={0.6} step={0.01} onValueChange={([v]) => setP({ ...p, smear: v })} />
                </Row>

                <Row label={`Restitution (${p.restitution.toFixed(2)})`} help="Bounciness / secondary rebounds. Higher = more secondary hits and scattered texture.">
                  <Slider value={[p.restitution]} min={0} max={0.6} step={0.01} onValueChange={([v]) => setP({ ...p, restitution: v })} />
                </Row>

                <Row label={`Drag (${p.drag.toFixed(2)})`} help="Air resistance during flight. Higher = shorter travel before impact.">
                  <Slider value={[p.drag]} min={0} max={0.8} step={0.01} onValueChange={([v]) => setP({ ...p, drag: v })} />
                </Row>

                <div className="pt-2 border-t" />

                <Row label={`Noise (${p.noise.toFixed(3)})`} help="Adds subtle randomness to the density field to break perfectly smooth edges.">
                  <Slider value={[p.noise]} min={0} max={0.12} step={0.001} onValueChange={([v]) => setP({ ...p, noise: v })} />
                </Row>

                <Row label={`Blur passes (${p.blur})`} help="Post-blur on the density field before contour extraction. Higher = smoother blobs, fewer spikes.">
                  <Slider value={[p.blur]} min={0} max={5} step={1} onValueChange={([v]) => setP({ ...p, blur: v })} />
                </Row>

                <Row label={`Threshold (${p.threshold.toFixed(2)})`} help="Contour cutoff. Lower = larger blobs; higher = thinner/fragmented blobs.">
                  <Slider value={[p.threshold]} min={0.05} max={0.8} step={0.01} onValueChange={([v]) => setP({ ...p, threshold: v })} />
                </Row>

                <Row label={`Smooth (${p.smooth})`} help="Chaikin smoothing iterations on the extracted contour.">
                  <Slider value={[p.smooth]} min={0} max={5} step={1} onValueChange={([v]) => setP({ ...p, smooth: v })} />
                </Row>

                <Row label="Invert Y in export" help="Flips Y axis in exported SVG (useful for some coordinate systems).">
                  <div className="flex items-center justify-end gap-3">
                    <span className="text-xs text-muted-foreground">{p.invertY ? "On" : "Off"}</span>
                    <Switch checked={p.invertY} onCheckedChange={(v) => setP({ ...p, invertY: v })} />
                  </div>
                </Row>
              </CardContent>
            </Card>
          </div>

          <div className="lg:col-span-7 space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <SvgPreview d={result.d} size={p.svgSize} />
              <PreviewCanvas field={result.preview ?? null} />
            </div>

            <Card className="rounded-2xl shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Placement</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Row label={`Pan X (${Math.round(p.panX)}px)`} help="Horizontal offset applied to the exported path (pixels).">
                  <Slider value={[p.panX]} min={-p.svgSize / 2} max={p.svgSize / 2} step={1} onValueChange={([v]) => setP({ ...p, panX: v })} />
                </Row>
                <Row label={`Pan Y (${Math.round(p.panY)}px)`} help="Vertical offset applied to the exported path (pixels).">
                  <Slider value={[p.panY]} min={-p.svgSize / 2} max={p.svgSize / 2} step={1} onValueChange={([v]) => setP({ ...p, panY: v })} />
                </Row>
                <Row label={`Scale (${p.userScale.toFixed(2)}×)`} help="Scale multiplier applied to the exported path.">
                  <Slider value={[p.userScale]} min={0.2} max={1.5} step={0.01} onValueChange={([v]) => setP({ ...p, userScale: v })} />
                </Row>

                <div className="text-xs text-muted-foreground">
                  Manual placement is always on. It’s OK if the blob goes out of frame — adjust Pan/Scale to compose it.
                </div>
              </CardContent>
            </Card>

            <Card className="rounded-2xl shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Export</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-muted-foreground">
                    Filename: <span className="font-mono text-xs">{filename}</span>
                  </div>
                  <Button onClick={() => downloadText(filename, svgText)} className="rounded-2xl" disabled={!result.d}>
                    <Download className="h-4 w-4 mr-2" />
                    Export SVG
                  </Button>
                </div>

                <details className="rounded-xl border bg-muted/20 p-3">
                  <summary className="cursor-pointer text-sm">Show SVG source</summary>
                  <pre className="mt-2 text-xs overflow-auto max-h-64 p-2 rounded-lg bg-white border font-mono">{svgText}</pre>
                </details>

                <div className="text-xs text-muted-foreground">
                  The export is a single <code>&lt;path&gt;</code> using <code>fill-rule=\"evenodd\"</code>.
                </div>

                <div className="pt-3 border-t" />

                <div className="grid grid-cols-2 gap-2">
                  <Button variant="outline" className="rounded-2xl" onClick={() => setP(DEFAULT)}>
                    Reset
                  </Button>
                  <Button
                    className="rounded-2xl"
                    onClick={() => {
                      const h = xmur3(p.seed)();
                      setP({ ...p, seed: `${p.seed}-${(h % 100000).toString().padStart(5, "0")}` });
                    }}
                  >
                    <RefreshCcw className="h-4 w-4 mr-2" />
                    New seed
                  </Button>
                </div>

                <div className="text-xs text-muted-foreground leading-relaxed">
                  Tips: If you get "no contour", lower <b>Threshold</b> or increase <b>Packets</b>/<b>Base radius</b>. Higher <b>Field res</b> yields more detail.
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="mt-6 text-xs text-muted-foreground"
          >
            Determinism: All randomness is derived from the seed string. Changing any slider changes output deterministically.
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
