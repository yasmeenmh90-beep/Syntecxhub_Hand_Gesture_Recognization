// Rule-based gesture classifier: a port of gesture/classifier.py (v2: PIP-angle + reach
// finger states, ext/curl/ambiguous tri-state, hysteresis, fist safety, 🤘 horns).
// Keep the two in sync; tests/test_web_parity.py runs both on the same hands
// (stateless and with hysteresis) and fails on any difference.
//
// Landmarks: 21 points, {x, y, z} objects (MediaPipe JS) or [x, y, z] arrays, in
// isotropic units (see toIsotropic). Indices: 0 wrist | thumb 1-4 | index 5-8 |
// middle 9-12 | ring 13-16 | pinky 17-20.

export const WRIST = 0;
export const THUMB_MCP = 2, THUMB_IP = 3, THUMB_TIP = 4;
export const INDEX_MCP = 5, MIDDLE_MCP = 9, PINKY_MCP = 17;
export const INDEX_TIP = 8;
export const FINGER_JOINTS = { index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], pinky: [17, 18, 19, 20] };
export const FOUR = ["index", "middle", "ring", "pinky"];
export const ALL_FIVE = ["thumb", ...FOUR];
export const OK_PINCH_RATIO = 0.25;
export const GESTURES = ["open_palm", "fist", "thumbs_up", "thumbs_down", "peace", "pointing",
                         "rock", "ok", "call_me", "unknown"];
export const EXT = "ext", CURL = "curl", AMB = "amb";

const FINGER_EXT_STRAIGHT = [160.0, 0.95];
const FINGER_EXT_REACH = [145.0, 1.10];
const FINGER_CURL_ANGLE = 120.0, FINGER_CURL_REACH = 0.85;
const THUMB_EXT = { away: 1.02, angle: 145.0, out: 0.40 };
const THUMB_CURL = { away: 0.97, angle: 120.0, out: 0.30 };
// thumbs up / down (see gesture/classifier.py for the real-photo measurements)
export const THUMBS_ENTER_DEG = 50.0, THUMBS_STAY_DEG = 70.0;
const THUMB_OUT = { away: 1.0, away_min: 0.9, angle: 100.0, clear: 0.30, clear_min: 0.18, reach: 1.15 };
const THUMB_TUCKED = { reach: 1.05, away: 0.85, clear: 0.17 };
const FOLDED_REACH = 1.0;
export const DIRECTIONAL_GESTURES = ["thumbs_up", "thumbs_down"];

const P = (p) => (Array.isArray(p) ? [p[0], p[1], p[2] ?? 0] : [p.x, p.y, p.z ?? 0]);
const dist = (a, b) => { const [ax, ay] = P(a), [bx, by] = P(b); return Math.hypot(ax - bx, ay - by); };

function angle(a, b, c) {
  const pa = P(a), pb = P(b), pc = P(c);
  const v1 = [0, 1, 2].map((i) => pa[i] - pb[i]), v2 = [0, 1, 2].map((i) => pc[i] - pb[i]);
  const n = Math.sqrt(v1.reduce((s, x) => s + x * x, 0) * v2.reduce((s, x) => s + x * x, 0));
  if (n < 1e-12) return 180.0;
  const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
  return (Math.acos(Math.max(-1, Math.min(1, dot / n))) * 180) / Math.PI;
}

/** Undo MediaPipe's per-axis normalisation: x and z are scaled by width/height. */
export function toIsotropic(lm, width, height) {
  const k = width / height;
  return lm.map((p) => { const [x, y, z] = P(p); return [x * k, y, z * k]; });
}

export function fingerMetrics(lm) {
  const wrist = lm[WRIST];
  const palm = dist(wrist, lm[MIDDLE_MCP]) || 1e-6;
  const m = { palm };
  for (const [name, [mcp, pip, dip, tip]] of Object.entries(FINGER_JOINTS)) {
    m[`${name}_angle`] = angle(lm[mcp], lm[pip], lm[dip]);
    m[`${name}_reach`] = dist(wrist, lm[tip]) / (dist(wrist, lm[pip]) || 1e-6);
  }
  m.thumb_angle = angle(lm[THUMB_MCP], lm[THUMB_IP], lm[THUMB_TIP]);
  m.thumb_away = dist(lm[THUMB_TIP], lm[PINKY_MCP]) / (dist(lm[THUMB_IP], lm[PINKY_MCP]) || 1e-6);
  m.thumb_out = dist(lm[THUMB_TIP], lm[INDEX_MCP]) / palm;
  m.pinch = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm;
  m.thumb_clear = Math.min(...[5, 6, 7, 8, 9, 10, 11, 12].map((j) => dist(lm[THUMB_TIP], lm[j]))) / palm;
  m.thumb_reach = dist(lm[THUMB_TIP], wrist) / palm;
  const [tx, ty] = P(lm[THUMB_TIP]), [mx, my] = P(lm[THUMB_MCP]);
  const dx = tx - mx, dy = ty - my;
  m.thumb_from_up = (dx || dy) ? (Math.atan2(Math.abs(dx), -dy) * 180) / Math.PI : 90.0;
  return m;
}

export function thumbOutOfFist(m) {
  const t = THUMB_OUT, away = m.thumb_away, clear = m.thumb_clear, reach = m.thumb_reach;
  return m.thumb_angle >= t.angle && away >= t.away_min && clear >= t.clear_min &&
    (clear >= t.clear || reach >= t.reach) && (away >= t.away || reach >= t.reach);
}

export function thumbTucked(m) {
  const t = THUMB_TUCKED;
  return m.thumb_reach < t.reach || m.thumb_away < t.away || m.thumb_clear < t.clear;
}

export function fingersFolded(tri, m) {
  return FOUR.every((f) => tri[f] === CURL || (tri[f] === AMB && m[`${f}_reach`] < FOLDED_REACH));
}

export function thumbsPose(lm, maxDeg = THUMBS_STAY_DEG) {
  const m = fingerMetrics(lm);
  if (!(fingersFolded(fingerTristate(m), m) && thumbOutOfFist(m))) return null;
  if (m.thumb_from_up <= maxDeg) return "thumbs_up";
  if (m.thumb_from_up >= 180 - maxDeg) return "thumbs_down";
  return null;
}

export function fingerTristate(m) {
  const out = {};
  for (const f of FOUR) {
    const a = m[`${f}_angle`], r = m[`${f}_reach`];
    if (a <= FINGER_CURL_ANGLE || r <= FINGER_CURL_REACH) out[f] = CURL;
    else if ((a >= FINGER_EXT_STRAIGHT[0] && r >= FINGER_EXT_STRAIGHT[1]) ||
             (a >= FINGER_EXT_REACH[0] && r >= FINGER_EXT_REACH[1])) out[f] = EXT;
    else out[f] = AMB;
  }
  const { thumb_away: away, thumb_angle: ang, thumb_out: tout } = m;
  if (away <= THUMB_CURL.away || ang <= THUMB_CURL.angle || tout <= THUMB_CURL.out) out.thumb = CURL;
  else if (away >= THUMB_EXT.away && ang >= THUMB_EXT.angle && tout >= THUMB_EXT.out) out.thumb = EXT;
  else out.thumb = AMB;
  return out;
}

function shortAmbiguous(tri, m) {
  return new Set(FOUR.filter((f) => tri[f] === AMB && m[`${f}_reach`] < FOLDED_REACH));
}

function lean(f, m) {
  if (f === "thumb") return m.thumb_away >= 1.0 && m.thumb_angle >= 135 && m.thumb_out >= 0.35;
  return m[`${f}_angle`] >= 140 && m[`${f}_reach`] >= 1.0;
}

function rules(s, lm, m, okPinchRatio, tri, upDeg = THUMBS_ENTER_DEG, downDeg = THUMBS_ENTER_DEG) {
  const fourUp = FOUR.map((f) => s[f]);
  tri = tri || fingerTristate(m);
  const ang = m.thumb_from_up;
  const up = ang <= upDeg, down = ang >= 180 - downDeg;
  const curled4 = ["4 fingers curled", !fourUp.some(Boolean)];
  const folded4 = ["4 fingers folded", fingersFolded(tri, m)];
  const out = thumbOutOfFist(m);
  const thumbOut = ["thumb out of fist", out];
  return [
    ["ok", [["middle up", s.middle], ["ring up", s.ring], ["pinky up", s.pinky],
            [`pinch ${m.pinch.toFixed(2)} < ${okPinchRatio.toFixed(2)}`, m.pinch < okPinchRatio]]],
    ["open_palm", [["thumb up", s.thumb], ["index up", s.index], ["middle up", s.middle],
                   ["ring up", s.ring], ["pinky up", s.pinky]]],
    ["thumbs_up", [folded4, thumbOut, [`thumb ${ang.toFixed(0)}° from up <= ${upDeg.toFixed(0)}°`, up]]],
    ["thumbs_down", [folded4, thumbOut, [`thumb ${(180 - ang).toFixed(0)}° from down <= ${downDeg.toFixed(0)}°`, down]]],
    ["unknown", [folded4, thumbOut, ["thumb sideways", !up && !down]]],
    ["fist", [curled4, ["thumb tucked", !s.thumb && !out && thumbTucked(m)]]],
    ["rock", [["index up", s.index], ["pinky up", s.pinky], ["middle down", !s.middle], ["ring down", !s.ring]]],
    ["call_me", [["thumb up", s.thumb], ["pinky up", s.pinky], ["index down", !s.index],
                 ["middle down", !s.middle], ["ring down", !s.ring]]],
    ["peace", [["index up", s.index], ["middle up", s.middle], ["ring down", !s.ring], ["pinky down", !s.pinky]]],
    ["pointing", [["index up", s.index], ["middle down", !s.middle], ["ring down", !s.ring], ["pinky down", !s.pinky]]],
  ];
}

function decide(s, lm, m, okPinchRatio, tri, prevLabel) {
  const upDeg = prevLabel === "thumbs_up" ? THUMBS_STAY_DEG : THUMBS_ENTER_DEG;
  const downDeg = prevLabel === "thumbs_down" ? THUMBS_STAY_DEG : THUMBS_ENTER_DEG;
  const rs = rules(s, lm, m, okPinchRatio, tri, upDeg, downDeg);
  for (const [label, conds] of rs) {
    if (conds.every(([, ok]) => ok)) {
      return label === "unknown" ? ["unknown", "thumb points sideways (not up or down)"] : [label, ""];
    }
  }
  const fails = rs.filter(([l]) => l !== "unknown").map(([l, c]) => [l, c.filter(([, ok]) => !ok).map(([n]) => n)]);
  const best = Math.min(...fails.map(([, f]) => f.length));
  const near = fails.filter(([, f]) => f.length === best).slice(0, 3).map(([l, f]) => `${l} (${f.join(", ")})`);
  return ["unknown", `blocked: ${near.join("; ")}`];
}

/** Classify and say why. `prev` = last frame's resolved states (hysteresis). */
export function explain(lm, okPinchRatio = OK_PINCH_RATIO, prev = null, prevLabel = null) {
  if (!lm || lm.length !== 21) {
    return { label: "unknown", states: Object.fromEntries(ALL_FIVE.map((f) => [f, false])),
             tri: Object.fromEntries(ALL_FIVE.map((f) => [f, AMB])), metrics: {}, reason: "need 21 landmarks" };
  }
  const m = fingerMetrics(lm);
  const tri = fingerTristate(m);
  const short = shortAmbiguous(tri, m); // half-bent but short: folded, never extended
  const amb = ALL_FIVE.filter((f) => tri[f] === AMB && !short.has(f));
  const base = Object.fromEntries(ALL_FIVE.map((f) => [f, tri[f] === EXT]));
  let label, reason, states;
  if (prev) {
    states = Object.fromEntries(ALL_FIVE.map((f) => [f, amb.includes(f) ? (f in prev ? prev[f] : lean(f, m)) : base[f]]));
    [label, reason] = decide(states, lm, m, okPinchRatio, tri, prevLabel);
  } else {
    states = Object.fromEntries(ALL_FIVE.map((f) => [f, amb.includes(f) ? lean(f, m) : base[f]]));
    const outcomes = new Set();
    for (let bits = 0; bits < 1 << amb.length; bits++) {
      // same enumeration order as itertools.product((True, False), ...): True first
      const s = { ...base };
      amb.forEach((f, i) => { s[f] = ((bits >> (amb.length - 1 - i)) & 1) === 0; });
      outcomes.add(decide(s, lm, m, okPinchRatio, tri, prevLabel)[0]);
    }
    if (outcomes.size === 1) {
      label = [...outcomes][0];
      reason = label !== "unknown" ? "" : decide(states, lm, m, okPinchRatio, tri, prevLabel)[1];
    } else {
      label = "unknown";
      reason = `ambiguous ${amb.join("/")}: could be ${[...outcomes].sort().join(" or ")}`;
    }
  }
  if (label === "fist" && FOUR.some((f) => tri[f] === AMB)) {
    const fuzzy = FOUR.filter((f) => tri[f] === AMB);
    label = "unknown";
    reason = `fist needs clearly curled fingers (${fuzzy.join("/")} unclear)`;
  }
  return { label, states, tri, metrics: m, reason };
}

export function classify(lm, okPinchRatio = OK_PINCH_RATIO, prev = null, prevLabel = null) {
  return explain(lm, okPinchRatio, prev, prevLabel).label;
}

/** Back-compat: {finger: isExtended}; ambiguous fingers keep `prev` if given. */
export function fingerStates(lm, prev = null) {
  const m = fingerMetrics(lm), tri = fingerTristate(m), short = shortAmbiguous(tri, m);
  return Object.fromEntries(ALL_FIVE.map((f) =>
    [f, tri[f] !== AMB ? tri[f] === EXT : short.has(f) ? false : (prev && f in prev ? prev[f] : lean(f, m))]));
}

/** Per-hand classifier with finger-state hysteresis across frames. */
export class PoseTracker {
  constructor(okPinchRatio = OK_PINCH_RATIO) {
    this.okPinchRatio = okPinchRatio; this.prev = null; this.prevLabel = null; this.last = null;
  }
  update(lm) {
    const info = explain(lm, this.okPinchRatio, this.prev, this.prevLabel);
    this.prev = info.states; this.last = info;
    if (info.label !== "unknown") this.prevLabel = info.label;
    else if (info.reason.startsWith("thumb points sideways")) this.prevLabel = null;
    return info;
  }
  reset() { this.prev = null; this.last = null; this.prevLabel = null; }
}

/** Keeps a confirmed thumbs up/down through up to `grace` s of "unknown" (port of HoldThrough). */
export class HoldThrough {
  constructor(gestures = DIRECTIONAL_GESTURES, grace = 0.3) {
    this.gestures = new Set(gestures); this.grace = grace; this.held = null; this.seen = -Infinity;
  }
  update(t, stable) {
    if (this.gestures.has(stable)) { this.held = stable; this.seen = t; return stable; }
    if (stable !== "unknown") { this.held = null; return stable; }
    if (this.held && t - this.seen <= this.grace + 1e-9) return this.held;
    this.held = null;
    return stable;
  }
  reset() { this.held = null; }
}

/** Majority vote over the last N frames (port of GestureSmoother, plus `confidence`). */
export class GestureSmoother {
  constructor(window = 7, minRatio = 0.6) {
    this.window = window;
    this.minRatio = minRatio;
    this.history = [];
  }
  _top() {
    const counts = new Map(); // Map keeps insertion order, like collections.Counter
    for (const g of this.history) counts.set(g, (counts.get(g) || 0) + 1);
    let best = "unknown", bestN = 0;
    for (const [g, n] of counts) if (n > bestN) { best = g; bestN = n; } // ties: first inserted wins
    return [best, bestN];
  }
  update(gesture) {
    this.history.push(gesture);
    if (this.history.length > this.window) this.history.shift();
    const [label, count] = this._top();
    if (this.history.length === this.window && count / this.history.length >= this.minRatio) return label;
    return "unknown";
  }
  get confidence() {
    return this.history.length ? this._top()[1] / this.window : 0;
  }
  reset() { this.history = []; }
}

/** Fires an action once per new stable gesture (cooldown) and repeats held `repeat` ones.
 *  Same logic as gesture/actions.ActionDispatcher.update(). Times are in seconds. */
export class ActionDispatcher {
  constructor(bindings, { cooldown = 1.0, repeatInterval = 0.35 } = {}) {
    this.bindings = bindings; // {gesture: {action, label, repeat}}
    this.cooldown = cooldown;
    this.repeatInterval = repeatInterval;
    this.lastGesture = "unknown";
    this.lastFire = -Infinity;
  }
  update(gesture, now) {
    let fired = null;
    const b = this.bindings[gesture];
    if (b) {
      const isNew = gesture !== this.lastGesture;
      if (isNew && now - this.lastFire >= this.cooldown) fired = b;
      else if (!isNew && b.repeat && now - this.lastFire >= this.repeatInterval) fired = b;
      if (fired) this.lastFire = now;
    }
    this.lastGesture = gesture;
    return fired;
  }
}
