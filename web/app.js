// Browser version of the hand gesture demo. No build step: plain ES modules.
// MediaPipe Tasks Vision (pinned) finds the hand, classifier.js (a port of
// gesture/classifier.py) names the gesture, and gestures drive a demo <audio> player,
// since a web page can't press OS media keys.
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs";
import { ActionDispatcher, GestureSmoother, HoldThrough, PoseTracker, toIsotropic } from "./classifier.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ------------------------------------------------------------------ theme + mappings
const PINK = "#ff4fd8", PURPLE = "#8b5cf6", WHITE = "#ffffff";
const ICONS = {
  open_palm: "✋", fist: "✊", thumbs_up: "👍", thumbs_down: "👎", peace: "✌️", pointing: "☝️",
  rock: "🤟", ok: "👌", call_me: "🤙", unknown: "🤔", none: "🫥",
};
const BINDINGS = { // same gestures and labels as the desktop "media" mode in config.yaml
  open_palm:   { action: "play_pause",  label: "Play / Pause" },
  fist:        { action: "mute",        label: "Mute" },
  thumbs_up:   { action: "volume_up",   label: "Volume +", repeat: true },
  thumbs_down: { action: "volume_down", label: "Volume -", repeat: true },
  peace:       { action: "next",        label: "Next track" },
  pointing:    { action: "previous",    label: "Previous track" },
};
const pretty = (g) => g.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
const FINGER_T = [0, ...Array.from({ length: 20 }, (_, i) => ((i % 4) + 1) / 4)]; // 0 wrist .. 1 tip
const CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10],
  [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");

// ------------------------------------------------------------------ generated playlist
// Three short original loops synthesised to WAV in the browser: nothing to license or download.
const midi = (m) => 440 * 2 ** ((m - 69) / 12);
const TRACKS = [
  { title: "Neon Drift", bpm: 100, wave: "tri", chords: [[57, 60, 64], [53, 57, 60], [48, 52, 55, 60], [55, 59, 62]] },
  { title: "Glass Garden", bpm: 118, wave: "sine", chords: [[60, 64, 67, 71], [64, 67, 71, 74], [65, 69, 72, 76], [67, 71, 74, 77]] },
  { title: "Midnight Pulse", bpm: 92, wave: "soft", chords: [[62, 65, 69], [58, 62, 65], [65, 69, 72], [60, 64, 67]] },
];

function osc(wave, phase) {
  const s = Math.sin(phase);
  if (wave === "sine") return s;
  if (wave === "tri") return (2 / Math.PI) * Math.asin(s);
  return s + Math.sin(3 * phase) / 3 + Math.sin(5 * phase) / 5; // soft square
}

function renderTrack({ bpm, chords, wave }, rate = 22050) {
  const beat = 60 / bpm, bar = 4 * beat, bars = chords.length * 2;
  const out = new Float32Array(Math.ceil(bars * bar * rate));
  const add = (freq, start, dur, amp, w, decay) => {
    const i0 = Math.floor(start * rate), n = Math.floor(dur * rate);
    for (let i = 0; i < n && i0 + i < out.length; i++) {
      const t = i / rate;
      const env = Math.min(1, t / 0.006) * Math.exp(-t * decay);
      out[i0 + i] += amp * env * osc(w, 2 * Math.PI * freq * t);
    }
  };
  for (let b = 0; b < bars; b++) {
    const chord = chords[Math.floor(b / 2) % chords.length], t0 = b * bar;
    add(midi(chord[0] - 24), t0, bar, 0.32, "sine", 1.2);                      // bass
    for (const n of chord) add(midi(n), t0, bar, 0.05, "sine", 0.8);           // pad
    const arp = [...chord, chord[1] + 12, ...chord.slice().reverse()];
    for (let k = 0; k < 8; k++) add(midi(arp[k % arp.length] + 12), t0 + k * beat / 2, beat, 0.16, wave, 5); // arp
  }
  let peak = 1e-6;
  for (const v of out) peak = Math.max(peak, Math.abs(v));
  return encodeWav(out.map((v) => (v / peak) * 0.8), rate);
}

function encodeWav(samples, rate) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); v.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true);
  v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}

// ------------------------------------------------------------------ player
class Player {
  constructor(audio, tracks) {
    this.audio = audio;
    this.tracks = tracks;
    this.index = 0;
    this.audio.volume = 0.7;
    this.audio.src = tracks[0].url;
    this.audio.addEventListener("ended", () => this.next(true));
    for (const ev of ["play", "pause", "volumechange", "loadedmetadata"]) {
      this.audio.addEventListener(ev, () => this.render());
    }
    this.render();
  }
  get playing() { return !this.audio.paused; }
  async play() {
    try { await this.audio.play(); return true; } catch { return false; }
  }
  togglePlay() { return this.playing ? (this.audio.pause(), Promise.resolve(true)) : this.play(); }
  load(i, keepPlaying) {
    const wasPlaying = keepPlaying ?? this.playing;
    this.index = (i + this.tracks.length) % this.tracks.length;
    this.audio.src = this.tracks[this.index].url;
    if (wasPlaying) this.play();
    this.render();
  }
  next(keepPlaying) { this.load(this.index + 1, keepPlaying); }
  previous() { this.load(this.index - 1); }
  changeVolume(d) {
    this.audio.muted = false;
    this.audio.volume = Math.round(Math.max(0, Math.min(1, this.audio.volume + d)) * 10) / 10;
  }
  toggleMute() { this.audio.muted = !this.audio.muted; }
  get volumePct() { return Math.round(this.audio.volume * 100); }
  render() {
    $("track-title").textContent = this.tracks[this.index].title;
    $("track-no").textContent = `· ${this.index + 1} / ${this.tracks.length}`;
    $("btn-play").textContent = this.playing ? "⏸" : "▶";
    $("vol-fill").style.width = `${this.audio.muted ? 0 : this.volumePct}%`;
    $("vol-text").textContent = this.audio.muted ? "muted" : `${this.volumePct}%`;
    $("vol-icon").textContent = this.audio.muted || this.volumePct === 0 ? "🔇" : this.volumePct < 50 ? "🔉" : "🔊";
    $("player").classList.toggle("muted-state", this.audio.muted);
  }
  /** Run a gesture action; returns the toast text. */
  async run(action, label) {
    switch (action) {
      case "play_pause": {
        const ok = await this.togglePlay();
        return ok ? (this.playing ? "▶  Play" : "⏸  Pause") : "Tap ▶ once to allow audio";
      }
      case "mute": this.toggleMute(); return this.audio.muted ? "🔇  Mute" : "🔊  Unmute";
      case "volume_up": this.changeVolume(+0.1); return `👍  Volume ${this.volumePct}%`;
      case "volume_down": this.changeVolume(-0.1); return `👎  Volume ${this.volumePct}%`;
      case "next": this.next(); return `⏭  ${this.tracks[this.index].title}`;
      case "previous": this.previous(); return `⏮  ${this.tracks[this.index].title}`;
      default: return label;
    }
  }
}

// ------------------------------------------------------------------ drawing
const lerpHex = (a, b, t) => {
  const pa = a.match(/\w\w/g).map((h) => parseInt(h, 16)), pb = b.match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`;
};

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function drawHand(lm, label, s) {
  const W = canvas.width, H = canvas.height;
  const pts = lm.map((p) => [p.x * W, p.y * H]);
  const bone = Math.max(2, 4 * s);
  const strokeBones = (width) => {
    for (const [a, b] of CONNECTIONS) {
      const g = ctx.createLinearGradient(pts[a][0], pts[a][1], pts[b][0], pts[b][1]);
      g.addColorStop(0, lerpHex(PINK, PURPLE, FINGER_T[a]));
      g.addColorStop(1, lerpHex(PINK, PURPLE, FINGER_T[b]));
      ctx.strokeStyle = g;
      ctx.lineWidth = width;
      ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke();
    }
  };
  ctx.save();
  ctx.lineCap = "round";
  ctx.globalAlpha = 0.5; ctx.shadowColor = PINK; ctx.shadowBlur = 18 * s;  // soft glow layer
  strokeBones(bone * 3);
  ctx.restore();
  ctx.save();
  ctx.lineCap = "round";
  strokeBones(bone);
  pts.forEach(([x, y], i) => {
    const c = lerpHex(PINK, PURPLE, FINGER_T[i]);
    if (FINGERTIPS.has(i)) {
      const r = Math.max(4, 9 * s);
      ctx.fillStyle = WHITE; ctx.beginPath(); ctx.arc(x, y, r + 2 * s, 0, 7); ctx.fill();
      ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
    } else {
      ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, Math.max(2, 4 * s), 0, 7); ctx.fill();
    }
  });
  // rounded box + tag
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]), m = 22 * s;
  const bx = Math.max(Math.min(...xs) - m, 0), by = Math.max(Math.min(...ys) - m, 0);
  const bw = Math.min(Math.max(...xs) + m, W - 1) - bx, bh = Math.min(Math.max(...ys) + m, H - 1) - by;
  ctx.strokeStyle = PINK; ctx.lineWidth = Math.max(1, 2 * s);
  roundRect(bx, by, bw, bh, 18 * s); ctx.stroke();
  if (label) {
    const size = Math.max(12, Math.round(18 * s));
    ctx.font = `700 ${size}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
    const tw = ctx.measureText(label).width, ph = size + 12 * s, pw = tw + 24 * s;
    let tx = bx, ty = by - ph - 6 * s;
    if (ty < 0) { tx = bx + 6 * s; ty = by + 6 * s; }
    ctx.fillStyle = PINK; roundRect(tx, ty, pw, ph, ph / 2); ctx.fill();
    ctx.fillStyle = WHITE; ctx.textBaseline = "middle"; ctx.fillText(label, tx + 12 * s, ty + ph / 2 + 1);
  }
  ctx.restore();
}

function palmCenter(lm) {
  const idx = [0, 5, 9, 13, 17];
  return [idx.reduce((a, i) => a + lm[i].x, 0) / 5 * canvas.width, idx.reduce((a, i) => a + lm[i].y, 0) / 5 * canvas.height];
}

function drawPulse(pulse, now, s) {
  const t = (now - pulse.t0) / 600;
  if (t >= 1) return false;
  const ease = 1 - (1 - t) ** 3;
  ctx.save();
  ctx.globalAlpha = 1 - t;
  ctx.strokeStyle = lerpHex(PINK, PURPLE, t);
  ctx.lineWidth = Math.max(1, 8 * s * (1 - t)) + 1;
  ctx.beginPath(); ctx.arc(pulse.x, pulse.y, 30 * s + 110 * s * ease, 0, 7); ctx.stroke();
  ctx.restore();
  return true;
}

// ------------------------------------------------------------------ UI state (DOM)
const setText = (el, text) => { if (el.textContent !== text) el.textContent = text; };

function buildLegend() {
  $("legend-rows").innerHTML = Object.entries(BINDINGS).map(([g, b]) =>
    `<li data-g="${g}"><span class="g">${ICONS[g]}&nbsp; ${pretty(g)}</span><span>${b.label}</span></li>`).join("");
}

function updateHud(raw, stable, conf, handPresent) {
  const g = handPresent ? stable : "none";
  setText($("hud-icon"), ICONS[g] || "");
  setText($("hud-name"), handPresent ? pretty(stable) : "No hand");
  const action = $("hud-action");
  if (BINDINGS[g]) { setText(action, `→  ${BINDINGS[g].label}`); action.classList.add("bound"); }
  else {
    action.classList.remove("bound");
    setText(action, !handPresent ? "Waiting for a hand" : ["rock", "ok", "call_me"].includes(g)
      ? "Desktop-only action" : "Hold a gesture steady…");
  }
  setText($("hud-raw"), `raw: ${raw.replace(/_/g, " ")}   ${Math.round(conf * 100)}%`);
  $("conf-fill").style.width = `${Math.round((handPresent ? conf : 0) * 100)}%`;
  for (const li of $("legend-rows").children) li.classList.toggle("active", li.dataset.g === stable && handPresent);
  $("hint").hidden = handPresent;
}

function showToast(text) {
  const el = $("toast");
  setText($("toast-text"), text);
  el.classList.remove("show");
  void el.offsetWidth; // restart the CSS animation
  el.classList.add("show");
}

// ------------------------------------------------------------------ main loop
let landmarker, player;
const smoother = new GestureSmoother(7, 0.6);
const hold = new HoldThrough(); // thumbs up/down survive 0.3 s of unknown frames (volume repeat)
const poseTrackers = new Map(); // per hand (by handedness): finger-state hysteresis
const dispatcher = new ActionDispatcher(BINDINGS, { cooldown: 1.0, repeatInterval: 0.35 });
let lastVideoTime = -1, hands = [], raw = "no hand", stable = "unknown";
let fps = 0, lastDetect = performance.now(), pulse = null;

function frame(now) {
  if (video.readyState >= 2 && video.videoWidth) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }
    const s = canvas.width / 1280;

    if (video.currentTime !== lastVideoTime) { // only run the model on new camera frames
      lastVideoTime = video.currentTime;
      const res = landmarker.detectForVideo(video, now);
      // mirror like the desktop demo (it flips the frame before detection)
      hands = res.landmarks.map((lm) => lm.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z })));
      // classify in isotropic units (MediaPipe divides x by width but y by height)
      const seen = [];
      const labels = hands.map((lm, i) => {
        const side = res.handedness?.[i]?.[0]?.categoryName || "Hand";
        const key = seen.includes(side) ? `${side}${i}` : side;
        seen.push(key);
        if (!poseTrackers.has(key)) poseTrackers.set(key, new PoseTracker());
        return poseTrackers.get(key).update(toIsotropic(lm, canvas.width, canvas.height)).label;
      });
      for (const key of [...poseTrackers.keys()]) if (!seen.includes(key)) poseTrackers.delete(key);
      fps = 0.9 * fps + 0.1 * (1000 / Math.max(now - lastDetect, 1));
      lastDetect = now;

      if (hands.length) {
        raw = labels[0];
        stable = hold.update(now / 1000, smoother.update(raw));
      } else {
        smoother.reset();
        hold.reset();
        raw = "no hand"; stable = "unknown";
      }
      const fired = dispatcher.update(stable, now / 1000);
      if (fired) {
        const gesture = stable;
        player.run(fired.action, fired.label).then(showToast);
        if (hands.length) { const [x, y] = palmCenter(hands[0]); pulse = { x, y, t0: now }; }
        console.debug(`[action] ${gesture} -> ${fired.label}`);
      }
      updateHud(raw, stable, smoother.confidence, hands.length > 0);
      setText($("fps"), `${Math.round(fps)} FPS`);
    }

    ctx.save(); // mirrored camera image
    ctx.translate(canvas.width, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();
    hands.forEach((lm, i) => drawHand(lm, i === 0 ? `${ICONS[stable] || ""} ${pretty(stable)}` : null, s));
    if (pulse && !drawPulse(pulse, now, s)) pulse = null;
    if (player.audio.duration) $("progress").style.width = `${(player.audio.currentTime / player.audio.duration) * 100}%`;
  }
  requestAnimationFrame(frame);
}

async function createLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.5,
  });
  try { return await HandLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch (e) { console.warn("GPU delegate unavailable, using CPU", e); return HandLandmarker.createFromOptions(fileset, opts("CPU")); }
}

function fail(msg) {
  const st = $("start-status");
  st.textContent = msg; st.classList.add("error");
  $("start-btn").disabled = false; $("start-btn").textContent = "Try again";
}

async function start() {
  const btn = $("start-btn"), st = $("start-status");
  // Unlock audio inside the click (browsers block play() that isn't started by a user gesture)
  player.audio.muted = true;
  const unlock = player.audio.play().then(() => { player.audio.pause(); player.audio.currentTime = 0; })
    .catch(() => {}).finally(() => { player.audio.muted = false; player.render(); });

  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    return fail("The camera needs a secure page: open this over https:// or http://localhost.");
  }
  btn.disabled = true; btn.textContent = "Starting…";
  st.classList.remove("error");
  st.textContent = "Loading the hand model (about 8 MB, first time only)…";
  try {
    const [stream, lmk] = await Promise.all([
      navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false }),
      landmarker ? Promise.resolve(landmarker) : createLandmarker(),
    ]);
    landmarker = lmk;
    video.srcObject = stream;
    await video.play();
    await unlock;
  } catch (e) {
    console.error(e);
    if (e?.name === "NotAllowedError") return fail("Camera permission was denied. Allow it in the address bar, then try again.");
    if (e?.name === "NotFoundError") return fail("No camera found.");
    return fail(`Couldn't start: ${e?.message || e}`);
  }
  $("start").hidden = true;
  for (const id of ["hud", "pills", "player", "hint", "legend-btn"]) $(id).hidden = false; // legend: 'h' or ? button
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ boot
TRACKS.forEach((t) => { t.url = renderTrack(t); });
player = new Player($("audio"), TRACKS);
buildLegend();
$("start-btn").addEventListener("click", start);
$("btn-play").addEventListener("click", () => player.togglePlay());
$("btn-next").addEventListener("click", () => player.next());
$("btn-prev").addEventListener("click", () => player.previous());
function toggleLegend() {
  const lg = $("legend");
  lg.hidden = !lg.hidden; lg.classList.toggle("force", !lg.hidden);
  $("legend-btn").setAttribute("aria-expanded", String(!lg.hidden));
}
$("legend-btn").addEventListener("click", toggleLegend);
window.addEventListener("keydown", (e) => {
  if (e.key === "h" && $("start").hidden) toggleLegend(); // only once the app is running
});
