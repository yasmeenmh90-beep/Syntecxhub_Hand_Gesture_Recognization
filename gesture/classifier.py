"""Rule-based gesture classifier on top of the 21 MediaPipe hand landmarks.

Each finger is judged from two cues that together work for upright *and* sideways hands:
  - the bend at its PIP joint (3-D angle MCP-PIP-DIP: straight ~160-180°, curled < 120°)
  - its reach: wrist->tip distance over wrist->PIP distance (curled fingers fold back, < ~0.85)
and gets a tri-state: extended / curled / ambiguous (in between).

Ambiguous fingers keep their previous state when a `prev` state is given (hysteresis:
becoming extended and becoming curled need different, stricter evidence). Without
history, the pose is classified only if every reading of the ambiguous fingers gives the
same gesture; otherwise it's "unknown". Fist additionally needs all four fingers
*clearly* curled, so an unclear pose never fires Mute.

The thumb is extended when it's straight (IP angle) and points away from the palm (its tip
is further from the pinky knuckle than its IP joint is). An earlier fixed rule, "tip > 0.55
palm from the index knuckle", rejected real spread thumbs (measured 0.48), which turned
open palms into "unknown".

Landmarks should be in isotropic units: MediaPipe normalises x by the frame width and y by
the height, so on a 16:9 camera use to_isotropic() (x * w/h) before classifying. All rules
are ratios and angles, so the overall scale doesn't matter.

Landmark indices: 0 wrist | thumb 1-4 | index 5-8 | middle 9-12 | ring 13-16 | pinky 17-20
"""
from __future__ import annotations

import itertools
import math
from collections import Counter, deque
from dataclasses import dataclass, field

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
INDEX_TIP = 8
FINGERS = {  # name: (pip, tip), kept for backward compatibility
    "index": (6, 8),
    "middle": (10, 12),
    "ring": (14, 16),
    "pinky": (18, 20),
}
FINGER_JOINTS = {  # name: (mcp, pip, dip, tip)
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FOUR = ("index", "middle", "ring", "pinky")
ALL_FIVE = ("thumb",) + FOUR

GESTURES = ["open_palm", "fist", "thumbs_up", "thumbs_down", "peace", "pointing",
            "rock", "ok", "call_me", "unknown"]
OK_PINCH_RATIO = 0.25  # 👌 thumb tip–index tip distance below this × palm size

EXT, CURL, AMB = "ext", "curl", "amb"

# ---- thresholds (see module docstring). Between "curl" and "ext" a finger is ambiguous.
FINGER_EXT_STRAIGHT = (160.0, 0.95)  # (min PIP angle, min reach): clearly straight, even sideways
FINGER_EXT_REACH = (145.0, 1.10)     # ...or fairly straight and clearly reaching out
FINGER_CURL_ANGLE = 120.0            # curled if PIP angle <= this
FINGER_CURL_REACH = 0.85             # ...or reach <= this
THUMB_EXT = dict(away=1.02, angle=145.0, out=0.40)
THUMB_CURL = dict(away=0.97, angle=120.0, out=0.30)

# ---- thumbs up / down (measured on real photos: thumbs-up thumbs clear the finger joints
# by 0.66-0.87 palm and reach 1.4-1.6 palm from the wrist; tucked thumbs 0.07-0.18 / 0.83-0.89)
THUMBS_ENTER_DEG = 50.0   # thumb within this of vertical to *become* thumbs up/down
THUMBS_STAY_DEG = 70.0    # ...and it stays thumbs until it tilts past this (hysteresis)
THUMB_OUT = dict(away=1.0, away_min=0.9, angle=100.0, clear=0.30, clear_min=0.18, reach=1.15)
# a *tucked* thumb (fist): near the wrist, pointing back toward the palm, or lying on the
# fingers (real tucked thumbs: away 0.59-0.70, reach 0.83-0.89, clearance 0.07-0.18 palm)
THUMB_TUCKED = dict(reach=1.05, away=0.85, clear=0.17)
FOLDED_REACH = 1.0        # an ambiguous finger still counts as folded if its reach is below this
DIRECTIONAL_GESTURES = ("thumbs_up", "thumbs_down")


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle(a, b, c) -> float:
    """Angle at b (degrees) between b->a and b->c, in 3-D when z is present."""
    v1 = [a[i] - b[i] for i in range(min(len(a), len(b), 3))]
    v2 = [c[i] - b[i] for i in range(min(len(c), len(b), 3))]
    n = math.sqrt(sum(x * x for x in v1) * sum(x * x for x in v2))
    if n < 1e-12:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(v1, v2)) / n))))


def to_isotropic(lm, width: float, height: float):
    """Undo MediaPipe's per-axis normalisation: x and z are scaled by width/height."""
    k = width / height
    return [(p[0] * k, p[1], (p[2] if len(p) > 2 else 0.0) * k) for p in lm]


def palm_size(lm) -> float:
    return _dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6


def finger_metrics(lm) -> dict[str, float]:
    """Raw numbers behind the finger states (also shown in demo.py's 'g' debug overlay)."""
    wrist = lm[WRIST]
    palm = palm_size(lm)
    m: dict[str, float] = {"palm": palm}
    for name, (mcp, pip, dip, tip) in FINGER_JOINTS.items():
        m[f"{name}_angle"] = _angle(lm[mcp], lm[pip], lm[dip])
        m[f"{name}_reach"] = _dist(wrist, lm[tip]) / (_dist(wrist, lm[pip]) or 1e-6)
    m["thumb_angle"] = _angle(lm[THUMB_MCP], lm[THUMB_IP], lm[THUMB_TIP])
    m["thumb_away"] = _dist(lm[THUMB_TIP], lm[PINKY_MCP]) / (_dist(lm[THUMB_IP], lm[PINKY_MCP]) or 1e-6)
    m["thumb_out"] = _dist(lm[THUMB_TIP], lm[INDEX_MCP]) / palm
    m["pinch"] = _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm
    # thumb vs the fist: clearance from the index/middle joints, and reach from the wrist
    m["thumb_clear"] = min(_dist(lm[THUMB_TIP], lm[j]) for j in (5, 6, 7, 8, 9, 10, 11, 12)) / palm
    m["thumb_reach"] = _dist(lm[THUMB_TIP], wrist) / palm
    # thumb direction: angle of thumb MCP -> tip from straight up (0 = up, 180 = down)
    dx = lm[THUMB_TIP][0] - lm[THUMB_MCP][0]
    dy = lm[THUMB_TIP][1] - lm[THUMB_MCP][1]
    m["thumb_from_up"] = math.degrees(math.atan2(abs(dx), -dy)) if (dx or dy) else 90.0
    return m


def finger_tristate(m: dict[str, float]) -> dict[str, str]:
    """ext / curl / amb per finger from finger_metrics()."""
    out = {}
    for f in FOUR:
        a, r = m[f"{f}_angle"], m[f"{f}_reach"]
        if a <= FINGER_CURL_ANGLE or r <= FINGER_CURL_REACH:
            out[f] = CURL
        elif (a >= FINGER_EXT_STRAIGHT[0] and r >= FINGER_EXT_STRAIGHT[1]) or \
                (a >= FINGER_EXT_REACH[0] and r >= FINGER_EXT_REACH[1]):
            out[f] = EXT
        else:
            out[f] = AMB
    away, ang, tout = m["thumb_away"], m["thumb_angle"], m["thumb_out"]
    if away <= THUMB_CURL["away"] or ang <= THUMB_CURL["angle"] or tout <= THUMB_CURL["out"]:
        out["thumb"] = CURL
    elif away >= THUMB_EXT["away"] and ang >= THUMB_EXT["angle"] and tout >= THUMB_EXT["out"]:
        out["thumb"] = EXT
    else:
        out["thumb"] = AMB
    return out


def thumb_out_of_fist(m: dict[str, float]) -> bool:
    """The thumb sticks clearly out of the hand: roughly straight (a slight bend is fine),
    pointing away from the palm, and clear of the curled fingers (or far from the wrist)."""
    t = THUMB_OUT
    away, clear, reach = m["thumb_away"], m["thumb_clear"], m["thumb_reach"]
    return (m["thumb_angle"] >= t["angle"] and away >= t["away_min"] and clear >= t["clear_min"]
            and (clear >= t["clear"] or reach >= t["reach"])
            and (away >= t["away"] or reach >= t["reach"]))  # leaning over the fist: reach decides


def thumb_tucked(m: dict[str, float]) -> bool:
    """Thumb folded against / across the fist (not merely 'not sticking out')."""
    t = THUMB_TUCKED
    return m["thumb_reach"] < t["reach"] or m["thumb_away"] < t["away"] or m["thumb_clear"] < t["clear"]


def fingers_folded(tri: dict[str, str], m: dict[str, float]) -> bool:
    """All four fingers folded for a thumbs pose: curled, or ambiguous but short (a loose
    fist). No finger may be clearly extended."""
    return all(tri[f] == CURL or (tri[f] == AMB and m[f"{f}_reach"] < FOLDED_REACH) for f in FOUR)


def thumbs_pose(lm, max_deg: float = THUMBS_STAY_DEG) -> str | None:
    """'thumbs_up' / 'thumbs_down' if the hand is a thumbs pose within max_deg of vertical."""
    m = finger_metrics(lm)
    if not (fingers_folded(finger_tristate(m), m) and thumb_out_of_fist(m)):
        return None
    if m["thumb_from_up"] <= max_deg:
        return "thumbs_up"
    if m["thumb_from_up"] >= 180 - max_deg:
        return "thumbs_down"
    return None


def _lean(f: str, m: dict[str, float]) -> bool:
    """Best single guess for an ambiguous finger (used only for display/bool APIs)."""
    if f == "thumb":
        return m["thumb_away"] >= 1.0 and m["thumb_angle"] >= 135 and m["thumb_out"] >= 0.35
    return m[f"{f}_angle"] >= 140 and m[f"{f}_reach"] >= 1.0


def finger_states(lm, prev: dict[str, bool] | None = None) -> dict[str, bool]:
    """{finger: is_extended} for all five fingers. Ambiguous fingers keep `prev` if given."""
    m = finger_metrics(lm)
    tri = finger_tristate(m)
    short = _short_ambiguous(tri, m)
    return {f: (tri[f] == EXT) if tri[f] != AMB else
            (False if f in short else prev[f] if prev and f in prev else _lean(f, m))
            for f in ALL_FIVE}


def _short_ambiguous(tri, m) -> set:
    """Ambiguous fingers whose tip is closer to the wrist than their PIP joint (reach < 1):
    half-bent, but they can't be extended, so they only ever resolve as folded."""
    return {f for f in FOUR if tri[f] == AMB and m[f"{f}_reach"] < FOLDED_REACH}


# ------------------------------------------------------------------ rules
def _rules(s: dict[str, bool], lm, m, ok_pinch_ratio: float, tri=None, up_deg=THUMBS_ENTER_DEG,
           down_deg=THUMBS_ENTER_DEG):
    """(label, [(condition, passed), ...]) in priority order; the first full match wins."""
    four_up = [s[f] for f in FOUR]
    tri = tri or finger_tristate(m)
    ang = m["thumb_from_up"]
    up, down = ang <= up_deg, ang >= 180 - down_deg
    curled4 = ("4 fingers curled", not any(four_up))
    folded4 = ("4 fingers folded", fingers_folded(tri, m))
    out = thumb_out_of_fist(m)
    thumb_out = ("thumb out of fist", out)
    return [
        # 👌 first: the index state is unreliable while it curls into the circle
        ("ok", [("middle up", s["middle"]), ("ring up", s["ring"]), ("pinky up", s["pinky"]),
                (f"pinch {m['pinch']:.2f} < {ok_pinch_ratio:.2f}", m["pinch"] < ok_pinch_ratio)]),
        ("open_palm", [("thumb up", s["thumb"]), ("index up", s["index"]), ("middle up", s["middle"]),
                       ("ring up", s["ring"]), ("pinky up", s["pinky"])]),
        ("thumbs_up", [folded4, thumb_out, (f"thumb {ang:.0f}° from up <= {up_deg:.0f}°", up)]),
        ("thumbs_down", [folded4, thumb_out, (f"thumb {180 - ang:.0f}° from down <= {down_deg:.0f}°", down)]),
        ("unknown", [folded4, thumb_out, ("thumb sideways", not up and not down)]),
        # a thumb sticking out of the fist is never a fist (tilted thumbs-ups used to fire Mute)
        ("fist", [curled4, ("thumb tucked", not s["thumb"] and not out and thumb_tucked(m))]),
        # 🤟 / 🤘: index + pinky up, middle + ring down; the thumb may be out or tucked
        ("rock", [("index up", s["index"]), ("pinky up", s["pinky"]),
                  ("middle down", not s["middle"]), ("ring down", not s["ring"])]),
        ("call_me", [("thumb up", s["thumb"]), ("pinky up", s["pinky"]), ("index down", not s["index"]),
                     ("middle down", not s["middle"]), ("ring down", not s["ring"])]),
        ("peace", [("index up", s["index"]), ("middle up", s["middle"]),
                   ("ring down", not s["ring"]), ("pinky down", not s["pinky"])]),
        ("pointing", [("index up", s["index"]), ("middle down", not s["middle"]),
                      ("ring down", not s["ring"]), ("pinky down", not s["pinky"])]),
    ]


def _decide(s, lm, m, ok_pinch_ratio, tri=None, prev_label=None):
    # hysteresis on the thumb direction: once thumbs up/down, stay until 70° (enter at 50°)
    up_deg = THUMBS_STAY_DEG if prev_label == "thumbs_up" else THUMBS_ENTER_DEG
    down_deg = THUMBS_STAY_DEG if prev_label == "thumbs_down" else THUMBS_ENTER_DEG
    rules = _rules(s, lm, m, ok_pinch_ratio, tri, up_deg, down_deg)
    for i, (label, conds) in enumerate(rules):
        if all(ok for _, ok in conds):
            if label == "unknown":
                return "unknown", "thumb points sideways (not up or down)"
            return label, ""
    # nearest rules = fewest failed conditions; report what blocked each of them
    fails = [(label, [n for n, ok in conds if not ok]) for label, conds in rules if label != "unknown"]
    best = min(len(f) for _, f in fails)
    near = [f"{label} ({', '.join(f)})" for label, f in fails if len(f) == best]
    return "unknown", "blocked: " + "; ".join(near[:3])


@dataclass
class PoseInfo:
    label: str
    states: dict[str, bool]  # resolved per-finger states
    tri: dict[str, str]  # this frame's ext / curl / amb
    metrics: dict[str, float] = field(default_factory=dict)
    reason: str = ""  # why it's unknown / what blocked the closest gesture


def explain(lm, ok_pinch_ratio: float = OK_PINCH_RATIO, prev: dict[str, bool] | None = None,
            prev_label: str | None = None) -> PoseInfo:
    """Classify and say why. `prev` = last frame's resolved states and `prev_label` the
    last named gesture, for hysteresis (finger states / thumbs direction)."""
    if lm is None or len(lm) != 21:
        return PoseInfo("unknown", {f: False for f in ALL_FIVE}, {f: AMB for f in ALL_FIVE},
                        reason="need 21 landmarks")
    m = finger_metrics(lm)
    tri = finger_tristate(m)
    short = _short_ambiguous(tri, m)  # half-bent but short: folded, never extended
    amb = [f for f in ALL_FIVE if tri[f] == AMB and f not in short]
    base = {f: tri[f] == EXT for f in ALL_FIVE}

    if prev is not None:  # hysteresis: ambiguous fingers keep their last state
        states = {f: (prev.get(f, _lean(f, m)) if f in amb else base[f]) for f in ALL_FIVE}
        label, reason = _decide(states, lm, m, ok_pinch_ratio, tri, prev_label)
    else:  # no history: only answer if every reading of the ambiguous fingers agrees
        states = {f: (_lean(f, m) if f in amb else base[f]) for f in ALL_FIVE}
        outcomes = {}
        for combo in itertools.product((True, False), repeat=len(amb)):
            s = dict(base, **dict(zip(amb, combo)))
            outcomes.setdefault(_decide(s, lm, m, ok_pinch_ratio, tri, prev_label)[0], s)
        if len(outcomes) == 1:
            label = next(iter(outcomes))
            reason = "" if label != "unknown" else _decide(states, lm, m, ok_pinch_ratio, tri, prev_label)[1]
        else:
            label = "unknown"
            reason = f"ambiguous {'/'.join(amb)}: could be {' or '.join(sorted(outcomes))}"

    # never fire Mute on an unclear hand: fist needs all four fingers *clearly* curled
    if label == "fist" and any(tri[f] == AMB for f in FOUR):
        fuzzy = [f for f in FOUR if tri[f] == AMB]
        label, reason = "unknown", f"fist needs clearly curled fingers ({'/'.join(fuzzy)} unclear)"
    return PoseInfo(label, states, tri, m, reason)


def classify(lm, ok_pinch_ratio: float = OK_PINCH_RATIO, prev: dict[str, bool] | None = None,
             prev_label: str | None = None) -> str:
    """Classify a single frame of 21 (x, y[, z]) landmarks (see explain())."""
    return explain(lm, ok_pinch_ratio, prev, prev_label).label


class PoseTracker:
    """Per-hand classifier with finger-state hysteresis across frames."""

    def __init__(self, ok_pinch_ratio: float = OK_PINCH_RATIO):
        self.ok_pinch_ratio = ok_pinch_ratio
        self.prev: dict[str, bool] | None = None
        self.prev_label: str | None = None  # last *named* gesture (survives unknown frames)
        self.last: PoseInfo | None = None

    def update(self, lm) -> PoseInfo:
        info = explain(lm, self.ok_pinch_ratio, self.prev, self.prev_label)
        self.prev, self.last = info.states, info
        if info.label != "unknown":
            self.prev_label = info.label
        elif info.reason.startswith("thumb points sideways"):
            self.prev_label = None  # tilted past the stay limit: must come back within 50° to re-enter
        return info

    def reset(self):
        self.prev = self.last = self.prev_label = None


class GestureSmoother:
    """Majority vote over the last N frames to remove single-frame flicker."""

    def __init__(self, window: int = 7, min_ratio: float = 0.6):
        self.history: deque[str] = deque(maxlen=window)
        self.min_ratio = min_ratio

    def update(self, gesture: str) -> str:
        self.history.append(gesture)
        label, count = Counter(self.history).most_common(1)[0]
        if len(self.history) == self.history.maxlen and count / len(self.history) >= self.min_ratio:
            return label
        return "unknown"

    def reset(self):
        self.history.clear()


class HoldThrough:
    """Keeps a confirmed directional gesture (thumbs up/down) alive through short gaps.

    Feed it the smoother's output each frame. Once thumbs_up/down is confirmed it keeps
    reporting it through up to `grace` seconds of "unknown" (ambiguous / dropped frames),
    so a held volume repeat doesn't stop. Any *other* confirmed gesture ends it at once.
    """

    def __init__(self, gestures=DIRECTIONAL_GESTURES, grace: float = 0.3):
        self.gestures, self.grace = set(gestures), grace
        self.held: str | None = None
        self._seen = float("-inf")

    def update(self, t: float, stable: str) -> str:
        if stable in self.gestures:
            self.held, self._seen = stable, t
            return stable
        if stable != "unknown":  # another clearly confirmed gesture ends the hold
            self.held = None
            return stable
        if self.held and t - self._seen <= self.grace + 1e-9:
            return self.held
        self.held = None
        return stable

    def reset(self):
        self.held = None
