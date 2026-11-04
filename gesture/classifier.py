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


def _lean(f: str, m: dict[str, float]) -> bool:
    """Best single guess for an ambiguous finger (used only for display/bool APIs)."""
    if f == "thumb":
        return m["thumb_away"] >= 1.0 and m["thumb_angle"] >= 135 and m["thumb_out"] >= 0.35
    return m[f"{f}_angle"] >= 140 and m[f"{f}_reach"] >= 1.0


def finger_states(lm, prev: dict[str, bool] | None = None) -> dict[str, bool]:
    """{finger: is_extended} for all five fingers. Ambiguous fingers keep `prev` if given."""
    m = finger_metrics(lm)
    tri = finger_tristate(m)
    return {f: (tri[f] == EXT) if tri[f] != AMB else (prev[f] if prev and f in prev else _lean(f, m))
            for f in ALL_FIVE}


# ------------------------------------------------------------------ rules
def _rules(s: dict[str, bool], lm, m, ok_pinch_ratio: float):
    """(label, [(condition, passed), ...]) in priority order; the first full match wins."""
    four_up = [s[f] for f in FOUR]
    dx = lm[THUMB_TIP][0] - lm[THUMB_MCP][0]
    dy = lm[THUMB_TIP][1] - lm[THUMB_MCP][1]
    norm = math.hypot(dx, dy) or 1e-6
    up, down = dy / norm < -0.6, dy / norm > 0.6
    curled4 = ("4 fingers curled", not any(four_up))
    return [
        # 👌 first: the index state is unreliable while it curls into the circle
        ("ok", [("middle up", s["middle"]), ("ring up", s["ring"]), ("pinky up", s["pinky"]),
                (f"pinch {m['pinch']:.2f} < {ok_pinch_ratio:.2f}", m["pinch"] < ok_pinch_ratio)]),
        ("open_palm", [("thumb up", s["thumb"]), ("index up", s["index"]), ("middle up", s["middle"]),
                       ("ring up", s["ring"]), ("pinky up", s["pinky"])]),
        ("thumbs_up", [curled4, ("thumb up", s["thumb"]), ("thumb points up", up)]),
        ("thumbs_down", [curled4, ("thumb up", s["thumb"]), ("thumb points down", down)]),
        ("unknown", [curled4, ("thumb up", s["thumb"]), ("thumb sideways", not up and not down)]),
        ("fist", [curled4, ("thumb tucked", not s["thumb"])]),
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


def _decide(s, lm, m, ok_pinch_ratio):
    rules = _rules(s, lm, m, ok_pinch_ratio)
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


def explain(lm, ok_pinch_ratio: float = OK_PINCH_RATIO, prev: dict[str, bool] | None = None) -> PoseInfo:
    """Classify and say why. `prev` = last frame's resolved states, for hysteresis."""
    if lm is None or len(lm) != 21:
        return PoseInfo("unknown", {f: False for f in ALL_FIVE}, {f: AMB for f in ALL_FIVE},
                        reason="need 21 landmarks")
    m = finger_metrics(lm)
    tri = finger_tristate(m)
    amb = [f for f in ALL_FIVE if tri[f] == AMB]
    base = {f: tri[f] == EXT for f in ALL_FIVE}

    if prev is not None:  # hysteresis: ambiguous fingers keep their last state
        states = {f: (prev.get(f, _lean(f, m)) if tri[f] == AMB else base[f]) for f in ALL_FIVE}
        label, reason = _decide(states, lm, m, ok_pinch_ratio)
    else:  # no history: only answer if every reading of the ambiguous fingers agrees
        states = {f: (_lean(f, m) if tri[f] == AMB else base[f]) for f in ALL_FIVE}
        outcomes = {}
        for combo in itertools.product((True, False), repeat=len(amb)):
            s = dict(base, **dict(zip(amb, combo)))
            outcomes.setdefault(_decide(s, lm, m, ok_pinch_ratio)[0], s)
        if len(outcomes) == 1:
            label = next(iter(outcomes))
            reason = "" if label != "unknown" else _decide(states, lm, m, ok_pinch_ratio)[1]
        else:
            label = "unknown"
            reason = f"ambiguous {'/'.join(amb)}: could be {' or '.join(sorted(outcomes))}"

    # never fire Mute on an unclear hand: fist needs all four fingers *clearly* curled
    if label == "fist" and any(tri[f] == AMB for f in FOUR):
        fuzzy = [f for f in FOUR if tri[f] == AMB]
        label, reason = "unknown", f"fist needs clearly curled fingers ({'/'.join(fuzzy)} unclear)"
    return PoseInfo(label, states, tri, m, reason)


def classify(lm, ok_pinch_ratio: float = OK_PINCH_RATIO, prev: dict[str, bool] | None = None) -> str:
    """Classify a single frame of 21 (x, y[, z]) landmarks (see explain())."""
    return explain(lm, ok_pinch_ratio, prev).label


class PoseTracker:
    """Per-hand classifier with finger-state hysteresis across frames."""

    def __init__(self, ok_pinch_ratio: float = OK_PINCH_RATIO):
        self.ok_pinch_ratio = ok_pinch_ratio
        self.prev: dict[str, bool] | None = None
        self.last: PoseInfo | None = None

    def update(self, lm) -> PoseInfo:
        info = explain(lm, self.ok_pinch_ratio, self.prev)
        self.prev, self.last = info.states, info
        return info

    def reset(self):
        self.prev = self.last = None


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
