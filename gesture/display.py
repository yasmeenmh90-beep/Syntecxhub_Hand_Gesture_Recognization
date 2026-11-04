"""Pure display helpers for the live gesture overlay (demo.py).

Nothing here classifies or fires anything: it only turns the classifier/smoother state
into what's drawn (colours, ring progress, animation curves, smoothed positions).
All functions are side-effect free or small state holders, so they are unit-tested.
"""
from __future__ import annotations

import math
from collections import Counter, deque

# ------------------------------------------------------------------ colours (BGR)
GESTURE_COLORS = {
    "open_palm":   (216, 79, 255),   # pink    #FF4FD8
    "fist":        (80, 80, 255),    # red     #FF5050
    "thumbs_up":   (120, 220, 60),   # green   #3CDC78
    "thumbs_down": (60, 154, 255),   # orange  #FF9A3C
    "peace":       (240, 230, 46),   # cyan    #2EE6F0
    "pointing":    (246, 92, 139),   # purple  #8B5CF6
    "rock":        (251, 64, 224),   # magenta #E040FB
    "ok":          (61, 217, 255),   # yellow  #FFD93D
    "call_me":     (166, 201, 32),   # teal    #20C9A6
    "pinch":       (255, 180, 90),   # sky blue #5AB4FF (volume slider)
    "swipe_left":  (216, 79, 255),   # swipes are open-palm motions
    "swipe_right": (216, 79, 255),
}
UNKNOWN_COLOR = (150, 140, 150)  # muted grey


def gesture_color(gesture: str | None):
    return GESTURE_COLORS.get(gesture or "", UNKNOWN_COLOR)


def lerp_color(a, b, t: float):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


class ColorFade:
    """Smoothly fades to a new target colour over `duration` seconds (default 150 ms)."""

    def __init__(self, color=UNKNOWN_COLOR, duration: float = 0.15):
        self.duration = duration
        self.start = self.target = tuple(color)
        self.t0 = float("-inf")

    def set(self, color, now: float):
        color = tuple(color)
        if color != self.target:
            self.start = self.value(now)
            self.target = color
            self.t0 = now

    def value(self, now: float):
        if self.duration <= 0:
            return self.target
        return lerp_color(self.start, self.target, (now - self.t0) / self.duration)

    def done(self, now: float) -> bool:
        return now - self.t0 >= self.duration - 1e-9


# ------------------------------------------------------------------ hold-to-fire ring
def needed_votes(window: int, min_ratio: float) -> int:
    """How many matching frames the smoother needs in a full window."""
    return max(1, math.ceil(min_ratio * window - 1e-9))


def hold_progress(history, window: int, min_ratio: float, stable: str | None):
    """(label, progress 0..1, confirmed) for the hold-to-fire ring.

    Mirrors GestureSmoother's rule (full window and top-label share >= min_ratio) without
    changing it: a confirmed gesture is 1.0; otherwise the leading label's votes over the
    votes it needs, capped just below 1 until the smoother actually confirms it.
    """
    if stable and stable != "unknown":
        return stable, 1.0, True
    hist = list(history)
    if not hist:
        return "unknown", 0.0, False
    top, n = Counter(hist).most_common(1)[0]
    if top == "unknown":
        return "unknown", 0.0, False
    return top, min(n / needed_votes(window, min_ratio), 0.99), False


# ------------------------------------------------------------------ animation curves
def _ease_out(t: float) -> float:
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return 1 - (1 - t) ** 3


def pop_curve(t: float, duration: float = 0.8):
    """Big-emoji pop: scale 0.6 -> 1.1 -> 1.0, fading out by `duration`. None when over."""
    if t < 0 or t > duration:
        return None
    up, settle = 0.12, 0.28
    if t <= up:
        scale = 0.6 + 0.5 * _ease_out(t / up)
    elif t <= settle:
        scale = 1.1 - 0.1 * _ease_out((t - up) / (settle - up))
    else:
        scale = 1.0
    fade_start = 0.45 * duration
    alpha = 1.0 if t <= fade_start else max(0.0, 1 - (t - fade_start) / (duration - fade_start))
    return scale, alpha


def float_curve(t: float, duration: float = 1.0):
    """Action pill floating up from the hand: (rise 0..1, alpha). None when over."""
    if t < 0 or t > duration:
        return None
    alpha = 1.0 if t <= 0.6 * duration else max(0.0, 1 - (t - 0.6 * duration) / (0.4 * duration))
    return _ease_out(t / duration), alpha


def flash_curve(t: float, duration: float = 0.35):
    """Ring flash when an action fires: (scale, white mix). None when over."""
    if t < 0 or t > duration:
        return None
    k = t / duration
    return 1.0 + 0.3 * math.sin(math.pi * k), 1.0 - k


# ------------------------------------------------------------------ smoothing + placement
def clamp_rect(rect, frame_w: int, frame_h: int):
    """Shift a rect so it lies inside the frame (shrinking only if it's bigger)."""
    x1, y1, x2, y2 = rect
    w, h = min(x2 - x1, frame_w), min(y2 - y1, frame_h)
    x1 = min(max(x1, 0), frame_w - w)
    y1 = min(max(y1, 0), frame_h - h)
    return (x1, y1, x1 + w, y1 + h)


def overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


class RectSmoother:
    """EMA on a rect (x1, y1, x2, y2). The first rect (or one after reset) snaps."""

    def __init__(self, alpha: float = 0.35):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, rect):
        if self.value is None:
            self.value = tuple(float(v) for v in rect)
        else:
            a = self.alpha
            self.value = tuple(v + a * (r - v) for v, r in zip(self.value, rect))
        return tuple(int(round(v)) for v in self.value)


class PointSmoother(RectSmoother):
    """EMA on a point (x, y)."""


def place_tag(box, tag_w: int, tag_h: int, frame_w: int, frame_h: int, avoid=(), gap: int = 6):
    """Where to put a hand tag: above / below / inside the box, inside the frame and not
    over any `avoid` rect (HUD, legend, ...). Returns (x, y) or None if nothing fits."""
    bx1, by1, bx2, by2 = box
    candidates = [(bx1, by1 - tag_h - gap), (bx2 - tag_w, by1 - tag_h - gap),
                  (bx1, by2 + gap), (bx2 - tag_w, by2 + gap), (bx1 + gap, by1 + gap)]
    for x, y in candidates:
        rect = clamp_rect((x, y, x + tag_w, y + tag_h), frame_w, frame_h)
        if not any(overlaps(rect, r) for r in avoid):
            return rect[0], rect[1]
    return None


# ------------------------------------------------------------------ action feedback
ACTION_ICONS = {
    "media_play_pause": "⏯", "media_next": "⏭", "media_previous": "⏮",
    "media_volume_up": "🔊", "media_volume_down": "🔉", "media_volume_mute": "🔇",
    "right": "➡️", "left": "⬅️", "f5": "▶️",
    "toggle_dry_run": "🔁", "screenshot": "📸", "open_spotify": "🎵",
}


def action_icon(binding) -> str:
    """Emoji for a config Binding (by command, else by its first key)."""
    if binding is None:
        return "✨"
    if getattr(binding, "command", None):
        return ACTION_ICONS.get(binding.command, "✨")
    keys = getattr(binding, "keys", ()) or ()
    return ACTION_ICONS.get(keys[0], "⌨️") if keys else "✨"


class VolumeMeter:
    """Approximate volume for the meter next to the hand while thumbs up/down repeat.

    The system volume can't be read back cheaply, so this is an estimate: it starts at 50
    (or the last pinch-slider value) and moves one media-key step per repeat.
    """

    STEP = 100 / 16  # macOS volume keys move in 16 steps

    def __init__(self, level: float = 50.0, linger: float = 0.8):
        self.level = float(level)
        self.linger = linger
        self.until = float("-inf")
        self.recent: deque[float] = deque(maxlen=8)  # tick times, for the bump animation

    def set_level(self, level: float):
        self.level = float(min(100, max(0, level)))

    def tick(self, direction: int, now: float):
        self.set_level(self.level + direction * self.STEP)
        self.until = now + self.linger
        self.recent.append(now)

    def visible(self, now: float, holding: bool) -> bool:
        return holding or now < self.until
