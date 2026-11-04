"""Motion and continuous controls on top of the static poses in classifier.py.

- SwipeDetector  : open-palm swipe left/right from the palm-centre trajectory
- VolumeSlider   : thumb-index pinch distance -> system volume 0-100
- MouseLogic     : index fingertip -> cursor, pinch = click/drag, peace = scroll
- HoldTimer      : "both hands open for 1 s" exit gesture

The detectors are pure (they take timestamps + landmarks and return decisions), so
they are unit-tested without a camera, mouse or speakers. The thin OS backends at the
bottom (VirtualMouse, set_system_volume, screen_size) are the only side effects.
"""
from __future__ import annotations

import math
import subprocess
import sys
from collections import deque
from dataclasses import dataclass, field

from .classifier import MIDDLE_MCP, THUMB_TIP, WRIST, finger_metrics, finger_states, thumbs_pose

INDEX_TIP, MIDDLE_TIP = 8, 12
PALM_IDX = (0, 5, 9, 13, 17)


# ------------------------------------------------------------------ geometry
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def palm_size(lm) -> float:
    return _dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6


def palm_center(lm) -> tuple[float, float]:
    """Mean of wrist + the four finger MCPs, in normalised image coordinates."""
    return (sum(lm[i][0] for i in PALM_IDX) / 5, sum(lm[i][1] for i in PALM_IDX) / 5)


def pinch_ratio(lm) -> float:
    """Thumb tip to index tip distance, in palm sizes (scale-invariant)."""
    return _dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm_size(lm)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ------------------------------------------------------------------ swipe
class SwipeDetector:
    """Detect horizontal open-palm swipes from the palm-centre trajectory.

    A swipe: within the last `window` seconds the palm travelled more than
    `min_travel` (fraction of frame width) horizontally, the vertical spread stayed
    below `max_vertical_ratio` x that travel, and most samples were open_palm.

    `gate()` also stops the static open_palm action from firing when the palm is
    moving or has just swiped: open_palm only passes once the palm has been still for
    `still_hold` seconds, and not again until the pose changes after a swipe.
    """

    def __init__(self, window=0.4, min_travel=0.25, max_vertical_ratio=0.5, cooldown=1.0,
                 still_hold=0.3, still_eps=0.04, pose="open_palm", min_pose_fraction=0.6):
        self.window, self.min_travel = window, min_travel
        self.max_vertical_ratio, self.cooldown = max_vertical_ratio, cooldown
        self.still_hold, self.still_eps = still_hold, still_eps
        self.pose, self.min_pose_fraction = pose, min_pose_fraction
        self.samples: deque[tuple[float, float, float, bool]] = deque()
        self.last_fire = float("-inf")
        self.consumed = False  # a swipe used up the current open palm

    def reset(self):
        self.samples.clear()
        self.consumed = False

    def update(self, t: float, center: tuple[float, float], pose: str) -> str | None:
        """Add one sample; returns "swipe_left" / "swipe_right" when a swipe completes."""
        self.samples.append((t, center[0], center[1], pose == self.pose))
        keep = max(self.window, self.still_hold)
        while self.samples and t - self.samples[0][0] > keep:
            self.samples.popleft()

        win = [s for s in self.samples if t - s[0] <= self.window]
        if len(win) < 3 or t - self.last_fire < self.cooldown:
            return None
        if sum(s[3] for s in win) < self.min_pose_fraction * len(win):
            return None
        dx = win[-1][1] - win[0][1]
        ys = [s[2] for s in win]
        dy = max(ys) - min(ys)
        if abs(dx) > self.min_travel and dy < self.max_vertical_ratio * abs(dx):
            self.last_fire = t
            self.consumed = True
            self.samples.clear()
            return "swipe_right" if dx > 0 else "swipe_left"
        return None

    def is_still(self, t: float) -> bool:
        if not self.samples or t - self.samples[0][0] < 0.9 * self.still_hold:
            return False  # not enough history yet
        recent = [s for s in self.samples if t - s[0] <= self.still_hold + 1e-6]
        xs, ys = [s[1] for s in recent], [s[2] for s in recent]
        return max(xs) - min(xs) < self.still_eps and max(ys) - min(ys) < self.still_eps

    def gate(self, stable: str, t: float) -> str:
        """What the static dispatcher should see this frame."""
        if stable != self.pose:
            self.consumed = False
            return stable
        if self.consumed or not self.is_still(t):
            return "unknown"
        return stable


# ------------------------------------------------------------------ pinch volume
def pinch_to_volume(ratio: float, min_ratio: float = 0.25, max_ratio: float = 1.1) -> int:
    """Linear map of the pinch ratio onto 0-100, clamped."""
    if max_ratio <= min_ratio:
        raise ValueError("max_ratio must be > min_ratio")
    return int(round(100 * clamp((ratio - min_ratio) / (max_ratio - min_ratio), 0.0, 1.0)))


def index_reach(lm) -> float:
    """Wrist to index tip, in palm sizes. ~0.7 in a fist, > 1 when pinching or pointing."""
    return _dist(lm[WRIST], lm[INDEX_TIP]) / palm_size(lm)


def is_pinch_pose(lm, states=None, thumb_near: float = 0.45, min_index_reach: float = 0.9) -> bool:
    """Thumb + index only: middle/ring/pinky curled, the index not folded into the palm,
    and the thumb either pointing away from the palm or near the index tip.

    Robust across the whole gap: when the tips touch, the index bends and the thumb reads
    as tucked; half-open, the thumb is between states. A plain pointing hand (thumb tucked
    toward the pinky, far from the index tip) and a fist are *not* a pinch pose.
    """
    s = states or finger_states(lm)
    if s["middle"] or s["ring"] or s["pinky"] or index_reach(lm) < min_index_reach:
        return False
    if thumbs_pose(lm):  # a (loose-fist) thumbs up/down must never start pinch-volume mode
        return False
    return pinch_ratio(lm) < thumb_near or finger_metrics(lm)["thumb_away"] >= 1.0


class VolumeSlider:
    """Pinch distance -> smoothed volume; `update` returns the value to send (or None)."""

    def __init__(self, min_ratio=0.25, max_ratio=1.1, step=3, alpha=0.4):
        self.min_ratio, self.max_ratio = min_ratio, max_ratio
        self.step, self.alpha = step, alpha
        self.value: float | None = None  # smoothed, 0-100
        self.sent: int | None = None

    @property
    def level(self) -> int:
        return int(round(self.value)) if self.value is not None else 0

    def reset(self):
        self.value = None  # keep `sent` so re-pinching at the same level sends nothing

    def update(self, ratio: float) -> int | None:
        target = pinch_to_volume(ratio, self.min_ratio, self.max_ratio)
        self.value = target if self.value is None else self.value + self.alpha * (target - self.value)
        v = self.level
        if self.sent is None or abs(v - self.sent) >= self.step or (v in (0, 100) and v != self.sent):
            self.sent = v
            return v
        return None


@dataclass
class PinchState:
    active: bool = False    # pinch mode on: volume follows the gap, HUD/tag show "Pinch"
    gating: bool = False    # block every static action this frame (on from the first pinch frame)
    level: int | None = None  # smoothed volume 0-100 while active
    send: int | None = None   # value to send to the OS this frame (changed by >= step), if any
    ended: bool = False     # pinch mode just ended: the caller resets the gesture smoother


class PinchController:
    """Pinch-volume mode with debounced start, a short exit grace, and clean hand-off.

    - Static actions are gated from the very first pinch-pose frame.
    - The volume only follows the gap after `enter_frames` consecutive pinch frames, so a
      one-frame misread can't jump it.
    - Mode stays on for `exit_grace` s after the last pinch frame (brief misreads).
    - On exit `ended` is set: the caller clears the smoother so the L-shape's leftover
      "pointing" vote can't fire Previous track the instant the pinch ends.
    """

    def __init__(self, slider: "VolumeSlider", enter_frames: int = 2, exit_grace: float = 0.3):
        self.slider = slider
        self.enter_frames, self.exit_grace = enter_frames, exit_grace
        self.active = False
        self._streak = 0
        self._last_pose_t = float("-inf")

    def reset(self) -> bool:
        """Hand lost / mode switch. Returns True if pinch mode was active."""
        was = self.active
        self.active, self._streak = False, 0
        self.slider.reset()
        return was

    def update(self, t: float, iso, states=None, enabled: bool = True) -> PinchState:
        pose = bool(enabled and iso is not None and is_pinch_pose(iso, states))
        out = PinchState()
        if pose:
            self._streak += 1
            self._last_pose_t = t
            if self._streak >= self.enter_frames:
                self.active = True
        else:
            self._streak = 0
            if self.active and (not enabled or t - self._last_pose_t > self.exit_grace):
                self.active = False
                out.ended = True
                self.slider.reset()
        out.active = self.active
        out.gating = self.active or pose
        if self.active and pose:
            out.send = self.slider.update(pinch_ratio(iso))
        if self.active:
            out.level = self.slider.level
        return out


# ------------------------------------------------------------------ virtual mouse
def map_to_screen(x: float, y: float, screen_w: int, screen_h: int, box: float = 0.6):
    """Map the centre `box` fraction of the frame (normalised x, y) onto the full screen."""
    m = (1 - box) / 2
    u = clamp((x - m) / box, 0.0, 1.0)
    v = clamp((y - m) / box, 0.0, 1.0)
    return u * (screen_w - 1), v * (screen_h - 1)


class EMA2D:
    """Exponential moving average for a point. alpha = weight of the newest sample."""

    def __init__(self, alpha: float = 0.35):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: tuple[float, float] | None = None

    def reset(self):
        self.value = None

    def update(self, p):
        if self.value is None:
            self.value = (float(p[0]), float(p[1]))
        else:
            a = self.alpha
            self.value = (self.value[0] + a * (p[0] - self.value[0]),
                          self.value[1] + a * (p[1] - self.value[1]))
        return self.value


@dataclass
class MouseFrame:
    cursor: tuple[float, float] | None = None  # smoothed screen position (None = not moving)
    events: list[tuple] = field(default_factory=list)  # ("move",x,y) ("press",) ("release",) ("scroll",n)
    pinch: float = 0.0
    pressed: bool = False
    scrolling: bool = False


class MouseLogic:
    """Turn landmarks into mouse events (no OS calls).

    - index fingertip moves the cursor (centre box -> screen, EMA smoothed)
    - pinch (thumb tip–index tip < click_ratio palm sizes) presses the left button,
      un-pinch releases it: a quick pinch is a click, holding it while moving is a drag.
      Release uses hysteresis (release_factor) and a new press needs `debounce` s.
      If middle, ring and pinky are all up it's the 👌 gesture, not a click; and the
      index tip must reach > min_reach palm sizes from the wrist, so a fist (thumb
      tucked onto a curled index) isn't a click either.
    - peace pose (index + middle up) scrolls with vertical hand movement.
    """

    def __init__(self, screen_w: int, screen_h: int, alpha=0.35, box=0.6, click_ratio=0.30,
                 release_factor=1.4, debounce=0.3, confirm_frames=2, click_freeze=0.15,
                 scroll_gain=25.0, min_reach=1.0):
        self.sw, self.sh = screen_w, screen_h
        self.box, self.click_ratio, self.release_factor = box, click_ratio, release_factor
        self.debounce, self.confirm_frames = debounce, confirm_frames
        self.click_freeze, self.scroll_gain = click_freeze, scroll_gain
        self.min_reach = min_reach
        self.ema = EMA2D(alpha)
        self.pressed = False
        self._pinch_frames = 0
        self._last_release = float("-inf")
        self._press_t = float("-inf")
        self._scroll_y: float | None = None
        self._scroll_acc = 0.0

    def reset(self) -> list[tuple]:
        """Hand lost: release the button if held."""
        self.ema.reset()
        self._pinch_frames, self._scroll_y, self._scroll_acc = 0, None, 0.0
        if self.pressed:
            self.pressed = False
            return [("release",)]
        return []

    def update(self, t: float, lm, iso=None, states=None) -> MouseFrame:
        """`lm`: normalised landmarks (cursor position). `iso`: the same hand in isotropic
        units (Hand.iso) for pose/pinch decisions; defaults to `lm`. `states`: finger
        states from the caller's PoseTracker (hysteresis), if any."""
        iso = lm if iso is None else iso
        s = states or finger_states(iso)
        out = MouseFrame(pinch=pinch_ratio(iso))
        ok_pose = s["middle"] and s["ring"] and s["pinky"]

        # ---- scroll (peace): no cursor movement, no clicks
        if s["index"] and s["middle"] and not s["ring"] and not s["pinky"]:
            if self.pressed:
                self.pressed = False
                self._last_release = t
                out.events.append(("release",))
            y = (lm[INDEX_TIP][1] + lm[MIDDLE_TIP][1]) / 2
            if self._scroll_y is not None:
                self._scroll_acc += (self._scroll_y - y) * self.scroll_gain  # hand up = scroll up
                steps = int(round(self._scroll_acc, 6))  # round off float noise, then truncate
                if steps:
                    self._scroll_acc -= steps
                    out.events.append(("scroll", steps))
            self._scroll_y = y
            self._pinch_frames = 0
            out.scrolling = True
            out.pressed = self.pressed
            return out
        self._scroll_y, self._scroll_acc = None, 0.0

        # ---- pinch state with hysteresis + debounce
        threshold = self.click_ratio * (self.release_factor if self.pressed else 1.0)
        pinched = out.pinch < threshold and not ok_pose and index_reach(iso) > self.min_reach
        if pinched and not self.pressed:
            self._pinch_frames += 1
            if self._pinch_frames >= self.confirm_frames and t - self._last_release >= self.debounce:
                self.pressed = True
                self._press_t = t
                out.events.append(("press",))
        elif not pinched:
            self._pinch_frames = 0
            if self.pressed:
                self.pressed = False
                self._last_release = t
                out.events.append(("release",))

        # ---- cursor: follow the index tip while it's up (or pinching)
        if s["index"] or self.pressed or pinched:
            target = map_to_screen(lm[INDEX_TIP][0], lm[INDEX_TIP][1], self.sw, self.sh, self.box)
            frozen = self.pressed and t - self._press_t < self.click_freeze
            if not frozen:  # hold still briefly after a press so clicks land where aimed
                out.cursor = self.ema.update(target)
                out.events.insert(0, ("move", *out.cursor))
            else:
                out.cursor = self.ema.value
        out.pressed = self.pressed
        return out


# ------------------------------------------------------------------ two-hand exit
class HoldTimer:
    """Progress 0..1 while a condition holds continuously; `done` once held `hold` s."""

    def __init__(self, hold: float = 1.0):
        self.hold = hold
        self.start: float | None = None

    def update(self, t: float, active: bool) -> float:
        if not active:
            self.start = None
            return 0.0
        if self.start is None:
            self.start = t
        return clamp((t - self.start) / self.hold, 0.0, 1.0)

    def done(self, t: float) -> bool:
        return self.start is not None and t - self.start >= self.hold


# ------------------------------------------------------------------ OS backends
def screen_size() -> tuple[int, int]:
    """Main screen size in points (the units pynput uses): Quartz on macOS, else pyautogui."""
    if sys.platform == "darwin":
        try:
            from Quartz import CGDisplayBounds, CGMainDisplayID
            b = CGDisplayBounds(CGMainDisplayID())
            return int(b.size.width), int(b.size.height)
        except Exception:
            pass
    try:
        import pyautogui
        w, h = pyautogui.size()
        return int(w), int(h)
    except Exception:
        print("[motion] couldn't read the screen size; assuming 1920x1080.")
        return 1920, 1080


_warned_volume = False


def set_system_volume(level: int) -> bool:
    """Set output volume 0-100. macOS only (osascript); other systems are skipped."""
    global _warned_volume
    if sys.platform != "darwin":
        if not _warned_volume:
            print("[motion] pinch volume only sets the system volume on macOS; skipping.")
            _warned_volume = True
        return False
    level = int(clamp(level, 0, 100))
    try:  # fire-and-forget so the video loop never waits on osascript
        subprocess.Popen(["osascript", "-e", f"set volume output volume {level}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"[motion] osascript failed: {e}")
        return False


class VirtualMouse:
    """Applies MouseLogic events with pynput.mouse (loaded lazily)."""

    def __init__(self):
        self._mouse = None
        self._button = None
        self.failed = False
        self.pressed = False

    def release_all(self):
        if self.pressed and self._mouse is not None:
            self._mouse.release(self._button)
        self.pressed = False

    def _ensure(self) -> bool:
        if self._mouse is None and not self.failed:
            try:
                from pynput.mouse import Button, Controller
                self._mouse, self._button = Controller(), Button.left
            except Exception as e:
                print(f"[motion] pynput mouse unavailable ({e}); mouse mode is display-only.")
                self.failed = True
        return self._mouse is not None

    def apply(self, events):
        if not events or not self._ensure():
            return
        for ev in events:
            kind = ev[0]
            if kind == "move":
                self._mouse.position = (int(ev[1]), int(ev[2]))
            elif kind == "press":
                self._mouse.press(self._button)
                self.pressed = True
            elif kind == "release" and self.pressed:
                self._mouse.release(self._button)
                self.pressed = False
            elif kind == "scroll":
                self._mouse.scroll(0, ev[1])
