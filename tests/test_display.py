"""Pure display helpers (gesture/display.py): colours, hold-to-fire ring, tag smoothing."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gesture import GestureSmoother  # noqa: E402
from gesture import display as D  # noqa: E402
from gesture.classifier import GESTURES  # noqa: E402
from gesture.config import Binding  # noqa: E402


# ---------------------------------------------------------------- colours
def test_every_gesture_has_its_own_colour():
    named = [g for g in GESTURES if g != "unknown"]
    colours = [D.gesture_color(g) for g in named]
    assert len(set(colours)) == len(named)  # all distinct
    assert D.UNKNOWN_COLOR not in colours
    assert D.gesture_color("unknown") == D.gesture_color(None) == D.gesture_color("nope") == D.UNKNOWN_COLOR
    assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in colours)


def test_requested_colour_families():
    b, g, r = D.gesture_color("fist")
    assert r > 200 and g < 120 and b < 120  # red
    b, g, r = D.gesture_color("thumbs_up")
    assert g > 200 and g > r and g > b  # green
    b, g, r = D.gesture_color("peace")
    assert b > 200 and g > 200 and r < 80  # cyan
    b, g, r = D.gesture_color("ok")
    assert r > 200 and g > 200 and b < 100  # yellow
    assert D.gesture_color("open_palm") == (216, 79, 255)  # the theme pink


def test_colour_fade_takes_150_ms():
    f = D.ColorFade((0, 0, 0), duration=0.15)
    f.set((200, 100, 50), now=1.0)
    assert f.value(1.0) == (0, 0, 0)
    assert f.value(1.075) == (100, 50, 25)  # halfway
    assert f.value(1.15) == (200, 100, 50) and f.done(1.15)
    # retargeting mid-fade starts from the current colour, no jump
    f.set((0, 0, 0), now=1.2)
    f2 = D.ColorFade((0, 0, 0), duration=0.15)
    f2.set((200, 100, 50), 0.0)
    mid = f2.value(0.075)
    f2.set((0, 0, 0), 0.075)
    assert f2.value(0.075) == mid


# ---------------------------------------------------------------- hold-to-fire ring
def test_needed_votes():
    assert D.needed_votes(7, 0.6) == 5  # 0.6 * 7 = 4.2 -> 5 frames
    assert D.needed_votes(7, 0.7) == 5
    assert D.needed_votes(5, 0.6) == 3
    assert D.needed_votes(10, 0.5) == 5


def test_hold_progress_tracks_the_real_smoother():
    s = GestureSmoother(window=7, min_ratio=0.6)
    seen = []
    for _ in range(7):
        stable = s.update("peace")
        label, p, confirmed = D.hold_progress(s.history, 7, 0.6, stable)
        seen.append((label, round(p, 2), confirmed))
    assert all(lab == "peace" for lab, _, _ in seen)
    progress = [p for _, p, _ in seen]
    assert progress == sorted(progress)  # only fills up
    assert progress[0] == pytest.approx(0.2)  # 1 of the 5 votes needed
    assert all(p < 1 for p, (_, _, c) in zip(progress, seen) if not c)
    assert seen[-1] == ("peace", 1.0, True)  # 100% exactly when the smoother confirms
    first_confirm = next(i for i, (_, _, c) in enumerate(seen) if c)
    assert first_confirm == 6  # the window has to be full first


def test_hold_progress_unknown_and_mixed():
    assert D.hold_progress([], 7, 0.6, "unknown") == ("unknown", 0.0, False)
    assert D.hold_progress(["unknown"] * 5, 7, 0.6, "unknown") == ("unknown", 0.0, False)
    label, p, c = D.hold_progress(["fist", "peace", "peace", "unknown"], 7, 0.6, "unknown")
    assert label == "peace" and p == pytest.approx(2 / 5) and not c
    assert D.hold_progress(["fist"] * 7, 7, 0.6, "fist") == ("fist", 1.0, True)


# ---------------------------------------------------------------- tag smoothing / placement
def test_rect_smoother_ema():
    sm = D.RectSmoother(alpha=0.5)
    assert sm.update((0, 0, 100, 100)) == (0, 0, 100, 100)  # first value snaps
    assert sm.update((100, 100, 200, 200)) == (50, 50, 150, 150)  # halfway, no jump
    for _ in range(20):
        r = sm.update((100, 100, 200, 200))
    assert r == (100, 100, 200, 200)
    sm.reset()
    assert sm.update((7, 7, 9, 9)) == (7, 7, 9, 9)
    with pytest.raises(ValueError):
        D.RectSmoother(alpha=0)


def test_smoothing_removes_jitter():
    import random
    rng = random.Random(0)
    sm = D.RectSmoother(alpha=0.3)
    raw, out = [], []
    for _ in range(200):
        j = rng.gauss(0, 8)
        box = (500 + j, 300 + j, 700 + j, 500 + j)
        raw.append(box[0])
        out.append(sm.update(box)[0])
    spread = lambda xs: max(xs[50:]) - min(xs[50:])  # noqa: E731
    assert spread(out) < spread(raw) / 2


def test_clamp_rect_keeps_inside_frame():
    assert D.clamp_rect((-20, -5, 80, 45), 640, 360) == (0, 0, 100, 50)
    assert D.clamp_rect((600, 340, 700, 390), 640, 360) == (540, 310, 640, 360)
    assert D.clamp_rect((10, 10, 50, 50), 640, 360) == (10, 10, 50, 50)


def test_place_tag_prefers_above_and_avoids_panels():
    box = (400, 300, 600, 500)
    assert D.place_tag(box, 120, 30, 1280, 720) == (400, 264)  # above-left
    hud = (0, 0, 470, 290)  # covers the above-left slot only
    x, y = D.place_tag(box, 120, 30, 1280, 720, avoid=[hud])
    assert (x, y) == (480, 264)  # above-right
    tag = (x, y, x + 120, y + 30)
    assert not D.overlaps(tag, hud)


def test_place_tag_stays_in_frame_and_can_give_up():
    # hand at the top-left corner: the tag is clamped inside the frame
    x, y = D.place_tag((0, 0, 150, 150), 120, 30, 1280, 720)
    assert x >= 0 and y >= 0
    # everything covered: no tag rather than one over a panel
    assert D.place_tag((100, 100, 300, 300), 120, 30, 400, 400, avoid=[(0, 0, 400, 400)]) is None


# ---------------------------------------------------------------- curves + feedback
def test_pop_curve():
    assert D.pop_curve(0.0) == (0.6, 1.0)
    assert D.pop_curve(0.12)[0] == pytest.approx(1.1)
    assert D.pop_curve(0.3) == (1.0, 1.0)
    scale, alpha = D.pop_curve(0.79)
    assert scale == 1.0 and 0 < alpha < 0.1
    assert D.pop_curve(0.81) is None and D.pop_curve(-0.1) is None
    peak = max(D.pop_curve(i / 100)[0] for i in range(0, 80))
    assert peak == pytest.approx(1.1, abs=1e-3)


def test_float_and_flash_curves():
    rise, alpha = D.float_curve(0.0)
    assert rise == 0 and alpha == 1
    assert D.float_curve(1.0)[0] == pytest.approx(1.0) and D.float_curve(1.0)[1] == pytest.approx(0)
    assert D.float_curve(1.1) is None
    assert D.flash_curve(0.0) == (1.0, 1.0)
    assert D.flash_curve(0.175)[0] == pytest.approx(1.3)
    assert D.flash_curve(0.4) is None


def test_action_icons():
    assert D.action_icon(Binding("peace", "Next track", keys=("media_next",))) == "⏭"
    assert D.action_icon(Binding("thumbs_up", "Volume +", keys=("media_volume_up",), repeat=True)) == "🔊"
    assert D.action_icon(Binding("ok", "Screenshot", command="screenshot")) == "📸"
    assert D.action_icon(Binding("x", "Combo", keys=("cmd", "shift", "p"))) == "⌨️"
    assert D.action_icon(None) == "✨"


def test_volume_meter_ticks():
    m = D.VolumeMeter(level=50)
    m.tick(+1, now=0.0)
    m.tick(+1, now=0.35)
    assert m.level == pytest.approx(50 + 2 * 100 / 16)
    for _ in range(20):
        m.tick(+1, now=1.0)
    assert m.level == 100  # clamped
    m.tick(-1, now=2.0)
    assert m.level == pytest.approx(100 - 100 / 16)
    assert m.visible(2.5, holding=False) and not m.visible(3.0, holding=False)
    assert m.visible(10.0, holding=True)
