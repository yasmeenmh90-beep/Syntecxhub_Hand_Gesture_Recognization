"""Unit tests for gesture/motion.py: swipes, pinch volume, virtual-mouse logic, exit hold.

Everything here is pure logic fed with synthetic data: no camera, mouse or speakers.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gesture import ActionDispatcher, motion  # noqa: E402
from test_gestures import make_hand  # noqa: E402

FPS = 30
DT = 1 / FPS


def trajectory(x0, y0, x1, y1, seconds, pose="open_palm", t0=0.0):
    """Palm-centre samples moving linearly from (x0, y0) to (x1, y1)."""
    n = max(2, int(round(seconds * FPS)))
    return [(t0 + i * DT, (x0 + (x1 - x0) * i / (n - 1), y0 + (y1 - y0) * i / (n - 1)), pose)
            for i in range(n)]


def run_swipe(det, samples):
    return [e for t, c, p in samples if (e := det.update(t, c, p))]


def shift(lm, dx=0.0, dy=0.0):
    return [(x + dx, y + dy, z) for x, y, z in lm]


# ---------------------------------------------------------------- swipe
@pytest.mark.parametrize("x0,x1,expected", [(0.3, 0.7, "swipe_right"), (0.7, 0.3, "swipe_left")])
def test_fast_horizontal_swipe(x0, x1, expected):
    assert run_swipe(motion.SwipeDetector(), trajectory(x0, 0.5, x1, 0.5, 0.3)) == [expected]


def test_slow_drift_is_not_a_swipe():
    # 0.4 of the frame over 1.5 s -> only ~0.1 within any 0.4 s window
    assert run_swipe(motion.SwipeDetector(), trajectory(0.3, 0.5, 0.7, 0.5, 1.5)) == []


def test_short_travel_is_not_a_swipe():
    assert run_swipe(motion.SwipeDetector(), trajectory(0.4, 0.5, 0.6, 0.5, 0.25)) == []


def test_too_much_vertical_travel_is_not_a_swipe():
    # 0.3 across but 0.2 down: 0.2 > 0.5 x 0.3
    assert run_swipe(motion.SwipeDetector(), trajectory(0.3, 0.3, 0.6, 0.5, 0.3)) == []
    # a little vertical wobble is fine
    assert run_swipe(motion.SwipeDetector(), trajectory(0.3, 0.50, 0.6, 0.55, 0.3)) == ["swipe_right"]


def test_swipe_needs_open_palm():
    assert run_swipe(motion.SwipeDetector(), trajectory(0.3, 0.5, 0.7, 0.5, 0.3, pose="fist")) == []
    # a couple of misclassified frames mid-swipe (motion blur) are tolerated
    samples = trajectory(0.3, 0.5, 0.7, 0.5, 0.3)
    samples[3] = (samples[3][0], samples[3][1], "unknown")
    samples[5] = (samples[5][0], samples[5][1], "unknown")
    assert run_swipe(motion.SwipeDetector(), samples) == ["swipe_right"]


def test_swipe_cooldown():
    det = motion.SwipeDetector(cooldown=1.0)
    first = trajectory(0.2, 0.5, 0.6, 0.5, 0.3)
    back = trajectory(0.6, 0.5, 0.2, 0.5, 0.3, t0=first[-1][0] + DT)        # within 1 s
    later = trajectory(0.2, 0.5, 0.6, 0.5, 0.3, t0=back[-1][0] + 1.0)        # after cooldown
    assert run_swipe(det, first) == ["swipe_right"]
    assert run_swipe(det, back) == []
    assert run_swipe(det, later) == ["swipe_right"]


def test_gate_blocks_static_open_palm_while_moving_and_after_swipe():
    det = motion.SwipeDetector(still_hold=0.3)
    # held still: open_palm passes once the palm has been still for still_hold
    still = trajectory(0.5, 0.5, 0.5, 0.5, 0.5)
    gated = []
    for t, c, p in still:
        det.update(t, c, p)
        gated.append(det.gate("open_palm", t))
    assert gated[0] == "unknown" and gated[-1] == "open_palm"

    # moving: blocked
    det = motion.SwipeDetector()
    for t, c, p in trajectory(0.3, 0.5, 0.45, 0.5, 0.3):
        det.update(t, c, p)
    assert det.gate("open_palm", t) == "unknown"

    # after a swipe, open_palm stays blocked even once the hand stops...
    det = motion.SwipeDetector()
    samples = trajectory(0.2, 0.5, 0.6, 0.5, 0.3)
    assert run_swipe(det, samples) == ["swipe_right"]
    t = samples[-1][0]
    for i in range(20):
        t += DT
        det.update(t, (0.6, 0.5), "open_palm")
    assert det.gate("open_palm", t) == "unknown"
    # ...until the pose changes; other gestures always pass straight through
    assert det.gate("fist", t) == "fist"
    assert det.gate("open_palm", t) == "open_palm"


def test_palm_center_is_mean_of_wrist_and_mcps():
    lm = make_hand({"index", "middle", "ring", "pinky"})
    cx, cy = motion.palm_center(lm)
    assert cx == pytest.approx(0.5) and cy == pytest.approx(0.8 - 0.18 * 4 / 5)


def test_dispatcher_trigger_swipes_per_mode():
    d = ActionDispatcher(dry_run=True)
    assert d.trigger("swipe_right", now=0.0) == "Next track"
    assert d.trigger("swipe_left", now=0.1) == "Previous track"  # trigger ignores update() cooldown
    d.cycle_mode()
    assert d.trigger("swipe_right", now=0.2) == "Next slide"
    d.cycle_mode()
    assert d.trigger("swipe_right", now=0.3) is None  # mouse mode: not bound


# ---------------------------------------------------------------- pinch volume
@pytest.mark.parametrize("ratio,expected", [
    (0.0, 0), (0.25, 0), (0.675, 50), (1.1, 100), (3.0, 100), (0.4625, 25),
])
def test_pinch_to_volume_mapping(ratio, expected):
    assert motion.pinch_to_volume(ratio, 0.25, 1.1) == expected


def test_pinch_to_volume_is_monotonic_and_rejects_bad_range():
    vals = [motion.pinch_to_volume(r / 100) for r in range(0, 150)]
    assert vals == sorted(vals)
    with pytest.raises(ValueError):
        motion.pinch_to_volume(0.5, 1.0, 1.0)


def test_volume_slider_step_and_smoothing():
    s = motion.VolumeSlider(min_ratio=0.25, max_ratio=1.1, step=3, alpha=1.0)  # no smoothing
    assert s.update(0.675) == 50          # first reading is always sent
    assert s.update(0.675 + 0.0085) is None   # +1 -> below the step of 3
    assert s.update(0.675 + 0.017) is None    # +2
    assert s.update(0.675 + 0.0255) == 53     # +3 -> sent
    assert s.update(0.0) == 0
    smooth = motion.VolumeSlider(alpha=0.5)
    smooth.update(0.25)                   # 0
    smooth.update(1.1)                    # target 100 -> EMA 50
    assert smooth.level == 50


@pytest.mark.parametrize("mirror", [False, True])
def test_pinch_pose(mirror):
    assert motion.is_pinch_pose(make_hand({"index"}, "out", mirror=mirror))       # L shape
    assert not motion.is_pinch_pose(make_hand({"index"}, "in", mirror=mirror))    # pointing
    assert not motion.is_pinch_pose(make_hand(set(), "in", mirror=mirror))        # fist
    assert not motion.is_pinch_pose(make_hand({"index", "middle"}, "out", mirror=mirror))
    # 👌 (other fingers up) is not the volume pinch
    assert not motion.is_pinch_pose(
        make_hand({"middle", "ring", "pinky"}, "out", mirror=mirror, pinch_gap=0.005))


def test_pinch_ratio_is_scale_invariant():
    lm = make_hand({"index"}, "out")
    big = [(0.5 + (x - 0.5) * 1.5, 0.8 + (y - 0.8) * 1.5, z) for x, y, z in lm]
    assert motion.pinch_ratio(big) == pytest.approx(motion.pinch_ratio(lm))


# ---------------------------------------------------------------- mouse mapping
@pytest.mark.parametrize("x,y,expected", [
    (0.2, 0.2, (0, 0)), (0.8, 0.8, (1919, 1079)), (0.5, 0.5, (959.5, 539.5)),
    (0.0, 1.0, (0, 1079)), (0.35, 0.65, (479.75, 809.25)),
])
def test_map_to_screen_centre_box(x, y, expected):
    sx, sy = motion.map_to_screen(x, y, 1920, 1080, box=0.6)
    assert (sx, sy) == pytest.approx(expected)


def test_ema_smoothing():
    e = motion.EMA2D(alpha=0.5)
    assert e.update((0, 0)) == (0, 0)
    assert e.update((100, 200)) == (50, 100)
    assert e.update((100, 200)) == (75, 150)
    for _ in range(30):
        p = e.update((100, 200))
    assert p == pytest.approx((100, 200), abs=1e-3)
    assert motion.EMA2D(alpha=1.0).update((3, 4)) == (3, 4)
    with pytest.raises(ValueError):
        motion.EMA2D(alpha=0)


def _events(frames, kind):
    return [e for f in frames for e in f.events if e[0] == kind]


def test_mouse_moves_with_index_tip():
    m = motion.MouseLogic(1000, 1000, alpha=1.0, box=0.6)
    lm = make_hand({"index"}, "in")
    f1 = m.update(0.0, lm)
    f2 = m.update(DT, shift(lm, dx=0.06))
    assert f1.cursor and f2.cursor[0] - f1.cursor[0] == pytest.approx(0.06 / 0.6 * 999)
    assert f1.events[0][0] == "move"
    # fist: index down -> cursor doesn't move
    assert m.update(2 * DT, make_hand(set(), "in")).cursor is None


def test_mouse_click_debounce_and_drag():
    m = motion.MouseLogic(1000, 1000, alpha=1.0, debounce=0.3, confirm_frames=2, click_freeze=0.1)
    pinch = make_hand(set(), "out", pinch_gap=0.005)
    open_ = make_hand({"index"}, "out")
    t = 0.0
    frames = []

    def step(lm, n=1):
        nonlocal t
        for _ in range(n):
            frames.append(m.update(t, lm))
            t += DT

    step(open_, 3)
    step(pinch, 1)
    assert not _events(frames, "press")   # one frame isn't enough (confirm_frames=2)
    step(pinch, 2)
    assert len(_events(frames, "press")) == 1 and m.pressed
    step(open_, 1)
    assert len(_events(frames, "release")) == 1 and not m.pressed      # = click
    step(pinch, 3)                        # re-pinch within 0.3 s debounce -> ignored
    assert len(_events(frames, "press")) == 1
    step(open_, 10)
    # drag: hold the pinch and move
    step(pinch, 2)
    assert len(_events(frames, "press")) == 2
    n_moves = len(_events(frames, "move"))
    for i in range(10):
        step(shift(pinch, dx=0.01 * i))
    assert m.pressed and len(_events(frames, "move")) > n_moves      # moving while held
    step(open_, 1)
    assert len(_events(frames, "release")) == 2


def test_mouse_fist_is_not_a_click():
    # thumb tucked onto the curled index: tips are close, but it's a fist
    m = motion.MouseLogic(1000, 1000)
    frames = [m.update(i * DT, make_hand(set(), "in")) for i in range(10)]
    assert not _events(frames, "press") and not _events(frames, "move")


def test_mouse_ok_sign_is_not_a_click():
    m = motion.MouseLogic(1000, 1000)
    ok = make_hand({"middle", "ring", "pinky"}, "out", pinch_gap=0.005)
    frames = [m.update(i * DT, ok) for i in range(10)]
    assert not _events(frames, "press")


def test_mouse_scroll_with_peace():
    m = motion.MouseLogic(1000, 1000, scroll_gain=100)
    peace = make_hand({"index", "middle"}, "in")
    frames = [m.update(i * DT, shift(peace, dy=-0.01 * i)) for i in range(11)]  # hand moves up
    scroll = sum(e[1] for e in _events(frames, "scroll"))
    assert scroll == 10 and all(f.scrolling for f in frames)
    assert not _events(frames, "move")
    frames = [m.update(1 + i * DT, shift(peace, dy=-0.1 + 0.01 * i)) for i in range(11)]  # back down
    assert sum(e[1] for e in _events(frames, "scroll")) == -10


def test_mouse_reset_releases_held_button():
    m = motion.MouseLogic(1000, 1000, confirm_frames=1)
    m.update(0.0, make_hand(set(), "out", pinch_gap=0.005))
    assert m.pressed
    assert m.reset() == [("release",)] and not m.pressed
    assert m.reset() == []


# ---------------------------------------------------------------- two-hand exit
def test_hold_timer():
    h = motion.HoldTimer(1.0)
    assert h.update(0.0, True) == 0.0
    assert h.update(0.5, True) == pytest.approx(0.5)
    assert not h.done(0.5)
    assert h.update(0.6, False) == 0.0     # interrupted -> restart
    h.update(1.0, True)
    assert h.update(2.0, True) == 1.0 and h.done(2.0)
