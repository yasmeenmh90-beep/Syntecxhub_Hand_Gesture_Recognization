"""Pinch-volume mode: it owns the hand while active.

The L-shape (thumb + index) classifies as "pointing", so without gating it would show
"Pointing → Previous track" and could fire Previous track. These tests run synthetic
pinch gestures through the same per-frame order as demo.run():
    pinch = controller.update(...); if pinch.ended: smoother.reset()
    stable = smoother.update(raw); dispatch = "unknown" if pinch.gating else stable
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gesture import ActionDispatcher, GestureSmoother, motion  # noqa: E402
from gesture.classifier import PoseTracker  # noqa: E402
from test_gestures import make_hand  # noqa: E402

FPS = 30


def pinch_hand(k, mirror=False):
    """k = 0: wide L-shape (thumb out); k = 1: thumb tip touching the index tip."""
    lm = list(make_hand({"index"}, "out"))
    tip_i = lm[8]
    target4 = (tip_i[0] + 0.005, tip_i[1] + 0.012, 0)
    target3 = (lm[2][0] + (target4[0] - lm[2][0]) * 0.5 - 0.01, lm[2][1] + (target4[1] - lm[2][1]) * 0.5, 0)
    lm[3] = tuple(a + (b - a) * k for a, b in zip(lm[3], target3))
    lm[4] = tuple(a + (b - a) * k for a, b in zip(lm[4], target4))
    if mirror:
        lm = [(1 - x, y, z) for x, y, z in lm]
    return lm


class Pipeline:
    """The demo's per-frame order for the primary hand (media mode)."""

    def __init__(self):
        self.tracker = PoseTracker()
        self.smoother = GestureSmoother(7, 0.6)
        self.ctl = motion.PinchController(motion.VolumeSlider(min_ratio=0.25, max_ratio=1.1, step=3, alpha=0.4))
        self.dispatcher = ActionDispatcher(dry_run=True, mode="media")
        self.t = 0.0
        self.log = []  # (t, raw, stable, pinch_state, fired)

    def frame(self, lm):
        self.t += 1 / FPS
        info = self.tracker.update(lm)
        pinch = self.ctl.update(self.t, lm, info.states, enabled=True)
        if pinch.ended:
            self.smoother.reset()
        stable = self.smoother.update(info.label)
        fired = self.dispatcher.update("unknown" if pinch.gating else stable, self.t)
        self.log.append((self.t, info.label, stable, pinch, fired))
        return pinch, fired


def pinch_session(p, mirror=False):
    ks = [0.0] * 10 + [i / 20 for i in range(21)] + [1 - i / 20 for i in range(21)] + [0.0] * 5 + \
        [i / 20 for i in range(11)]
    return [p.frame(pinch_hand(k, mirror)) for k in ks]


@pytest.mark.parametrize("mirror", [False, True])
def test_pointing_never_fires_while_pinch_mode_is_active(mirror):
    p = Pipeline()
    frames = pinch_session(p, mirror)
    raws = [raw for _, raw, _, _, _ in p.log]
    assert raws.count("pointing") > len(raws) * 0.8  # the L-shape really does read as pointing...
    assert all(pinch.gating for pinch, _ in frames)  # ...but pinch mode owns every frame
    assert all(fired is None for _, fired in frames)  # so nothing fires: no Previous track, no other action


@pytest.mark.parametrize("mirror", [False, True])
def test_no_previous_track_when_pinch_ends(mirror):
    """The bug from live testing: the leftover "pointing" fired Previous track on exit."""
    p = Pipeline()
    pinch_session(p, mirror)
    pointing = make_hand({"index"}, "in", mirror=mirror)  # thumb tucked: pinch ends, a real ☝️
    after = [p.frame(pointing) for _ in range(30)]
    ended_at = next(i for i, (pinch, _) in enumerate(after) if pinch.ended)
    fired = [(i, f) for i, (_, f) in enumerate(after) if f]
    # pinch mode lingers for the 0.3 s grace, then pointing must be confirmed afresh (7 frames)
    assert fired, "a genuine pointing gesture after the pinch must still work"
    first_i, label = fired[0]
    assert label == "Previous track"
    assert first_i >= ended_at + 6  # only after a full new smoother window, never the exit frame
    assert all(not pinch.gating for pinch, _ in after[ended_at:])


def test_volume_follows_the_gap_0_to_100():
    ctl = motion.PinchController(motion.VolumeSlider(min_ratio=0.25, max_ratio=1.1, step=3, alpha=1.0))
    t, levels, sends = 0.0, [], []
    for k in [i / 20 for i in range(21)]:  # open L (k=0) -> touching (k=1)
        t += 1 / FPS
        st = ctl.update(t, pinch_hand(k))
        if st.active:
            levels.append(st.level)
            if st.send is not None:
                sends.append(st.send)
    assert levels[0] == 100 and levels[-1] == 0  # wide open = 100 %, touching = 0 %
    assert levels == sorted(levels, reverse=True)  # closes smoothly, never jumps back
    assert all(abs(b - a) >= 3 for a, b in zip(sends, sends[1:]))  # OS only told about >= 3 changes
    assert motion.pinch_ratio(pinch_hand(0.0)) >= 1.1 and motion.pinch_ratio(pinch_hand(1.0)) < 0.25


@pytest.mark.parametrize("mirror", [False, True])
def test_pinch_pose_holds_across_the_whole_gap(mirror):
    for k in [i / 20 for i in range(21)]:
        assert motion.is_pinch_pose(pinch_hand(k, mirror)), k
    # and still rejects the look-alikes
    assert not motion.is_pinch_pose(make_hand({"index"}, "in", mirror=mirror))  # pointing, thumb tucked
    assert not motion.is_pinch_pose(make_hand(set(), "in", mirror=mirror))  # fist
    assert not motion.is_pinch_pose(make_hand({"index", "middle"}, "out", mirror=mirror))  # peace


def test_single_spurious_pinch_frame_gates_but_does_not_activate():
    ctl = motion.PinchController(motion.VolumeSlider(), enter_frames=2)
    st = ctl.update(0.0, pinch_hand(0.5))
    assert st.gating and not st.active and st.level is None and st.send is None
    st = ctl.update(0.033, make_hand({"index"}, "in"))
    assert not st.gating and not st.active and not st.ended


def test_exit_grace_bridges_brief_misreads():
    ctl = motion.PinchController(motion.VolumeSlider(), exit_grace=0.3)
    t = 0.0
    for _ in range(5):
        t += 1 / FPS
        ctl.update(t, pinch_hand(0.3))
    assert ctl.active
    for _ in range(3):  # 0.1 s of misread frames
        t += 1 / FPS
        st = ctl.update(t, make_hand({"index"}, "in"))
        assert st.active and st.gating and not st.ended
    t += 1 / FPS
    assert ctl.update(t, pinch_hand(0.3)).active
    for _ in range(12):  # > 0.3 s: mode ends exactly once
        t += 1 / FPS
        st = ctl.update(t, make_hand({"index"}, "in"))
        if st.ended:
            break
    assert st.ended and not st.active


def test_disabled_outside_volume_modes():
    ctl = motion.PinchController(motion.VolumeSlider())
    for i in range(5):
        st = ctl.update(i / FPS, pinch_hand(0.4), enabled=False)
        assert not st.gating and not st.active
