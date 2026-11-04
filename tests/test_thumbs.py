"""Thumbs up / down robustness (live-testing regressions).

Diagnosed causes: (1) the direction rule needed the thumb within 53° of vertical with no
hysteresis, and real thumbs already lean ~20-30°; (2) a thumb leaning over the fist failed
the old "thumb extended" distance test, so thumbs up became *fist* (Mute); (3) loose-fist
fingers read as ambiguous -> "unknown"; (4) a few unknown frames dropped the smoother and
the dispatcher's cooldown then interrupted the volume repeat.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gesture import ActionDispatcher, GestureSmoother, motion  # noqa: E402
from gesture.classifier import (AMB, EXT, HoldThrough, PoseTracker, classify, explain,  # noqa: E402
                                finger_metrics, finger_tristate, thumbs_pose)
from test_gestures import make_hand  # noqa: E402

FPS = 30


def _rot(p, c, deg):
    a = math.radians(deg)
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) - dy * math.sin(a), c[1] + dx * math.sin(a) + dy * math.cos(a), p[2])


def thumb_tilt(lm, deg):
    """Tilt only the thumb (IP + tip) about its MCP joint."""
    lm = list(lm)
    lm[3], lm[4] = _rot(lm[3], lm[2], deg), _rot(lm[4], lm[2], deg)
    return lm


def hand_tilt(lm, deg):
    """Tilt the whole hand about the wrist."""
    return [_rot(p, lm[0], deg) for p in lm]


def thumbs(direction="up", mirror=False):
    return make_hand(set(), direction, mirror=mirror)


def loose_fist(lm):
    """A loose fist: the PIP joints only half-bent (~130°, ambiguous), the fingertips curling
    toward the camera (z), so in the image they stay short (reach < 1)."""
    lm = list(lm)
    flip = 1 if lm[5][0] < lm[17][0] else -1  # keep it valid for mirrored hands too
    for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
        m = lm[mcp]
        lm[pip] = (m[0], m[1] - 0.05, 0.0)
        lm[dip] = (m[0], m[1] - 0.075, -0.03)
        lm[tip] = (m[0], m[1] - 0.035, -0.045)
    assert flip in (1, -1)
    return lm


def real_thumbs():
    """The real thumbs hands from tests/fixtures/real_hands.json (isotropic coords)."""
    from gesture.classifier import to_isotropic
    from gesture.debug import load_fixtures
    return [(s["label"], to_isotropic(s["landmarks"], s["aspect"], 1.0))
            for s in load_fixtures() if s["label"] in ("thumbs_up", "thumbs_down")]


def aim_thumb(lm, deg_from_vertical, whole_hand):
    """Rotate a real thumbs hand so its thumb points exactly `deg` from vertical
    (whole hand about the wrist, or just the thumb about its MCP)."""
    m = finger_metrics(lm)
    up = m["thumb_from_up"] <= 90
    cur = m["thumb_from_up"] if up else 180 - m["thumb_from_up"]
    dx = lm[4][0] - lm[2][0]
    side = 1 if dx >= 0 else -1  # which way it currently leans
    target_side = 1 if deg_from_vertical >= 0 else -1
    delta = abs(deg_from_vertical) * target_side - cur * side
    if not up:
        delta = -delta
    return hand_tilt(lm, delta) if whole_hand else thumb_tilt(lm, delta)


# ---------------------------------------------------------------- 1. direction + hysteresis
@pytest.mark.parametrize("direction,expected", [("up", "thumbs_up"), ("down", "thumbs_down")])
@pytest.mark.parametrize("deg", [-40, -25, 25, 40])
@pytest.mark.parametrize("mirror", [False, True])
def test_tilted_synthetic_hands_stay_thumbs(direction, expected, deg, mirror):
    tilted = hand_tilt(thumbs(direction, mirror), deg)
    assert explain(tilted).label == expected, (deg, explain(tilted).reason)


@pytest.mark.parametrize("deg", [-40, -30, -15, 0, 15, 30, 40])
@pytest.mark.parametrize("whole_hand", [True, False], ids=["hand-tilt", "thumb-tilt"])
def test_tilted_real_thumbs_stay_thumbs(deg, whole_hand):
    hands = real_thumbs()
    assert len(hands) >= 3
    for label, lm in hands:
        aimed = aim_thumb(lm, deg, whole_hand)
        m = finger_metrics(aimed)
        ang = m["thumb_from_up"] if label == "thumbs_up" else 180 - m["thumb_from_up"]
        assert ang == pytest.approx(abs(deg), abs=1.0)  # the thumb really is at ±deg
        info = explain(aimed)
        assert info.label == label, (label, deg, whole_hand, info.reason)


def test_direction_hysteresis_enter_50_stay_70():
    base = thumbs("up")
    trk = PoseTracker()
    labels = []
    for deg in list(range(0, 66, 5)) + [66] * 5:  # tilt the hand steadily to 66° from vertical
        info = trk.update(hand_tilt(base, deg))
        labels.append(info.label)
    assert all(lab == "thumbs_up" for lab in labels)  # entered upright, stays thumbs up to 66°
    assert classify(hand_tilt(base, 62)) == "unknown"  # without history, 62° is too far to *enter*
    assert classify(hand_tilt(base, 45)) == "thumbs_up"
    # beyond 70° it leaves; coming back it must get within 50° again
    assert trk.update(hand_tilt(base, 78)).label == "unknown"
    assert trk.update(hand_tilt(base, 62)).label == "unknown"
    assert trk.update(hand_tilt(base, 45)).label == "thumbs_up"


def test_thumb_direction_uses_thumb_mcp_to_tip():
    m = finger_metrics(thumbs("up"))
    assert m["thumb_from_up"] < 10
    assert finger_metrics(thumbs("down"))["thumb_from_up"] > 170
    assert 85 < finger_metrics(hand_tilt(thumbs("up"), 90))["thumb_from_up"] < 95


# ---------------------------------------------------------------- 2./3. loose fist, bent thumb
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("direction,expected", [("up", "thumbs_up"), ("down", "thumbs_down")])
def test_loose_fist_with_thumb(direction, expected, mirror):
    hand = loose_fist(thumbs(direction, mirror))
    m = finger_metrics(hand)
    tri = finger_tristate(m)
    fingers = ("index", "middle", "ring", "pinky")
    assert all(tri[f] == AMB for f in fingers)  # every finger half-bent: really a loose fist
    assert all(m[f + "_reach"] < 1.0 for f in fingers) and not any(tri[f] == EXT for f in fingers)
    assert explain(hand).label == expected, explain(hand).reason


def _bend_thumb(lm, deg):
    lm = list(lm)
    lm[4] = _rot(lm[4], lm[3], deg)  # bend the thumb tip at the IP joint
    return lm


def test_slightly_bent_thumb_still_counts():
    hand = _bend_thumb(thumbs("up"), -35)
    assert finger_metrics(hand)["thumb_angle"] < 155
    assert classify(hand) == "thumbs_up"
    bent_angles = []
    for label, lm in real_thumbs():  # real thumbs, bent 35° either way (down to ~124° at the IP)
        for deg in (-35, 35):
            hand = _bend_thumb(lm, deg)
            bent_angles.append(finger_metrics(hand)["thumb_angle"])
            assert classify(hand) == label, (label, deg)
    assert min(bent_angles) < 130  # well below the old 145° "extended" threshold


def test_finger_clearly_extended_is_not_thumbs():
    assert classify(make_hand({"index"}, "up")) != "thumbs_up"  # thumb + index out = not a thumbs up
    assert classify(make_hand({"pinky"}, "out")) == "call_me"


# ---------------------------------------------------------------- 4. hold-through
def test_hold_through_bridges_up_to_0_3_s():
    h = HoldThrough(grace=0.3)
    assert h.update(0.0, "thumbs_up") == "thumbs_up"
    assert h.update(0.10, "unknown") == "thumbs_up"
    assert h.update(0.29, "unknown") == "thumbs_up"
    assert h.update(0.35, "unknown") == "unknown"  # gap longer than 0.3 s ends it
    assert h.update(0.40, "thumbs_up") == "thumbs_up"


def test_other_confirmed_gesture_ends_hold_immediately():
    h = HoldThrough(grace=0.3)
    h.update(0.0, "thumbs_up")
    assert h.update(0.05, "fist") == "fist"
    assert h.update(0.10, "unknown") == "unknown"
    h.update(1.0, "thumbs_up")
    assert h.update(1.05, "thumbs_down") == "thumbs_down"  # clean flip, no stale "up"
    assert h.update(1.10, "unknown") == "thumbs_down"
    h.update(2.0, "peace")
    assert h.update(2.05, "unknown") == "unknown"  # only thumbs are held


class Pipeline:
    """demo.run() order for the primary hand: tracker -> smoother -> hold -> dispatcher."""

    def __init__(self):
        self.trk, self.sm, self.hold = PoseTracker(), GestureSmoother(7, 0.6), HoldThrough(grace=0.3)
        self.d = ActionDispatcher(dry_run=True, mode="media")
        self.t, self.log = 0.0, []

    def frame(self, lm):
        self.t += 1 / FPS
        raw = self.trk.update(lm).label if lm is not None else "unknown"
        stable = self.hold.update(self.t, self.sm.update(raw))
        fired = self.d.update(stable, self.t)
        self.log.append((raw, stable, fired))
        return stable, fired


def test_volume_repeat_survives_short_dropouts():
    p = Pipeline()
    up = thumbs("up")
    fires = []
    for i in range(120):  # 4 s of thumbs up with 4-frame (0.13 s) dropouts every 20 frames
        lm = None if i % 20 in (10, 11, 12, 13) else hand_tilt(up, 25 * math.sin(i / 15))
        _, fired = p.frame(lm)
        if fired:
            fires.append(p.t)
    stables = [s for _, s, _ in p.log[10:]]
    assert set(stables) == {"thumbs_up"}  # never drops to unknown after confirming
    gaps = [b - a for a, b in zip(fires, fires[1:])]
    assert len(fires) >= 9 and max(gaps) < 0.45  # repeats every ~0.35 s without interruption


def test_without_hold_the_repeat_breaks():
    """Documents the bug: same input, no hold-through -> stable drops and repeats stall."""
    p = Pipeline()
    p.hold.grace = 0.0
    up = thumbs("up")
    fires = []
    for i in range(120):
        lm = None if i % 20 in (10, 11, 12, 13) else up
        if p.frame(lm)[1]:
            fires.append(p.t)
    gaps = [b - a for a, b in zip(fires, fires[1:])]
    assert max(gaps) > 0.6


def test_up_down_switch_flips_cleanly():
    p = Pipeline()
    up, down = thumbs("up"), thumbs("down")
    for _ in range(20):
        p.frame(up)
    for _ in range(25):
        p.frame(down)
    raws = [r for r, _, _ in p.log]
    stables = [s for _, s, _ in p.log]
    assert set(raws) <= {"thumbs_up", "thumbs_down", "unknown"}  # never fist / call_me in between
    assert stables[-1] == "thumbs_down"
    first_down = stables.index("thumbs_down")
    assert all(s == "thumbs_up" for s in stables[7:20]) and all(s == "thumbs_down" for s in stables[first_down:])
    assert first_down - 20 <= 7  # one smoother window, then down; no lingering "up"
    labels = [f for _, _, f in p.log if f]
    assert "Volume +" in labels and "Volume -" in labels and "Mute" not in labels


# ---------------------------------------------------------------- 5. never pinch / fist / call_me
@pytest.mark.parametrize("mirror", [False, True])
def test_thumbs_never_pinch_fist_or_call_me(mirror):
    for direction in ("up", "down"):
        for make in (lambda b: b, loose_fist):
            base = make(thumbs(direction, mirror))
            for deg in range(-40, 41, 5):
                for hand in (hand_tilt(base, deg),):
                    label = classify(hand)
                    assert label not in ("fist", "call_me"), (direction, deg, label)
                    assert not motion.is_pinch_pose(hand), (direction, deg)
                    assert thumbs_pose(hand) == f"thumbs_{direction}"


def test_real_thumbs_never_pinch_fist_or_call_me():
    for label, lm in real_thumbs():
        for deg in range(-40, 41, 10):
            for whole in (True, False):
                hand = aim_thumb(lm, deg, whole)
                assert classify(hand) not in ("fist", "call_me")
                assert not motion.is_pinch_pose(hand)


def test_real_fist_and_pinch_still_work():
    assert classify(make_hand(set(), "in")) == "fist"
    assert motion.is_pinch_pose(make_hand({"index"}, "out"))
