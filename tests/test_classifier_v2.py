"""Regression tests for the bugs found in live testing (classifier v2).

1. open palm with a real (short) spread thumb -> was "unknown"
2. hand pointing sideways -> was "fist" (fired Mute); ambiguous -> must be unknown, not fist
3. peace flicker -> hysteresis keeps finger states stable
4. 🤘 horns (thumb tucked) -> rock
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gesture.classifier import (AMB, CURL, EXT, PoseTracker, classify, explain,  # noqa: E402
                                finger_metrics, finger_tristate, to_isotropic)
from test_gestures import ALL, make_hand  # noqa: E402

JOINTS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def _rot(p, c, deg):
    a = math.radians(deg)
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] + dx * math.cos(a) - dy * math.sin(a), c[1] + dx * math.sin(a) + dy * math.cos(a), p[2])


def bend(lm, finger, pip_angle):
    """Bend an extended finger at its PIP joint so the MCP-PIP-DIP angle is `pip_angle`."""
    lm = list(lm)
    _, pip, dip, tip = JOINTS[finger]
    for j in (dip, tip):
        lm[j] = _rot(lm[j], lm[pip], -(180 - pip_angle))  # negative = curl toward the palm here
    return lm


def swing(lm, finger, deg):
    """Rotate a whole (straight) finger about its MCP knuckle: pointing sideways."""
    lm = list(lm)
    mcp, pip, dip, tip = JOINTS[finger]
    for j in (pip, dip, tip):
        lm[j] = _rot(lm[j], lm[mcp], deg)
    return lm


# ---------------------------------------------------------------- 1. open palm thumb
def close_thumb(lm, deg):
    """Swing the straight thumb about its base (CMC) toward the index finger."""
    lm = list(lm)
    for j in (2, 3, 4):
        lm[j] = _rot(lm[j], lm[1], deg)
    return lm


def _short_spread_thumb(mirror):
    base = make_hand(ALL, "out", mirror=mirror)
    for deg in range(0, 60):
        for sign in (1, -1):
            hand = close_thumb(base, sign * deg)
            m = finger_metrics(hand)
            if 0.42 < m["thumb_out"] < 0.52 and m["thumb_away"] > 1.03:
                return hand, m
    raise AssertionError("couldn't build the thumb for the test")


@pytest.mark.parametrize("mirror", [False, True])
def test_open_palm_with_short_real_thumb(mirror):
    # the live bug: a straight, spread thumb whose tip is ~0.48 palm from the index knuckle
    # (measured on a real photo). The old rule needed > 0.55 and returned "unknown".
    hand, m = _short_spread_thumb(mirror)
    assert m["thumb_out"] < 0.55 and m["thumb_angle"] > 170
    assert classify(hand) == "open_palm"


def test_open_palm_survives_16_9_normalisation():
    # same hand as a 16:9 camera reports it (x squeezed by 9/16), corrected back
    hand = make_hand(ALL, "out")
    squeezed = [(x * 9 / 16, y, z) for x, y, z in hand]
    assert classify(to_isotropic(squeezed, 16, 9)) == "open_palm"


# ---------------------------------------------------------------- 2. sideways hands
def _sideways_pointing():
    base = make_hand({"index"}, "in")
    for deg in range(40, 160, 2):
        for sign in (1, -1):
            hand = swing(base, "index", sign * deg)
            m = finger_metrics(hand)
            if 0.95 <= m["index_reach"] < 1.15:
                return hand, m
    raise AssertionError("couldn't build a sideways finger for the test")


def test_sideways_pointing_is_pointing_not_fist():
    hand, m = _sideways_pointing()
    assert m["index_reach"] < 1.15  # the old wrist-distance rule called this finger curled -> fist
    assert m["index_angle"] > 170  # but it's perfectly straight
    assert classify(hand) == "pointing"


def test_half_bent_finger_is_unknown_not_fist():
    hand = bend(make_hand({"index"}, "in"), "index", 135)  # halfway: neither straight nor curled
    info = explain(hand)
    assert info.tri["index"] == AMB
    assert info.label == "unknown" and "pointing" in info.reason and "fist" in info.reason
    # with history, hysteresis may keep "pointing", but it must never jump to fist
    tracker = PoseTracker()
    tracker.update(make_hand({"index"}, "in"))  # clear pointing first
    assert tracker.update(hand).label == "pointing"
    tracker = PoseTracker()
    tracker.update(make_hand(set(), "in"))  # clear fist first
    assert tracker.update(hand).label == "unknown"  # fist needs all four clearly curled


@pytest.mark.parametrize("deg", [-90, -60, 60, 90])
def test_rotated_hands_keep_their_labels(deg):
    for expected, ext, thumb in [("pointing", {"index"}, "in"), ("peace", {"index", "middle"}, "in"),
                                 ("open_palm", ALL, "out"), ("fist", set(), "in")]:
        assert classify(make_hand(ext, thumb, rotate_deg=deg)) == expected, (expected, deg)


def test_finger_tristate_bands():
    straight = finger_tristate(finger_metrics(make_hand(ALL, "out")))
    curled = finger_tristate(finger_metrics(make_hand(set(), "in")))
    assert all(straight[f] == EXT for f in straight)
    assert all(curled[f] == CURL for f in curled)


# ---------------------------------------------------------------- 3. hysteresis (peace flicker)
def test_peace_is_stable_while_ring_wobbles():
    peace = make_hand({"index", "middle", "ring"}, "in")
    clear = make_hand({"index", "middle"}, "in")
    wobble = [bend(peace, "ring", a) for a in (126, 138, 129, 141, 124, 136) * 4]
    assert all(finger_tristate(finger_metrics(h))["ring"] == AMB for h in wobble)

    stateless = [classify(h) for h in wobble]
    assert "peace" not in stateless  # every frame alone is ambiguous...
    tracker = PoseTracker()
    tracker.update(clear)
    assert [tracker.update(h).label for h in wobble] == ["peace"] * len(wobble)  # ...but stable with history

    # clear evidence still switches the state (ring fully straight -> no longer peace)
    assert tracker.update(peace).label != "peace"
    assert tracker.update(clear).label == "peace"


def test_hysteresis_prev_resolves_only_ambiguous_fingers():
    hand = make_hand(ALL, "out")
    # prev claims everything was curled: clear evidence wins, prev is ignored
    assert classify(hand, prev={f: False for f in ("thumb", "index", "middle", "ring", "pinky")}) == "open_palm"


def test_smoother_default_min_ratio_is_0_6():
    from gesture import GestureSmoother
    s = GestureSmoother()
    assert s.min_ratio == 0.6
    out = [s.update(g) for g in ["peace"] * 4 + ["unknown"] * 3]  # 4/7 = 0.57 < 0.6
    assert out[-1] == "unknown"
    s.reset()
    out = [s.update(g) for g in ["peace"] * 5 + ["unknown"] * 2]  # 5/7 = 0.71
    assert out[-1] == "peace"


# ---------------------------------------------------------------- 4. horns
@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("thumb", ["in", "out"])
def test_rock_with_thumb_tucked_or_out(mirror, thumb):
    assert classify(make_hand({"index", "pinky"}, thumb, mirror=mirror)) == "rock"


# ---------------------------------------------------------------- debug reasons
def test_explain_reports_what_blocked_open_palm():
    hand = make_hand(ALL, "in")  # all fingers up, thumb tucked across the palm
    info = explain(hand)
    assert info.label == "unknown"
    assert "open_palm (thumb up)" in info.reason
