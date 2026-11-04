"""Unit tests with synthetic hand landmarks (no webcam or model needed)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gesture import ActionDispatcher, GestureSmoother, classify  # noqa: E402
from gesture import actions as actions_mod  # noqa: E402
from gesture.config import ConfigError, bindings_for, load_config  # noqa: E402

FINGER_X = {"index": -0.06, "middle": -0.02, "ring": 0.02, "pinky": 0.06}
FINGER_BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}


def make_hand(extended, thumb="out", rotate_deg=0.0, mirror=False, pinch_gap=None):
    """Build 21 landmarks: wrist at bottom, fingers pointing up (image y grows down).

    pinch_gap: if set, the index curls forward and the thumb tip sits `pinch_gap` away
    from the index tip (the 👌 circle). Palm size (wrist -> middle MCP) is 0.18.
    """
    lm = [None] * 21
    wx, wy = 0.5, 0.8
    lm[0] = (wx, wy, 0)
    for name, fx in FINGER_X.items():
        b = FINGER_BASE[name]
        mcp = (wx + fx, wy - 0.18)
        lm[b] = (*mcp, 0)
        if name in extended:
            lm[b + 1] = (mcp[0], mcp[1] - 0.08, 0)
            lm[b + 2] = (mcp[0], mcp[1] - 0.13, 0)
            lm[b + 3] = (mcp[0], mcp[1] - 0.17, 0)
        else:  # curled back toward the palm
            lm[b + 1] = (mcp[0], mcp[1] - 0.05, 0)
            lm[b + 2] = (mcp[0], mcp[1] + 0.01, 0)
            lm[b + 3] = (mcp[0], mcp[1] + 0.05, 0)
    # thumb on the index side
    lm[1] = (wx - 0.06, wy - 0.04, 0)
    lm[2] = (wx - 0.10, wy - 0.08, 0)
    if thumb == "out":      # sticking out sideways/up (open palm)
        lm[3], lm[4] = (wx - 0.15, wy - 0.12, 0), (wx - 0.19, wy - 0.16, 0)
    elif thumb == "up":
        lm[3], lm[4] = (wx - 0.11, wy - 0.18, 0), (wx - 0.11, wy - 0.27, 0)
    elif thumb == "down":
        lm[2] = (wx - 0.10, wy - 0.10, 0)
        lm[3], lm[4] = (wx - 0.12, wy + 0.0, 0), (wx - 0.12, wy + 0.08, 0)
    else:                   # tucked across the palm
        lm[3], lm[4] = (wx - 0.06, wy - 0.13, 0), (wx - 0.01, wy - 0.15, 0)

    if pinch_gap is not None:
        lm[6], lm[7], lm[8] = (wx - 0.06, wy - 0.25, 0), (wx - 0.075, wy - 0.27, 0), (wx - 0.08, wy - 0.24, 0)
        lm[3], lm[4] = (wx - 0.13, wy - 0.17, 0), (wx - 0.08 - pinch_gap, wy - 0.24, 0)

    out = []
    a = math.radians(rotate_deg)
    for x, y, z in lm:
        dx, dy = x - wx, y - wy
        if mirror:
            dx = -dx
        rx = dx * math.cos(a) - dy * math.sin(a)
        ry = dx * math.sin(a) + dy * math.cos(a)
        out.append((wx + rx, wy + ry, z))
    return out


ALL = set(FINGER_X)
CASES = [
    ("open_palm", ALL, "out"),
    ("fist", set(), "in"),
    ("thumbs_up", set(), "up"),
    ("thumbs_down", set(), "down"),
    ("peace", {"index", "middle"}, "in"),
    ("pointing", {"index"}, "in"),
]


@pytest.mark.parametrize("expected,ext,thumb", CASES)
@pytest.mark.parametrize("mirror", [False, True])
def test_basic_gestures(expected, ext, thumb, mirror):
    assert classify(make_hand(ext, thumb, mirror=mirror)) == expected


@pytest.mark.parametrize("expected,ext,thumb", [c for c in CASES if "thumbs" not in c[0]])
@pytest.mark.parametrize("deg", [-25, 25])
def test_rotation_tolerance(expected, ext, thumb, deg):
    assert classify(make_hand(ext, thumb, rotate_deg=deg)) == expected


def test_invalid_input():
    assert classify(None) == "unknown"
    assert classify([(0, 0, 0)] * 5) == "unknown"


def test_smoother_filters_flicker():
    s = GestureSmoother(window=5, min_ratio=0.6)
    seq = ["fist", "fist", "open_palm", "fist", "fist"]
    out = [s.update(g) for g in seq]
    assert out[-1] == "fist" and out[0] == "unknown"


def test_dispatcher_cooldown_and_repeat():
    d = ActionDispatcher(dry_run=True, cooldown=1.0, repeat_interval=0.3)
    assert d.update("open_palm", now=0.0) == "Play / Pause"
    assert d.update("open_palm", now=0.5) is None          # held: no repeat for toggles
    assert d.update("unknown", now=0.6) is None
    assert d.update("open_palm", now=0.8) is None          # within cooldown
    assert d.update("unknown", now=1.5) is None
    assert d.update("thumbs_up", now=2.0) == "Volume +"
    assert d.update("thumbs_up", now=2.4) == "Volume +"    # repeat while held
    assert d.update("thumbs_up", now=2.5) is None


# ---------------------------------------------------------------- new static gestures
THREE = {"middle", "ring", "pinky"}
NEW_CASES = [  # (expected, make_hand kwargs)
    ("rock", dict(extended={"index", "pinky"}, thumb="out")),
    ("call_me", dict(extended={"pinky"}, thumb="out")),
    ("ok", dict(extended=THREE, thumb="out", pinch_gap=0.005)),
]


@pytest.mark.parametrize("expected,kw", NEW_CASES)
@pytest.mark.parametrize("mirror", [False, True])
def test_new_gestures(expected, kw, mirror):
    assert classify(make_hand(**kw, mirror=mirror)) == expected


@pytest.mark.parametrize("expected,kw", NEW_CASES)
@pytest.mark.parametrize("deg", [-25, 25])
@pytest.mark.parametrize("mirror", [False, True])
def test_new_gestures_rotation(expected, kw, deg, mirror):
    assert classify(make_hand(**kw, rotate_deg=deg, mirror=mirror)) == expected


@pytest.mark.parametrize("mirror", [False, True])
def test_rock_call_me_pointing_dont_conflict(mirror):
    # pointing with the thumb out is still pointing (pinky is down), not rock
    assert classify(make_hand({"index"}, "out", mirror=mirror)) == "pointing"
    # 🤘 horns (thumb tucked) also count as rock; pinky alone with the thumb tucked is nothing
    assert classify(make_hand({"index", "pinky"}, "in", mirror=mirror)) == "rock"
    assert classify(make_hand({"pinky"}, "in", mirror=mirror)) == "unknown"
    # rock needs middle + ring curled
    assert classify(make_hand({"index", "middle", "pinky"}, "out", mirror=mirror)) != "rock"


@pytest.mark.parametrize("mirror", [False, True])
def test_ok_vs_open_palm(mirror):
    assert classify(make_hand(ALL, "out", mirror=mirror)) == "open_palm"
    # tips 0.08 apart (> 0.25 x 0.18 palm) -> not a pinch
    assert classify(make_hand(THREE, "out", mirror=mirror, pinch_gap=0.08)) != "ok"
    # pinching but the ring finger is curled -> not ok
    assert classify(make_hand({"middle", "pinky"}, "out", mirror=mirror, pinch_gap=0.005)) != "ok"


def test_ok_pinch_ratio_is_configurable():
    hand = make_hand(THREE, "out", pinch_gap=0.03)  # 0.03 < 0.045 (default) but > 0.018
    assert classify(hand) == "ok"
    assert classify(hand, ok_pinch_ratio=0.1) != "ok"


# ---------------------------------------------------------------- config + modes
class FakeKeyboard:
    def __init__(self):
        self.events = []

    def press(self, k):
        self.events.append(("press", k))

    def release(self, k):
        self.events.append(("release", k))


@pytest.fixture
def fake_pynput(monkeypatch):
    """Stand-in for pynput so 'live' mode can be tested without sending real keys."""
    import types
    kb_mod = types.ModuleType("pynput.keyboard")
    kb_mod.Controller = FakeKeyboard
    kb_mod.Key = types.SimpleNamespace(right="RIGHT", left="LEFT", f5="F5", cmd="CMD",
                                       media_play_pause="PLAY")
    pkg = types.ModuleType("pynput")
    pkg.keyboard = kb_mod
    monkeypatch.setitem(sys.modules, "pynput", pkg)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", kb_mod)
    return kb_mod


def test_config_has_all_modes_and_globals():
    cfg = load_config()
    assert list(cfg["modes"]) == ["media", "slides", "mouse"]
    for mode in cfg["modes"]:
        b = bindings_for(cfg, mode)
        assert {"rock", "ok", "call_me"} <= set(b)
    slides = bindings_for(cfg, "slides")
    assert slides["peace"].keys == ("right",) and slides["pointing"].keys == ("left",)
    assert slides["open_palm"].keys == ("f5",)
    assert set(bindings_for(cfg, "mouse")) == {"rock", "ok", "call_me"}  # placeholder


def test_config_null_disables_and_validation(tmp_path):
    good = tmp_path / "c.yaml"
    good.write_text("global:\n  ok: {command: screenshot, label: Shot}\n"
                    "modes:\n  a: {fist: {key: f5, label: F5}}\n  b: {ok: null}\n")
    cfg = load_config(str(good))
    assert set(bindings_for(cfg, "a")) == {"ok", "fist"}
    assert bindings_for(cfg, "b") == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("modes:\n  a: {fist: {key: f5, command: screenshot, label: X}}\n")
    with pytest.raises(ConfigError):
        load_config(str(bad))
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "missing.yaml"))


def test_mode_cycling_changes_bindings():
    d = ActionDispatcher(dry_run=True)
    assert d.mode == "media"
    assert d.update("peace", now=0.0) == "Next track"
    assert d.cycle_mode() == "slides"
    assert d.update("unknown", now=0.1) is None
    assert d.update("peace", now=2.0) == "Next slide"
    assert d.cycle_mode() == "mouse"
    assert d.update("unknown", now=2.1) is None
    assert d.update("peace", now=4.0) is None  # placeholder mode: no media/slide binding
    assert d.cycle_mode() == "media"


def test_slides_send_arrow_keys_when_live(fake_pynput):
    d = ActionDispatcher(dry_run=False, mode="slides")
    assert not d.dry_run
    assert d.update("peace", now=0.0) == "Next slide"
    assert d.update("unknown", now=0.1) is None
    assert d.update("open_palm", now=2.0) == "Start presentation"
    assert d._keyboard.events == [("press", "RIGHT"), ("release", "RIGHT"),
                                  ("press", "F5"), ("release", "F5")]


def test_rock_toggles_dry_run(fake_pynput):
    d = ActionDispatcher(dry_run=True)
    assert d.update("rock", now=0.0).endswith("LIVE") and not d.dry_run
    assert d.update("unknown", now=0.1) is None
    assert d.update("rock", now=2.0).endswith("DRY-RUN") and d.dry_run


class SyncThread:
    def __init__(self, target, daemon=None):
        self.target = target

    def start(self):
        self.target()


def test_dry_run_skips_side_effects(monkeypatch):
    calls = []
    monkeypatch.setattr(actions_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(actions_mod.webbrowser, "open", lambda url: calls.append(url))
    monkeypatch.setattr(actions_mod.threading, "Thread", SyncThread)
    d = ActionDispatcher(dry_run=True)
    assert d.update("ok", now=0.0) == "Screenshot"
    assert d.update("call_me", now=2.0) == "Open Spotify"
    assert calls == []


def test_screenshot_and_spotify_commands(monkeypatch, fake_pynput):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["open", "-a"]:
            raise actions_mod.subprocess.CalledProcessError(1, cmd)  # Spotify not installed

    monkeypatch.setattr(actions_mod.sys, "platform", "darwin")
    monkeypatch.setattr(actions_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(actions_mod.webbrowser, "open", lambda url: calls.append(url))
    monkeypatch.setattr(actions_mod.threading, "Thread", SyncThread)
    d = ActionDispatcher(dry_run=False)
    d.update("ok", now=0.0)
    d.update("call_me", now=2.0)
    shot, spotify, fallback = calls
    assert shot[:2] == ["screencapture", "-x"] and os.path.basename(shot[2]).startswith("gesture_")
    assert shot[2].endswith(".png") and shot[2].startswith(os.path.expanduser("~/Desktop"))
    assert spotify == ["open", "-a", "Spotify"]
    assert fallback == "https://music.youtube.com"
