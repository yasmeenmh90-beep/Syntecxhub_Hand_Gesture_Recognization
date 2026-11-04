"""Guided data collection logic (gesture/guided.py): order, countdown, rate limit,
near-duplicate skipping, pause, skip, redo, quit, takes and the summary table."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gesture.guided import COUNTDOWN, DONE, GUIDED_ORDER, HINTS, RECORDING, GuidedSession  # noqa: E402

FPS = 30


class Sink:
    def __init__(self):
        self.rows = []
        self.calls = 0

    def __call__(self, rows):
        self.rows.extend(rows)
        self.calls += 1


def rows_fn(label_of, n_rows=1):
    def make(take):
        return [{"label": label_of(), "take": take} for _ in range(n_rows)]
    return make


class Clock:
    """Feeds frames at 30 fps; every frame gets a slightly different feature vector."""

    def __init__(self, session, rng_seed=0):
        self.s, self.t, self.rng = session, 0.0, np.random.default_rng(rng_seed)
        self.i = 0

    def until_recording(self):
        """Step (without a hand) until the countdown ends; nothing gets recorded."""
        while self.s.phase == COUNTDOWN:
            self.frames(1 / FPS, hand=False)

    def frames(self, seconds, hand=True, same=False):
        kept = 0
        for _ in range(int(round(seconds * FPS))):
            self.t += 1 / FPS
            self.i += 1
            self.s.update(self.t)
            if hand and not self.s.done:
                feats = np.zeros(63) if same else self.rng.normal(0, 0.3, 63)
                kept += self.s.offer(self.t, feats, rows_fn(lambda: self.s.gesture))
        return kept


def session(**kw):
    sink = Sink()
    kw.setdefault("per_gesture", 20)
    return GuidedSession(sink, **kw), sink


def test_order_and_hints():
    assert GUIDED_ORDER == ["open_palm", "fist", "thumbs_up", "thumbs_down", "peace", "pointing",
                            "rock", "ok", "call_me", "unknown"]
    assert all(HINTS[g] for g in GUIDED_ORDER)
    s, _ = session()
    assert s.gesture == "open_palm" and s.position() == (1, 10)


def test_full_run_walks_every_gesture_in_order():
    s, sink = session(per_gesture=10)
    c = Clock(s)
    seen = []
    while not s.done:
        if not seen or seen[-1] != s.gesture:
            seen.append(s.gesture)
        c.frames(0.5)
        assert c.t < 120, "session never finished"
    assert seen == GUIDED_ORDER
    assert [r.samples for r in s.results] == [10] * 10
    assert [r.status for r in s.results] == ["done"] * 10
    assert len(sink.rows) == 100 and sink.calls == 10  # one write per finished gesture
    # rows are labelled with the gesture that was on screen, in order
    labels = [r["label"] for r in sink.rows]
    assert labels == [g for g in GUIDED_ORDER for _ in range(10)]


def test_countdown_blocks_recording_for_3_seconds():
    s, _ = session()
    c = Clock(s)
    assert c.frames(2.9) == 0 and s.phase == COUNTDOWN
    assert s.countdown_left(c.t) == 1
    c.frames(0.2)
    assert s.phase == RECORDING


def test_rate_limit_about_10_per_second():
    s, _ = session(per_gesture=1000)
    c = Clock(s)
    c.frames(3.05)  # countdown
    kept = c.frames(2.0)  # 60 frames at 30 fps
    assert 18 <= kept <= 21


def test_near_duplicates_are_skipped():
    s, _ = session(per_gesture=1000)
    c = Clock(s)
    c.until_recording()
    assert c.frames(2.0, same=True) == 1  # a frozen hand only yields one sample


def test_no_hand_no_samples():
    s, _ = session()
    c = Clock(s)
    c.until_recording()
    assert c.frames(2.0, hand=False) == 0 and s.count == 0


def test_progress_and_vary_tip_at_half():
    s, _ = session(per_gesture=20)
    c = Clock(s)
    c.until_recording()
    while s.count < 9:
        c.frames(1 / FPS)
    assert not s.show_vary_tip
    while s.count < 10:
        c.frames(1 / FPS)
    assert s.progress == pytest.approx(0.5) and s.show_vary_tip


def test_skip_keeps_partial_rows_and_moves_on():
    s, sink = session(per_gesture=50)
    c = Clock(s)
    c.until_recording()
    c.frames(1.0)
    partial = s.count
    assert partial > 0 and sink.rows == []  # buffered, not written yet
    s.skip(c.t)
    assert s.gesture == "fist" and s.phase == COUNTDOWN
    assert len(sink.rows) == partial and {r["label"] for r in sink.rows} == {"open_palm"}
    assert s.results[0].status == "skipped" and s.results[0].samples == partial


def test_skip_during_countdown_records_nothing():
    s, sink = session()
    c = Clock(s)
    c.frames(1.0)
    s.skip(c.t)
    assert s.gesture == "fist" and sink.rows == [] and s.results[0].samples == 0


def test_redo_discards_this_gestures_rows():
    s, sink = session(per_gesture=50)
    c = Clock(s)
    c.until_recording()
    c.frames(1.0)
    assert s.count > 0
    s.redo(c.t)
    assert s.gesture == "open_palm" and s.count == 0 and s.phase == COUNTDOWN
    assert sink.rows == []  # nothing from the first attempt is ever written
    c.until_recording()
    c.frames(0.5)
    first_take = s.take_id()
    assert "-a1-" in first_take  # second attempt gets its own take ids
    s.skip(c.t)
    assert sink.rows and all("-a1-" in r["take"] for r in sink.rows)


def test_redo_only_affects_the_current_gesture():
    s, sink = session(per_gesture=20)
    c = Clock(s)
    while s.gesture == "open_palm":
        c.frames(1 / FPS)
    done_rows = len(sink.rows)
    assert done_rows == 20
    c.until_recording()
    c.frames(0.5)
    assert 0 < s.count < 20 and s.gesture == "fist"
    s.redo(c.t)
    assert len(sink.rows) == done_rows and s.gesture == "fist"


def test_pause_freezes_countdown_and_recording():
    s, _ = session(per_gesture=100)
    c = Clock(s)
    c.frames(2.0)
    s.toggle_pause(c.t)
    left = s.countdown_left(c.t)
    c.frames(5.0)
    assert s.phase == COUNTDOWN and s.countdown_left(c.t) == left  # frozen
    s.toggle_pause(c.t)
    c.frames(1.1)
    assert s.phase == RECORDING
    c.frames(1.0)
    n = s.count
    s.toggle_pause(c.t)
    assert c.frames(2.0) == 0 and s.count == n  # nothing recorded while paused
    s.toggle_pause(c.t)
    assert c.frames(1.0) > 0


def test_quit_saves_current_partial_gesture():
    s, sink = session(per_gesture=100)
    c = Clock(s)
    c.until_recording()
    c.frames(1.0)
    n = s.count
    s.quit(c.t)
    assert s.done and len(sink.rows) == n and s.results[0].status == "partial"
    assert all(r.status == "pending" for r in s.results[1:])


def test_takes_split_each_gesture_for_train_split():
    s, sink = session(per_gesture=50, takes_per_gesture=5)
    c = Clock(s)
    while s.gesture == "open_palm":
        c.frames(0.5)
    takes = {r["take"] for r in sink.rows}
    assert len(takes) == 5  # enough takes per label for train.py's group split


def test_mirror_augment_rows_count_as_one_sample():
    s, sink = session(per_gesture=4)
    c = Clock(s)
    c.until_recording()
    while s.gesture == "open_palm":
        c.t += 0.2
        s.offer(c.t, np.random.default_rng(int(c.t * 10)).normal(0, 0.3, 63), rows_fn(lambda: "open_palm", 2))
    assert s.results[0].samples == 4 and s.results[0].rows == 8 and len(sink.rows) == 8


def test_summary_table():
    s, _ = session(per_gesture=3)
    c = Clock(s)
    while s.gesture in ("open_palm", "fist"):
        c.frames(0.5)
    s.skip(c.t)
    s.quit(c.t)
    table = s.summary_table()
    assert "open_palm" in table and "unknown" in table
    total = table.strip().splitlines()[-1].split()
    assert total[0] == "total" and int(total[1]) == 6
    assert "skipped" in table


def test_guided_rows_are_valid_csv_rows(tmp_path):
    """End to end with the real row builder (incl. mirror augment) and features."""
    from gesture.features import CSV_COLUMNS, extract_features, rows_for_sample
    from test_gestures import ALL, make_hand
    sink = Sink()
    s = GuidedSession(sink, ["open_palm", "fist"], per_gesture=3, session="t")
    t = 3.1
    s.update(0.0)
    poses = {"open_palm": (ALL, "out"), "fist": (set(), "in")}
    rot = 0
    while not s.done:
        s.update(t)
        ext, thumb = poses[s.gesture]
        lm = make_hand(ext, thumb, rotate_deg=rot)
        rot += 4
        label = s.gesture
        s.offer(t, extract_features(lm)[:63],
                lambda take, lm=lm, label=label: rows_for_sample(label, lm, "Right", f"s{rot}", take,
                                                                 mirror_augment=True))
        t += 0.15
        if s.phase == COUNTDOWN:
            t += 3.1
    assert len(sink.rows) == 12  # 2 gestures x 3 samples x 2 (mirrored)
    assert all(set(r) == set(CSV_COLUMNS) for r in sink.rows)
    assert [r["label"] for r in sink.rows] == ["open_palm"] * 6 + ["fist"] * 6
    assert s.phase == DONE
