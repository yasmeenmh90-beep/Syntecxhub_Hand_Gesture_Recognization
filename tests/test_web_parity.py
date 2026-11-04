"""The browser classifier (web/classifier.js) must agree with gesture/classifier.py.

Runs both on the same ~2000 synthetic hands (every finger combination x thumb pose x
mirror x rotation, pinches, plus random jitter) and on random smoother / dispatcher
sequences. Needs Node.js; skipped when `node` isn't installed.
"""
import itertools
import json
import os
import random
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from gesture import ActionDispatcher, GestureSmoother, classify  # noqa: E402
from gesture.classifier import PoseTracker, explain  # noqa: E402
from test_gestures import FINGER_X, make_hand  # noqa: E402

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js not installed")
JS_MODULE = "file://" + os.path.join(ROOT, "web", "classifier.js")

NODE_SCRIPT = """
import { readFileSync } from "node:fs";
const { classify, explain, PoseTracker, GestureSmoother, ActionDispatcher } = await import(process.argv[2]);
const data = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out = { gestures: data.hands.map((h) => classify(h)),
              tri: data.hands.map((h) => explain(h).tri) };
out.tracked = data.tracks.map((seq) => { const t = new PoseTracker(); return seq.map((h) => t.update(h).label); });
out.smoother = data.sequences.map((seq) => {
  const s = new GestureSmoother(7, 0.7);
  return seq.map((g) => [s.update(g), Math.round(s.confidence * 1e6) / 1e6]);
});
out.dispatch = data.dispatch.map(({ bindings, events }) => {
  const d = new ActionDispatcher(bindings, { cooldown: 1.0, repeatInterval: 0.35 });
  return events.map(([g, t]) => { const b = d.update(g, t); return b ? b.label : null; });
});
console.log(JSON.stringify(out));
"""


def synthetic_hands():
    hands = []
    fingers = list(FINGER_X)
    for r in range(len(fingers) + 1):
        for ext in itertools.combinations(fingers, r):
            for thumb, mirror, deg in itertools.product(("out", "up", "down", "in"), (False, True),
                                                        (-30, 0, 30)):
                hands.append(make_hand(set(ext), thumb, rotate_deg=deg, mirror=mirror))
    for gap, mirror in itertools.product((0.0, 0.005, 0.03, 0.05, 0.08), (False, True)):
        hands.append(make_hand({"middle", "ring", "pinky"}, "out", mirror=mirror, pinch_gap=gap))
    from gesture.classifier import to_isotropic
    from gesture.debug import load_fixtures
    for fx in load_fixtures():  # real hands from tests/fixtures/real_hands.json
        hands.append([list(p) for p in to_isotropic(fx["landmarks"], fx["aspect"], 1.0)])
    rng = random.Random(1234)
    base = list(hands)
    for _ in range(1500):  # jittered copies land near the decision boundaries
        h = rng.choice(base)
        s = rng.uniform(0.005, 0.04)
        hands.append([(x + rng.gauss(0, s), y + rng.gauss(0, s), z) for x, y, z in h])
    return hands


def run_node(payload, tmp_path):
    data = tmp_path / "fixtures.json"
    data.write_text(json.dumps(payload))
    script = tmp_path / "run.mjs"
    script.write_text(NODE_SCRIPT)
    res = subprocess.run([NODE, str(script), JS_MODULE, str(data)], capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_js_matches_python(tmp_path):
    rng = random.Random(7)
    labels = ["open_palm", "fist", "peace", "unknown", "thumbs_up"]
    hands = synthetic_hands()
    sequences = [[rng.choice(labels[: rng.randint(2, 5)]) for _ in range(40)] for _ in range(50)]
    bindings = {"open_palm": {"action": "play_pause", "label": "Play / Pause"},
                "thumbs_up": {"action": "volume_up", "label": "Volume +", "repeat": True},
                "peace": {"action": "next", "label": "Next track"}}
    dispatch = []
    for _ in range(30):
        t, events = 0.0, []
        for _ in range(60):
            t += rng.choice((0.05, 0.1, 0.2, 0.4, 0.9))
            events.append([rng.choice(["open_palm", "thumbs_up", "peace", "unknown"]), round(t, 3)])
        dispatch.append({"bindings": bindings, "events": events})

    # hysteresis: random walks between poses, with jitter, through the tracker
    poses = [h for h in hands[:400]]
    tracks = []
    for _ in range(40):
        a, b = rng.choice(poses), rng.choice(poses)
        seq = []
        for k in range(25):
            t = min(1.0, max(0.0, k / 24 + rng.gauss(0, 0.05)))
            seq.append([(pa[0] + (pb[0] - pa[0]) * t + rng.gauss(0, 0.004),
                         pa[1] + (pb[1] - pa[1]) * t + rng.gauss(0, 0.004), 0.0) for pa, pb in zip(a, b)])
        tracks.append(seq)

    js = run_node({"hands": hands, "sequences": sequences, "dispatch": dispatch, "tracks": tracks}, tmp_path)

    py = [classify(h) for h in hands]
    mismatches = [(i, p, j) for i, (p, j) in enumerate(zip(py, js["gestures"])) if p != j]
    assert not mismatches, f"{len(mismatches)} classify mismatches, e.g. {mismatches[:5]}"
    assert len(set(py)) >= 9  # the fixtures really exercise (almost) every gesture
    tri_mismatch = [i for i, (h, jt) in enumerate(zip(hands, js["tri"])) if explain(h).tri != jt]
    assert not tri_mismatch, f"{len(tri_mismatch)} finger-state mismatches, e.g. {tri_mismatch[:5]}"
    amb = sum(1 for h in hands if "amb" in explain(h).tri.values())
    assert amb > 100  # plenty of ambiguous fingers, so the enumeration path is covered too

    for seq, js_seq in zip(tracks, js["tracked"]):
        tracker = PoseTracker()
        assert [tracker.update(h).label for h in seq] == js_seq

    for seq, js_seq in zip(sequences, js["smoother"]):
        s = GestureSmoother(7, 0.7)
        for g, (js_label, js_conf) in zip(seq, js_seq):
            label = s.update(g)
            top = max(s.history.count(x) for x in set(s.history))
            assert label == js_label
            assert top / 7 == pytest.approx(js_conf, abs=1e-6)

    for case, js_out in zip(dispatch, js["dispatch"]):
        d = ActionDispatcher(dry_run=True, cooldown=1.0, repeat_interval=0.35)
        for (g, t), js_label in zip(case["events"], js_out):
            assert d.update(g, now=t) == js_label, (g, t)
