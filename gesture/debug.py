"""Debug helpers: a one-line summary of the classifier's view of a hand, and the
real-hand fixture file (tests/fixtures/real_hands.json) written by demo.py's 'p' key.

Fixture format:
{"version": 1, "samples": [{
    "label": "open_palm",            # what the person says the pose is
    "landmarks": [[x, y, z], ...],   # 21 MediaPipe-normalised points (x / width, y / height)
    "aspect": 1.777,                 # frame width / height (to rebuild isotropic coords)
    "handedness": "Right",
    "rule_output": "unknown",        # what classify() said at capture time
    "reason": "...", "source": "demo.py", "timestamp": "2026-11-04T12:00:00"}]}
"""
from __future__ import annotations

import json
import os
import time

from .classifier import ALL_FIVE, AMB, EXT, PoseInfo, explain, to_isotropic

DEFAULT_FIXTURES = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "real_hands.json")


def state_string(info: PoseInfo) -> str:
    """'T1 I1 M1 R0 P0' with '?' marking fingers that were ambiguous this frame."""
    return " ".join(f"{f[0].upper()}{int(info.states[f])}{'?' if info.tri[f] == AMB else ''}"
                    for f in ALL_FIVE)


def debug_lines(info: PoseInfo) -> list[str]:
    m = info.metrics
    if not m:
        return [info.reason or "no metrics"]
    fingers = "  ".join(f"{f[0].upper()} {m[f + '_angle']:.0f}° r{m[f + '_reach']:.2f}"
                        for f in ("index", "middle", "ring", "pinky"))
    return [
        f"{state_string(info)}   ->  {info.label}",
        f"thumb: {m['thumb_angle']:.0f}°  away {m['thumb_away']:.2f}  out {m['thumb_out']:.2f}   "
        f"pinch {m['pinch']:.2f}",
        fingers,
        info.reason or "rule matched",
    ]


def load_fixtures(path: str = DEFAULT_FIXTURES) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("samples", [])


def append_fixture(landmarks, aspect: float, expected: str, handedness: str = "Right",
                   path: str = DEFAULT_FIXTURES, source: str = "demo.py") -> dict:
    """Add one labelled real hand to the fixture file (created if missing)."""
    info = explain(to_isotropic(landmarks, aspect, 1.0))
    sample = {
        "label": expected,
        "landmarks": [[round(float(c), 6) for c in (p[0], p[1], p[2] if len(p) > 2 else 0.0)]
                      for p in landmarks],
        "aspect": round(float(aspect), 6),
        "handedness": handedness,
        "rule_output": info.label,
        "reason": info.reason,
        "source": source,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    samples = load_fixtures(path)
    samples.append(sample)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "samples": samples}, f, indent=1)
    os.replace(tmp, path)  # atomic: never leaves a half-written file
    return sample


__all__ = ["DEFAULT_FIXTURES", "EXT", "append_fixture", "debug_lines", "load_fixtures", "state_string"]
