"""Regression tests on real hands: tests/fixtures/real_hands.json.

Add samples live with demo.py: press 'p', then the number key of the pose you're making
(1 open_palm ... 9 call_me, 0 unknown). Skipped when the file doesn't exist.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gesture.classifier import classify, to_isotropic  # noqa: E402
from gesture.debug import DEFAULT_FIXTURES, append_fixture, load_fixtures  # noqa: E402

SAMPLES = load_fixtures(DEFAULT_FIXTURES)


@pytest.mark.skipif(not SAMPLES, reason="no tests/fixtures/real_hands.json yet (press 'p' in demo.py)")
@pytest.mark.parametrize("sample", SAMPLES,
                         ids=[f"{i}-{s['label']}" for i, s in enumerate(SAMPLES)] or None)
def test_real_hand(sample):
    lm = to_isotropic(sample["landmarks"], sample["aspect"], 1.0)
    assert classify(lm) == sample["label"], sample.get("source", "")


def test_fixture_roundtrip(tmp_path):
    from test_gestures import ALL, make_hand
    path = str(tmp_path / "fx.json")
    s = append_fixture(make_hand(ALL, "out"), 1.0, "open_palm", path=path, source="test")
    append_fixture(make_hand(set(), "in"), 1.0, "fist", path=path, source="test")
    loaded = load_fixtures(path)
    assert [x["label"] for x in loaded] == ["open_palm", "fist"]
    assert s["rule_output"] == "open_palm" and len(loaded[0]["landmarks"]) == 21
