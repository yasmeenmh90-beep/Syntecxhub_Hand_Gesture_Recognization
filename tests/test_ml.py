"""Tests for feature normalisation and the train / evaluate / inference pipeline."""
import csv
import math
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from gesture import features as F  # noqa: E402
from test_gestures import ALL, make_hand  # noqa: E402

HAND = make_hand(ALL, "out")


def transform(lm, scale=1.0, dx=0.0, dy=0.0, dz=0.0, about=(0.5, 0.5)):
    return [(about[0] + (x - about[0]) * scale + dx, about[1] + (y - about[1]) * scale + dy,
             z * scale + dz) for x, y, z in lm]


# ---------------------------------------------------------------- features
def test_feature_shape_and_names():
    f = F.extract_features(HAND, "Right")
    assert f.shape == (64,) and len(F.FEATURE_NAMES) == 64
    assert f[-1] == 1.0 and F.extract_features(HAND, "Left")[-1] == -1.0


def test_wrist_at_origin_and_unit_palm():
    n = F.normalize_landmarks(HAND).reshape(21, 3)
    assert np.allclose(n[0], 0)
    assert math.hypot(n[9][0], n[9][1]) == pytest.approx(1.0)


@pytest.mark.parametrize("dx,dy", [(0.1, -0.2), (-0.3, 0.05), (0.0, 0.15)])
def test_translation_invariance(dx, dy):
    a = F.normalize_landmarks(HAND)
    b = F.normalize_landmarks(transform(HAND, dx=dx, dy=dy, dz=0.3))
    assert np.allclose(a, b)


@pytest.mark.parametrize("scale", [0.4, 0.75, 1.6, 2.5])
def test_scale_invariance(scale):
    # hand closer to / further from the camera, scaled about an arbitrary point
    a = F.normalize_landmarks(HAND)
    b = F.normalize_landmarks(transform(HAND, scale=scale, about=(0.2, 0.9)))
    assert np.allclose(a, b)


def test_features_distinguish_poses():
    a = F.normalize_landmarks(make_hand(ALL, "out"))
    b = F.normalize_landmarks(make_hand(set(), "in"))
    assert np.abs(a - b).max() > 0.3


def test_mirror_flips_x_and_handedness():
    a = F.normalize_landmarks(HAND).reshape(21, 3)
    m = F.normalize_landmarks(F.mirror_landmarks(HAND)).reshape(21, 3)
    assert np.allclose(m[:, 0], -a[:, 0]) and np.allclose(m[:, 1:], a[:, 1:])
    assert F.swap_handedness("Right") == "Left" and F.swap_handedness("Left") == "Right"


def test_bad_input_raises():
    with pytest.raises(ValueError):
        F.normalize_landmarks([(0, 0, 0)] * 5)


def test_rows_roundtrip_and_mirror_augment():
    rows = F.rows_for_sample("open_palm", HAND, "Right", "s-1", "t-1", mirror_augment=True)
    assert len(rows) == 2 and set(rows[0]) == set(F.CSV_COLUMNS)
    orig, flip = rows
    assert orig["sample_id"] == flip["sample_id"] == "s-1"       # kept together in splits
    assert (orig["mirrored"], flip["mirrored"]) == (0, 1)
    assert (orig["handedness"], flip["handedness"]) == ("Right", "Left")
    assert np.allclose(F.raw_landmarks_from_row(orig), HAND)
    assert flip["x8"] == pytest.approx(-orig["x8"])


# ---------------------------------------------------------------- pipeline on a tiny synthetic CSV
POSES = {
    "open_palm": dict(extended=ALL, thumb="out"),
    "fist": dict(extended=set(), thumb="in"),
    "peace": dict(extended={"index", "middle"}, thumb="in"),
    "rock": dict(extended={"index", "pinky"}, thumb="out"),
}


def write_synthetic_csv(path, takes=6, frames=5, seed=0):
    rng = random.Random(seed)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=F.CSV_COLUMNS)
        w.writeheader()
        n = 0
        for label, kw in POSES.items():
            for t in range(takes):
                mirror = t % 2 == 1
                base = make_hand(**kw, rotate_deg=rng.uniform(-20, 20), mirror=mirror)
                base = transform(base, scale=rng.uniform(0.7, 1.3), dx=rng.uniform(-0.1, 0.1))
                for _ in range(frames):
                    lm = [(x + rng.gauss(0, 0.004), y + rng.gauss(0, 0.004), z) for x, y, z in base]
                    hand = "Left" if mirror else "Right"
                    for row in F.rows_for_sample(label, lm, hand, f"s{n}", f"{label}-t{t}"):
                        w.writerow(row)
                    n += 1
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    import train
    d = tmp_path_factory.mktemp("ml")
    csv_path = write_synthetic_csv(str(d / "landmarks.csv"))
    model = str(d / "models" / "gesture_model.joblib")
    reports = str(d / "reports")
    bundle = train.main(["--data", csv_path, "--model", model, "--reports", reports])
    return bundle, csv_path, model, reports


def test_train_runs_and_saves(trained):
    bundle, _, model, reports = trained
    assert os.path.exists(model)
    assert set(bundle["models"]) == {"random_forest", "mlp"} and bundle["name"] in bundle["models"]
    assert bundle["labels"] == sorted(POSES)
    assert "take" in bundle["split_method"]  # 6 takes per label -> group-aware split
    for name in ("random_forest", "mlp"):
        assert os.path.getsize(os.path.join(reports, f"confusion_{name}.png")) > 0
        assert bundle["metrics"][name]["accuracy"] > 0.5
    # the saved (best) model should separate these easy poses; the MLP may lag on so little data
    assert bundle["metrics"][bundle["name"]]["macro_f1"] > 0.9


def test_split_keeps_takes_together(trained):
    from gesture.ml import load_dataset, split_dataset
    df = load_dataset(trained[1])
    tr, te, _ = split_dataset(df)
    assert not set(df.loc[tr, "take"]) & set(df.loc[te, "take"])
    assert 0.1 < len(te) / len(df) < 0.35
    assert set(df.loc[te, "label"]) == set(POSES)  # stratified: every label in the test set


def test_split_fallback_with_few_takes(tmp_path):
    from gesture.ml import load_dataset, split_dataset
    df = load_dataset(write_synthetic_csv(str(tmp_path / "few.csv"), takes=2, frames=10))
    tr, te, method = split_dataset(df)
    assert "by sample" in method
    assert not set(df.loc[tr, "sample_id"]) & set(df.loc[te, "sample_id"])


def test_evaluate_writes_comparison(trained, tmp_path):
    import evaluate
    _, _, model, _ = trained
    rows = evaluate.main(["--model", model, "--reports", str(tmp_path)])
    assert set(rows) == {"rules", "random_forest", "mlp"}
    text = (tmp_path / "comparison.md").read_text()
    assert "Rule-based" in text and "RandomForest" in text and "MLP" in text
    assert rows["rules"]["accuracy"] > 0.9  # the synthetic poses follow the rules too
    assert all(r["ms"] > 0 for r in rows.values())


def test_ml_classifier_predicts_and_rejects(trained):
    from gesture.ml import MLGestureClassifier
    _, _, model, _ = trained
    clf = MLGestureClassifier.load(model, min_confidence=0.6)
    assert clf.predict(make_hand({"index", "middle"}, "in"), "Right") == "peace"
    assert clf.predict(None) == "unknown"
    strict = MLGestureClassifier.load(model, min_confidence=1.01)  # nothing is that confident
    assert strict.predict(make_hand(ALL, "out"), "Right") == "unknown"
    assert 0 < strict.last_confidence <= 1
