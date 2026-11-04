"""Landmark feature normalisation + the landmarks CSV format.

Shared by collect_data.py (writing), train.py / evaluate.py (reading) and the ML
classifier at inference time, so every stage sees exactly the same features.

Features (64):
  21 landmarks x (x, y, z), translated so the wrist is at the origin and divided by
  palm size (wrist -> middle-finger MCP, measured in x/y), flattened as
  x0, y0, z0, x1, ..., z20; plus `handed` (+1 Right, -1 Left).
"""
from __future__ import annotations

import math

import numpy as np

WRIST, MIDDLE_MCP = 0, 9
N_LANDMARKS = 21

LANDMARK_FEATURES = [f"{a}{i}" for i in range(N_LANDMARKS) for a in "xyz"]
FEATURE_NAMES = LANDMARK_FEATURES + ["handed"]          # 64 model inputs
RAW_COLUMNS = [f"raw_{a}{i}" for i in range(N_LANDMARKS) for a in "xyz"]  # for rule-based eval
META_COLUMNS = ["label", "handedness", "mirrored", "sample_id", "take", "timestamp"]
CSV_COLUMNS = META_COLUMNS + FEATURE_NAMES + RAW_COLUMNS

# Coordinate system of the stored landmarks/features. "isotropic" = MediaPipe x and z
# scaled by frame width/height (Hand.iso), so distances and angles are true to the image.
COORDS = "isotropic"

# number key -> label (collect_data.py, demo.py 'p'). 1-9 are the rule-based gestures;
# 0 is a "none of these" class so the ML model can learn to say unknown.
KEY_LABELS = {
    "1": "open_palm", "2": "fist", "3": "thumbs_up", "4": "thumbs_down", "5": "peace",
    "6": "pointing", "7": "rock", "8": "ok", "9": "call_me", "0": "unknown",
}


def palm_size(lm) -> float:
    return math.hypot(lm[MIDDLE_MCP][0] - lm[WRIST][0], lm[MIDDLE_MCP][1] - lm[WRIST][1]) or 1e-6


def normalize_landmarks(lm) -> np.ndarray:
    """21 (x, y[, z]) points -> 63 floats: wrist at the origin, scaled by palm size."""
    if lm is None or len(lm) != N_LANDMARKS:
        raise ValueError(f"expected {N_LANDMARKS} landmarks, got {None if lm is None else len(lm)}")
    pts = np.array([(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in lm], dtype=np.float64)
    pts -= pts[WRIST]
    pts /= palm_size(lm)
    return pts.reshape(-1)


def handed_value(handedness: str) -> float:
    return 1.0 if str(handedness).lower().startswith("r") else -1.0


def extract_features(lm, handedness: str = "Right") -> np.ndarray:
    """The 64-feature vector used by the ML classifier."""
    return np.append(normalize_landmarks(lm), handed_value(handedness))


def mirror_landmarks(lm):
    """x-flip normalised image landmarks (x -> 1 - x). Pair with swap_handedness()."""
    return [(1.0 - p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in lm]


def swap_handedness(handedness: str) -> str:
    return "Left" if handed_value(handedness) > 0 else "Right"


def make_row(label: str, lm, handedness: str, sample_id: str, take: str,
             timestamp: float = 0.0, mirrored: bool = False) -> dict:
    """One CSV row: metadata + 64 normalised features + 63 raw landmark values."""
    feats = extract_features(lm, handedness)
    raw = [float(v) for p in lm for v in (p[0], p[1], p[2] if len(p) > 2 else 0.0)]
    row = {"label": label, "handedness": handedness, "mirrored": int(mirrored),
           "sample_id": sample_id, "take": take, "timestamp": round(float(timestamp), 3)}
    row.update(zip(FEATURE_NAMES, (float(v) for v in feats)))
    row.update(zip(RAW_COLUMNS, raw))
    return row


def rows_for_sample(label, lm, handedness, sample_id, take, timestamp=0.0, mirror_augment=False):
    """The row for a sample, plus its x-flipped copy when mirror_augment is on.

    The copy keeps the same sample_id so train/test splits keep the pair together.
    """
    rows = [make_row(label, lm, handedness, sample_id, take, timestamp)]
    if mirror_augment:
        rows.append(make_row(label, mirror_landmarks(lm), swap_handedness(handedness),
                             sample_id, take, timestamp, mirrored=True))
    return rows


def raw_landmarks_from_row(row) -> list[tuple[float, float, float]]:
    """Rebuild the 21 raw (x, y, z) landmarks from a CSV row (dict or pandas Series)."""
    return [(float(row[f"raw_x{i}"]), float(row[f"raw_y{i}"]), float(row[f"raw_z{i}"]))
            for i in range(N_LANDMARKS)]
