"""Shared ML helpers: dataset loading, the leak-free stratified split, and the
runtime classifier used by demo.py --classifier ml."""
from __future__ import annotations

import os

import numpy as np

from .features import COORDS, CSV_COLUMNS, FEATURE_NAMES, extract_features

DEFAULT_DATA = os.path.join("data", "landmarks.csv")
DEFAULT_MODEL = os.path.join("models", "gesture_model.joblib")


def load_dataset(path: str = DEFAULT_DATA):
    import pandas as pd
    if not os.path.exists(path):
        raise SystemExit(f"No data at {path}. Record some with: python collect_data.py")
    df = pd.read_csv(path)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    df = df.dropna(subset=FEATURE_NAMES + ["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    df["take"] = df["take"].astype(str)
    if df["label"].nunique() < 2:
        raise SystemExit("Need samples for at least 2 gestures to train.")
    return df


def split_dataset(df, test_size: float = 0.2, seed: int = 42):
    """Stratified ~80/20 split that never puts near-duplicates on both sides.

    - If every label has >= 5 recording takes (one SPACE hold = one take), whole takes
      go to train or test (StratifiedGroupKFold): consecutive frames of a take are
      near-identical, so splitting them would inflate test scores.
    - Otherwise falls back to splitting by sample_id (a sample and its mirrored copy
      stay together), stratified by label.
    Returns (train_index, test_index, method).
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    takes_per_label = df.groupby("label")["take"].nunique()
    n_splits = max(2, int(round(1 / test_size)))
    if takes_per_label.min() >= n_splits:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["take"]))
        return np.sort(train_idx), np.sort(test_idx), "by recording take (StratifiedGroupKFold)"

    samples = df.drop_duplicates("sample_id")[["sample_id", "label"]]
    counts = samples["label"].value_counts()
    stratify = samples["label"] if counts.min() >= 2 else None
    train_ids, test_ids = train_test_split(samples["sample_id"], test_size=test_size,
                                           random_state=seed, stratify=stratify)
    test_set = set(test_ids)
    is_test = df["sample_id"].isin(test_set).to_numpy()
    method = "by sample (stratified; frames of one take may be on both sides)"
    return np.flatnonzero(~is_test), np.flatnonzero(is_test), method


class MLGestureClassifier:
    """Wraps the saved model bundle; `predict` returns a label or "unknown"."""

    def __init__(self, bundle: dict, min_confidence: float = 0.6):
        self.model = bundle["model"]
        self.labels = list(bundle["labels"])
        self.name = bundle.get("name", "model")
        self.min_confidence = min_confidence
        if list(bundle.get("feature_names", FEATURE_NAMES)) != FEATURE_NAMES:
            raise ValueError("model was trained on different features; retrain with train.py")
        if bundle.get("coords") != COORDS:
            print("[ml] warning: this model was trained on landmarks recorded before the aspect-ratio "
                  "fix; re-record with collect_data.py and retrain for best accuracy.")
        classes = [str(c) for c in getattr(self.model, "classes_", self.labels)]
        self._classes = classes
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1  # one row per frame: thread start-up costs more than it saves
        self.last_confidence = 0.0

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL, min_confidence: float = 0.6):
        import joblib
        if not os.path.exists(path):
            raise SystemExit(f"No model at {path}. Train one with: python train.py")
        return cls(joblib.load(path), min_confidence)

    def predict_proba(self, lm, handedness: str = "Right") -> dict[str, float]:
        x = extract_features(lm, handedness)[None, :]
        return dict(zip(self._classes, self.model.predict_proba(x)[0]))

    def predict(self, lm, handedness: str = "Right") -> str:
        if lm is None or len(lm) != 21:
            return "unknown"
        x = extract_features(lm, handedness)[None, :]
        proba = self.model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        self.last_confidence = float(proba[i])
        return self._classes[i] if proba[i] >= self.min_confidence else "unknown"
