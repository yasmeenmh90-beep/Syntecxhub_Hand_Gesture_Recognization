"""Compare the rule-based classifier with the trained RandomForest and MLP.

    python evaluate.py                 # uses models/gesture_model.joblib + its data CSV

All three are scored on the *same* held-out test samples that train.py set aside
(stored in the model bundle by sample_id). The rule-based classify() runs on the saved
raw landmarks; the ML models run on the normalised features. Inference time is measured
one sample at a time, as in the live demo. Writes reports/comparison.md.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from gesture import classify
from gesture.config import load_config, setting
from gesture.features import FEATURE_NAMES, raw_landmarks_from_row
from gesture.ml import DEFAULT_MODEL, load_dataset

PRETTY = {"rules": "Rule-based", "random_forest": "RandomForest", "mlp": "MLP"}


def _short_path(path: str) -> str:
    rel = os.path.relpath(path)
    return path if rel.startswith("..") else rel


def time_per_sample(fn, items, max_items=500, repeats=3) -> float:
    """Median-of-repeats mean time per call, in milliseconds."""
    items = items[:max_items]
    if not items:
        return float("nan")
    for it in items[:10]:  # warm-up
        fn(it)
    runs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        runs.append((time.perf_counter() - t0) / len(items))
    return float(np.median(runs) * 1000)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", help="CSV to evaluate on (default: the one train.py used)")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--config", help="config.yaml (for the rule-based pinch threshold)")
    args = ap.parse_args(argv)

    import joblib
    from sklearn.metrics import accuracy_score, f1_score

    if not os.path.exists(args.model):
        raise SystemExit(f"No model at {args.model}. Run: python train.py")
    bundle = joblib.load(args.model)
    data_path = args.data or bundle["data_path"]
    df = load_dataset(data_path)
    test = df[df["sample_id"].isin(set(bundle["test_sample_ids"]))].reset_index(drop=True)
    if test.empty:
        raise SystemExit("None of the model's test samples are in this CSV; retrain with train.py.")
    labels = list(bundle["labels"])
    y = test["label"].to_numpy()
    X = test[FEATURE_NAMES].to_numpy(dtype=np.float64)
    raw = [raw_landmarks_from_row(r) for _, r in test.iterrows()]
    ok_ratio = setting(load_config(args.config), "pinch", "ok_ratio", 0.25)

    preds, times = {}, {}
    preds["rules"] = np.array([classify(lm, ok_pinch_ratio=ok_ratio) for lm in raw])
    times["rules"] = time_per_sample(lambda lm: classify(lm, ok_pinch_ratio=ok_ratio), raw)

    from gesture.features import extract_features
    hands = test["handedness"].tolist()
    for name, model in bundle["models"].items():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1  # per-frame use: one row at a time
        preds[name] = model.predict(X)
        pairs = list(zip(raw, hands))
        times[name] = time_per_sample(
            lambda p, m=model: m.predict_proba(extract_features(p[0], p[1])[None, :]), pairs)

    rows = {}
    for name, pred in preds.items():
        rows[name] = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
            "ms": times[name],
        }

    best_ml = bundle["name"]
    lines = [
        "# Gesture classifier comparison",
        "",
        f"Test set: **{len(test)} samples** ({test['label'].nunique()} labels) held out by "
        f"`train.py`, split {bundle['split_method']}. Data: `{_short_path(data_path)}` "
        f"({bundle['n_samples']} samples total).",
        "",
        "| Classifier | Accuracy | Macro F1 | Inference / sample |",
        "|---|---:|---:|---:|",
    ]
    for name, r in rows.items():
        tag = " (saved model)" if name == best_ml else ""
        lines.append(f"| {PRETTY.get(name, name)}{tag} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} "
                     f"| {r['ms']:.3f} ms |")
    lines += [
        "",
        "Inference time is one sample at a time (as in the live demo). For the ML models it",
        "includes feature normalisation and `predict_proba`.",
        "",
        "Rule-based predictions never use the ML-only `unknown` training class, and they return",
        "`unknown` for poses outside their rules, which counts as an error here.",
        "",
        "Per-label accuracy (test set):",
        "",
        "| Label | n | " + " | ".join(PRETTY.get(n, n) for n in rows) + " |",
        "|---|---:|" + "---:|" * len(rows),
    ]
    for lab in labels:
        mask = y == lab
        if not mask.any():
            continue
        cells = [f"{(preds[n][mask] == lab).mean():.2f}" for n in rows]
        lines.append(f"| {lab} | {int(mask.sum())} | " + " | ".join(cells) + " |")
    report = "\n".join(lines) + "\n"

    os.makedirs(args.reports, exist_ok=True)
    out = os.path.join(args.reports, "comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"-> {out}")
    return rows


if __name__ == "__main__":
    main()
