"""Train ML gesture classifiers on data/landmarks.csv and save the best one.

    python train.py                               # defaults below
    python train.py --data data/landmarks.csv --model models/gesture_model.joblib --reports reports

Trains a RandomForest and an MLP (scikit-learn) on the 64 normalised features from
gesture/features.py, with a stratified 80/20 split that keeps recording takes together
(see gesture/ml.split_dataset). Prints accuracy, macro F1 and a classification report
per model, saves confusion matrices to reports/, and saves the best model (by macro F1)
with its label list to models/gesture_model.joblib.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np

from gesture.features import COORDS, FEATURE_NAMES
from gesture.ml import DEFAULT_DATA, DEFAULT_MODEL, load_dataset, split_dataset


def build_models(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=1,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=seed),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                           max_iter=800, random_state=seed)),
    }


PRETTY = {"random_forest": "RandomForest", "mlp": "MLP"}


def save_confusion_matrix(y_true, y_pred, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * len(labels), 0.6 + 0.7 * len(labels)), dpi=120)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax, cmap="RdPu",
                                            colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="where to save the best model")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    import joblib
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    df = load_dataset(args.data)
    train_idx, test_idx, method = split_dataset(df, args.test_size, args.seed)
    labels = sorted(df["label"].unique())
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()
    print(f"Loaded {len(df)} samples, {len(labels)} labels from {args.data}")
    print(f"Split {method}: {len(train_idx)} train / {len(test_idx)} test")
    print("Samples per label:", dict(df["label"].value_counts().sort_index()))

    os.makedirs(args.reports, exist_ok=True)
    results, fitted = {}, {}
    for name, model in build_models(args.seed).items():
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(X[train_idx], y[train_idx])
        fit_s = time.perf_counter() - t0
        pred = model.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)
        results[name] = {"accuracy": acc, "macro_f1": f1, "fit_seconds": fit_s}
        fitted[name] = model
        print(f"\n=== {PRETTY[name]}  accuracy {acc:.3f}  macro-F1 {f1:.3f}  (fit {fit_s:.1f}s)")
        print(classification_report(y[test_idx], pred, labels=labels, zero_division=0))
        png = os.path.join(args.reports, f"confusion_{name}.png")
        save_confusion_matrix(y[test_idx], pred, labels, f"{PRETTY[name]} (macro-F1 {f1:.3f})", png)
        print(f"confusion matrix -> {png}")

    best = max(results, key=lambda n: (results[n]["macro_f1"], results[n]["accuracy"]))
    bundle = {
        "model": fitted[best], "name": best, "labels": labels, "feature_names": FEATURE_NAMES,
        "coords": COORDS,
        "models": fitted,  # both, so evaluate.py can compare them on the same test set
        "test_sample_ids": sorted(set(df.loc[test_idx, "sample_id"])),
        "split_method": method, "metrics": results, "data_path": os.path.abspath(args.data),
        "n_samples": len(df), "seed": args.seed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
    joblib.dump(bundle, args.model, compress=3)
    print(f"\nBest: {PRETTY[best]} (macro-F1 {results[best]['macro_f1']:.3f}) -> {args.model}")
    return bundle


if __name__ == "__main__":
    main()
