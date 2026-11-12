"""Classify one sensor reading with the saved KNN or GaussianNB model.

Uses the StandardScaler and LabelEncoder that were fitted at training
time. A fresh scaler fitted on a single row would collapse every
feature to zero and is not used here.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from generate_dataset import FEATURES
from train import default_model_path


def load_bundle(path=None):
    path = Path(path) if path else default_model_path()
    if not path.exists():
        raise SystemExit(
            f"No saved model at {path}. Run:  python src/train.py"
        )
    return joblib.load(path)


def predict(values, model="knn", bundle=None):
    bundle = bundle or load_bundle()
    if len(values) != len(FEATURES):
        raise ValueError(f"Expected {len(FEATURES)} features {FEATURES}, got {len(values)}")
    X = pd.DataFrame([values], columns=FEATURES)
    Xs = pd.DataFrame(bundle["scaler"].transform(X), columns=FEATURES)
    est = bundle["knn"] if model == "knn" else bundle["nb"]
    pred = est.predict(Xs)
    label = bundle["encoder"].inverse_transform(pred)[0]
    proba = None
    if hasattr(est, "predict_proba"):
        proba = dict(zip(bundle["encoder"].classes_, est.predict_proba(Xs)[0]))
    return label, proba, Xs.iloc[0].to_dict()


def main():
    p = argparse.ArgumentParser(description="Classify a robotic sensor reading.")
    p.add_argument("--distance", type=float, required=True)
    p.add_argument("--size", type=float, required=True)
    p.add_argument("--color", type=float, required=True, dest="color_intensity")
    p.add_argument("--weight", type=float, required=True)
    p.add_argument("--voltage", type=float, required=True, dest="sensor_voltage")
    p.add_argument("--model", choices=["knn", "nb"], default="knn")
    args = p.parse_args()
    values = [args.distance, args.size, args.color_intensity, args.weight, args.sensor_voltage]
    label, proba, scaled = predict(values, model=args.model)
    print("Input (raw):    ", dict(zip(FEATURES, values)))
    print("Input (scaled): ", {k: round(v, 4) for k, v in scaled.items()})
    print(f"Model:          {args.model}")
    print(f"Predicted class: {label}")
    if proba:
        print("Class probabilities:")
        for name, pr in proba.items():
            print(f"    {name:16s}  {pr:.4f}")


if __name__ == "__main__":
    main()
