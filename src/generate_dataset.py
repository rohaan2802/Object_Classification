"""Build a 500-row synthetic robotic-sensor table.

Each of the four object classes has its own ranges for size, colour
intensity, weight and sensor voltage. Distance is shared (15–90 cm).
Seed 42, 125 rows per class.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["distance", "size", "color_intensity", "weight", "sensor_voltage"]
CLASSES = ["metal_can", "plastic_bottle", "wooden_block", "rubber_ball"]

# (distance, size, color_intensity, weight, sensor_voltage)
# color is integer; the rest are continuous.
RANGES = {
    "metal_can":       dict(distance=(15, 90), size=(8, 20),  color=(120, 221), weight=(250, 500), voltage=(3.0, 5.0)),
    "plastic_bottle":  dict(distance=(15, 90), size=(18, 40), color=(60, 181),  weight=(30, 150),  voltage=(0.5, 2.0)),
    "wooden_block":    dict(distance=(15, 90), size=(10, 30), color=(40, 131),  weight=(150, 320), voltage=(1.5, 3.2)),
    "rubber_ball":     dict(distance=(15, 90), size=(6, 18),  color=(150, 256), weight=(80, 220),  voltage=(1.0, 2.8)),
}

SAMPLES_PER_CLASS = 125
SEED = 42


def generate(seed=SEED, samples_per_class=SAMPLES_PER_CLASS):
    rng = np.random.RandomState(seed)
    rows = []
    for label, r in RANGES.items():
        for _ in range(samples_per_class):
            rows.append(
                {
                    "distance": rng.uniform(*r["distance"]),
                    "size": rng.uniform(*r["size"]),
                    "color_intensity": rng.randint(*r["color"]),
                    "weight": rng.uniform(*r["weight"]),
                    "sensor_voltage": rng.uniform(*r["voltage"]),
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def default_csv_path():
    return Path(__file__).resolve().parents[1] / "data" / "synthetic_robotic_dataset.csv"


def main(path=None):
    df = generate()
    path = Path(path) if path else default_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")
    print(df.head().to_string())
    print()
    print("Class counts:")
    print(df["label"].value_counts().to_string())
    print()
    print("Feature means by class:")
    print(df.groupby("label")[FEATURES].mean().to_string(float_format=lambda v: f"{v:,.2f}"))
    return df, path


if __name__ == "__main__":
    main()
