# Object Classification — KNN vs Gaussian Naïve Bayes

Fourteen-step Jupyter assignment that classifies **synthetic robotic sensor objects** into `metal_can`, `plastic_bottle`, `wooden_block`, `rubber_ball` using **k-nearest neighbors** and **GaussianNB**, then `GridSearchCV` and `joblib` persistence. Machine Learning for Robotics **A#04, Spring 2026**, student **i222327 (Mohammad Rohaan)**.

## Table of contents

- [Problem statement / academic context](#problem-statement--academic-context)
- [Features](#features)
- [Architecture / design](#architecture--design)
- [Algorithms and data structures](#algorithms-and-data-structures)
- [File-by-file reference](#file-by-file-reference)
- [Data formats / schemas](#data-formats--schemas)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Prerequisites and install](#prerequisites-and-install)
- [How to build and run](#how-to-build-and-run)
- [Usage walkthrough](#usage-walkthrough)
- [Configuration / constants](#configuration--constants)
- [Results / metrics](#results--metrics)
- [Controls / CLI](#controls--cli)
- [Known limitations / bugs](#known-limitations--bugs)
- [How to extend](#how-to-extend)
- [Author](#author)

## Problem statement / academic context

A robot “sees” five numeric sensor-style channels and must label the object class. The notebook **does not use a real sensor log**; it generates 500 rows, preprocesses them, trains two classical classifiers, compares confusion matrices / classification reports, tunes hyperparameters, sketches a real-time `predict` helper, and saves `.pkl` files. Environment printout: **NumPy 2.0.2**, **pandas 2.2.2**.

## Features

The 14 markdown steps:

| Step | Title | What the code does |
|------|-------|-------------------|
| 1 | Environment setup | Imports, `np.random.seed(42)`, `sns.set_style("whitegrid")` |
| 2 | Synthetic dataset | 500 rows, **features independent of labels** (`np.random.choice` for class) |
| 3 | Preprocessing | `isnull`, `StandardScaler` on all 5 features, `LabelEncoder` on `label`, 80/20 split → `X_train` (400×5), `X_test` (100×5) |
| 4 | Improved dataset | Class-conditional ranges, 125 samples/class, shuffle, **overwrite** `synthetic_robotic_dataset.csv` |
| 5 | KNN | `KNeighborsClassifier(n_neighbors=3)` on **Step 3** split |
| 6 | Naïve Bayes | `GaussianNB()` on the same split |
| 7 | Comparison | Labeled confusion DataFrames + metric lecture |
| 8 | Heatmaps / bars | Seaborn heatmaps; bar plots reference **undefined** `knn_report_data` / `nb_report_data` |
| 9 | Evaluation summary | Repeats accuracy/CM/report with `target_names=labels` |
| 10 | KNN GridSearch | `n_neighbors` 3–11, `weights` uniform/distance, `metric` euclidean/manhattan, `cv=5` (20×5 = 100 fits) |
| 11 | NB GridSearch | `var_smoothing` ∈ {1e-9 … 1e-5}, 25 fits |
| 12 | Final comparison | Tuned models; broken precision parser |
| 13 | Real-time stub | `preprocess_real_time_data` / `predict_with_trained_model` |
| 14 | Save models | `joblib.dump` → `best_knn_model.pkl`, `best_nb_model.pkl` |

## Architecture / design

```
Step 2 random CSV  →  Step 3 scale+encode+split  →  X_train, y_train, X_test, y_test
Step 4 class-conditional CSV (saved, NOT re-split)
Steps 5–12 train/evaluate on the Step 3 split (random labels ⇒ ~chance accuracy)
Step 13–14 predict [50,30,120,300,1.5] with a fresh StandardScaler.fit_transform on one row
```

No project classes except sklearn estimators. `label_encoder` maps strings to integers (preview: `metal_can→0`, `rubber_ball→2` in the first head after encode — order is alphabetical from `LabelEncoder`: metal_can, plastic_bottle, rubber_ball, wooden_block unless the first dataset’s unique order differs; reports later assume `labels = ['metal_can', 'plastic_bottle', 'wooden_block', 'rubber_ball']` which **need not match** encoder order).

## Algorithms and data structures

- **KNN**: class = majority vote among k nearest training points in scaled Euclidean (or Manhattan) space. Sensitive to scale (hence `StandardScaler`) and to label noise.
- **GaussianNB**: \(P(x_i \mid c) = \mathcal{N}(\mu_{c,i}, \sigma_{c,i}^2)\); class prior from frequencies; `var_smoothing` adds a portion of the largest variance to all features for numerical stability.
- **GridSearchCV**: exhaustive Cartesian product, 5-fold CV, `scoring='accuracy'`.
- **Improved generator (Step 4)** class-conditional ranges:

| Class | distance | size | color | weight (g) | voltage (V) |
|-------|----------|------|-------|------------|-------------|
| metal_can | 15–90 | 8–20 | 120–220 | 250–500 | 3.0–5.0 |
| plastic_bottle | 15–90 | 18–40 | 60–180 | 30–150 | 0.5–2.0 |
| wooden_block | 15–90 | 10–30 | 40–130 | 150–320 | 1.5–3.2 |
| rubber_ball | 15–90 | 6–18 | 150–255 | 80–220 | 1.0–2.8 |

Distance overlaps completely; **weight and voltage** separate metal vs plastic well — this dataset would be learnable if the split were rebuilt after Step 4.

## File-by-file reference

| File | Description |
|------|-------------|
| `i222327_ML_A#_04.ipynb` | Main notebook (28 cells). TREE also lists `i222327_ML_A# 04.ipynb` (space in name). |
| `Report.pdf` / `Report.docx` | Written report. |
| `i222327_ML_A# 04.ipynb - Colab.pdf` | Colab export. |
| `Asg-4-ML for Robotics-Spring 2026-NU.pdf` | Brief. |
| `synthetic_robotic_dataset.csv` | **Generated** at runtime (not stored in this snapshot). |
| `best_knn_model.pkl` / `best_nb_model.pkl` | **Generated** by Step 14. |

## Data formats / schemas

CSV columns:

```text
distance,size,color_intensity,weight,sensor_voltage,label
```

Units as documented in Step 2: cm, cm, 0–255 intensity, grams, volts, string class.

Example incoming vector in Step 13:

```python
np.array([[50, 30, 120, 300, 1.5]])
```

After a **one-row** `StandardScaler.fit_transform`, every z-score is **0.0** (variance undefined/zero). Prediction executed: class **`[0]`** (with a sklearn warning that the estimator was fitted with feature names).

## Tech stack

- Python 3 / Jupyter / Colab
- numpy 2.0.2, pandas 2.2.2 (printed)
- matplotlib, seaborn
- sklearn: `KNeighborsClassifier`, `GaussianNB`, `GridSearchCV`, `StandardScaler`, `LabelEncoder`, `train_test_split`, `accuracy_score`, `confusion_matrix`, `classification_report`
- `joblib` for `.pkl`

## Project structure

```
Object_Classification/
├── i222327_ML_A#_04.ipynb
├── Report.pdf
├── Report.docx
├── i222327_ML_A# 04.ipynb - Colab.pdf
└── Asg-4-ML for Robotics-Spring 2026-NU.pdf
```

## Prerequisites and install

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```

## How to build and run

```bash
cd Object_Classification
jupyter notebook "i222327_ML_A#_04.ipynb"
```

Run all cells. Step 2 and Step 4 both write `synthetic_robotic_dataset.csv` in the working directory. Step 14 writes the two pickle files.

## Usage walkthrough

1. Step 1 — confirm library versions.
2. Step 2 — inspect random labels (`head` can show three `metal_can` in a row; labels are independent of sensors).
3. Step 3 — 0 missing values; scaled features; split 400/100.
4. Step 4 — balanced 125×4 improved CSV. **Re-run preprocessing on `df` here if you want the later models to use it** (the stock notebook does not).
5. Steps 5–7 — expect ~25% KNN / ~31% NB on the random-label split (4-class chance is 25%).
6. Step 8 — heatmaps should display; bar charts may throw `NameError` unless you define `knn_report_data`.
7. Steps 10–11 — GridSearch verbose fits; best KNN `{'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}` at **31%**; NB `var_smoothing=1e-09` at **31%**.
8. Step 14 — load pickles and predict (still all-zero scaled stub).

## Configuration / constants

| Item | Value |
|------|-------|
| `np.random.seed` | 42 (Steps 1 and 4) |
| `num_samples` (Step 2) | 500 |
| `samples_per_class` (Step 4) | 125 |
| `test_size` | 0.2 |
| Default KNN k | 3 |
| KNN grid | k ∈ {3,5,7,9,11}, weights, metric |
| NB grid | `var_smoothing` 1e-9 … 1e-5 |
| `cv` | 5 |
| CSV / pkl names | `synthetic_robotic_dataset.csv`, `best_knn_model.pkl`, `best_nb_model.pkl` |

## Results / metrics

**Default models (random-label test set, n=100):**

| Model | Accuracy | Notes |
|-------|----------|--------|
| KNN k=3 | **22.00%** | Wooden_block recall 0.00 in the report |
| GaussianNB | **31.00%** | Best of the untuned pair |

KNN CM (rows true, cols pred, encoder ints 0–3):

```
[[13  7  3  8]
 [11  4  2  4]
 [12 10  0  3]
 [11  4  3  5]]
```

**After GridSearch (same test set):**

| Model | Best params | Accuracy |
|-------|-------------|----------|
| KNN | euclidean, k=7, distance weights | **31.00%** |
| GaussianNB | `var_smoothing=1e-09` (the default) | **31.00%** |

Step 12 prints “Naïve Bayes model performed better in terms of accuracy” because the comparison is `if knn_accuracy > nb_accuracy` else NB — **ties go to NB**.

Per-class reports (tuned): KNN metal_can P/R/F1 0.43/0.42/0.43; NB metal_can 0.36/0.52/0.43. Macro F1 ~0.30 vs ~0.29.

## Controls / CLI

Notebook only. `predict_with_trained_model(data, model='knn'|'naive_bayes')`.

## Known limitations / bugs

1. **Step 4 never refreshes `X_train`/`X_test`.** All classifiers learn the **Step 2 random-label** data. Accuracies near chance are expected and do **not** evaluate the class-conditional generator.
2. **Step 8** uses `knn_report_data` / `nb_report_data` never assigned (`classification_report(..., output_dict=True)` was omitted).
3. **Step 12 precision loop** does `.splitlines()[i+2].split()[0]`, which is the **class name**, not precision — prints `KNN Precision = metal_can`.
4. **Label name list vs LabelEncoder order** can mislabel confusion-matrix axes (`wooden_block` vs `rubber_ball` swap risk).
5. **Real-time scaler** calls `fit_transform` on a **single** sample → all zeros; should reuse the Step 3 `scaler`.
6. Feature-name warning: ndarray vs DataFrame columns at predict time.
7. Step 8 `plt.subplots_adjust` after `plt.close()` can emit an empty figure (executed output includes a 0-axes figure).
8. Models saved without the scaler/`LabelEncoder` — raw sensor units will not match training space.

## How to extend

- After Step 4, copy the Step 3 preprocessing onto `df` and re-split **before** KNN/NB (expect a large accuracy jump).
- `classification_report(..., output_dict=True)` for bar charts.
- Persist `joblib.dump({'model': best_knn, 'scaler': scaler, 'encoder': label_encoder}, ...)`.
- Calibrate / try SVM or RandomForest as a third baseline.
- Replace the stub with a ROS or serial callback that applies the **stored** scaler.

## Author

**Mohammad Rohaan** — i222327 · [rohaan2802](https://github.com/rohaan2802)
