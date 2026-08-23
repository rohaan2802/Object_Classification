# Object Classification

Supervised classification of robotic sensor objects using **K-Nearest Neighbors (KNN)** and **Gaussian Naïve Bayes**, with preprocessing, hyperparameter tuning, evaluation, and model persistence.

## Overview

Machine Learning for Robotics assignment (**A#04**, Spring 2026) by **i222327**. Builds a **synthetic multi-class sensor dataset**, trains two classical classifiers, compares them, tunes with `GridSearchCV`, and saves best models with `joblib`.

### Object classes

`metal_can` · `plastic_bottle` · `wooden_block` · `rubber_ball`

### Sensor-style features

| Feature | Description |
|---------|-------------|
| `distance` | Proximity reading (cm) |
| `size` | Object size (cm) |
| `color_intensity` | Intensity proxy (0–255) |
| `weight` | Mass proxy (g) |
| `sensor_voltage` | Analog sensor response (V) |

An improved generator uses **class-conditional feature ranges** so classes are more separable than fully random labels.

## Repository Contents

| File | Description |
|------|-------------|
| `i222327_ML_A# 04.ipynb` | Main assignment notebook |
| `Report.pdf` / `Report.docx` | Written report |
| `i222327_ML_A# 04.ipynb - Colab.pdf` | Colab export |
| `Asg-4-ML for Robotics-Spring 2026-NU.pdf` | Assignment brief |

## Pipeline

1. Environment setup (NumPy, pandas, Matplotlib, Seaborn, scikit-learn)  
2. Synthetic dataset (~500 balanced samples) → `synthetic_robotic_dataset.csv`  
3. Preprocessing: missing checks, `StandardScaler`, `LabelEncoder`, train/test split  
4. **KNN** + accuracy / confusion matrix / classification report  
5. **GaussianNB** on the same features  
6. Comparison heatmaps and metric tables  
7. `GridSearchCV` for KNN (`n_neighbors`, `weights`, `metric`) and NB (`var_smoothing`)  
8. Real-time integration stub for a new sensor reading  
9. Export `best_knn_model.pkl` / `best_nb_model.pkl`  

## Tech Stack

Python 3 / Jupyter / Colab · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · joblib

## Getting Started

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
jupyter notebook "i222327_ML_A# 04.ipynb"
```

The notebook generates its own CSV — no external download required.

## Project Structure

```
Object_Classification/
├── i222327_ML_A# 04.ipynb
├── Report.pdf
├── Report.docx
├── i222327_ML_A# 04.ipynb - Colab.pdf
└── Asg-4-ML for Robotics-Spring 2026-NU.pdf
```

## Author

**Mohammad Rohaan** — i222327 · [rohaan2802](https://github.com/rohaan2802)
