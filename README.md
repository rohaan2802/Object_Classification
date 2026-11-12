# Object Classification from Robotic Sensor Features (CS-4090)

**Machine Learning for Robotics assignment from Spring 2026 at FAST-NUCES — KNN and Gaussian Naive Bayes on synthetic sensor readings.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-5b21b6?style=for-the-badge&logo=python&logoColor=white)](https://github.com/rohaan2802/Object_Classification)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN%20%2B%20GaussianNB-0e7490?style=for-the-badge)](https://github.com/rohaan2802/Object_Classification)
[![Course](https://img.shields.io/badge/Course-CS--4090-5b21b6?style=for-the-badge)](https://github.com/rohaan2802/Object_Classification)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## Live Links

| What | Link |
|------|------|
| Repository | [https://github.com/rohaan2802/Object_Classification](https://github.com/rohaan2802/Object_Classification) |
| Website (GitHub Pages) | [https://rohaan2802.github.io/Object_Classification/](https://rohaan2802.github.io/Object_Classification/) |
| Download (ZIP) | [https://github.com/rohaan2802/Object_Classification/archive/refs/heads/main.zip](https://github.com/rohaan2802/Object_Classification/archive/refs/heads/main.zip) |

---

## Table of Contents

1. [Live Links](#live-links)
2. [Overview](#overview)
3. [What's Inside](#whats-inside)
4. [Dataset](#dataset)
5. [Models and Results](#models-and-results)
6. [Folder Structure](#folder-structure)
7. [How to Run](#how-to-run)
8. [Screenshots](#screenshots)
9. [License](#license)
10. [Author](#author)

---

## Overview

This is my CS-4090 Machine Learning for Robotics assignment (Assignment 4, Spring 2026) at FAST-NUCES, with Dr. Maryam Iqbal. The brief asked for a classification system a robot could use to tell objects apart from sensor readings, using K-Nearest Neighbours and Naive Bayes.

I generated a 500-row synthetic table — 125 rows each of metal can, plastic bottle, wooden block and rubber ball — with five features: distance, size, colour intensity, weight and sensor voltage. Each class has its own ranges for size, colour, weight and voltage. Distance is shared (15–90). The data is in [`data/synthetic_robotic_dataset.csv`](data/synthetic_robotic_dataset.csv) and can be rebuilt with the same seed from [`src/generate_dataset.py`](src/generate_dataset.py).

I scaled the features, encoded the labels, and held out 100 stratified rows for testing. Default KNN (k=3) scores 99% on that test set (one wooden block is called a rubber ball). Gaussian Naive Bayes scores 100%. After a small grid search both models reach 100% on the same 100 rows. Those figures are high because the classes were generated with fairly distinct weight and voltage ranges — this is a clean synthetic table, not a claim about real sensors.

The write-up I submitted is in [`report/report.pdf`](report/report.pdf). It recorded 31% for both models. That run had trained on a first version of the table where labels were assigned at random, so the features had no relationship with the class. The scripts and notebook in this repo train on the class-conditional table described above.

> Course: **CS-4090 – Machine Learning for Robotics** · **Spring 2026** · **FAST-NUCES** · Assignment 4 · Roll: **22I-2327**

---

## What's Inside

| Item | Path | Notes |
|------|------|-------|
| Assignment brief | [assignment/assignment-04.pdf](assignment/assignment-04.pdf) | The original Assignment 4 statement |
| Submitted report | [report/report.pdf](report/report.pdf), [report/report.docx](report/report.docx) | The write-up I turned in (the 31% run) |
| Earlier Colab printout | [report/notebook-colab-export.pdf](report/notebook-colab-export.pdf) | PDF export of an earlier notebook run |
| Notebook | [notebooks/object_classification.ipynb](notebooks/object_classification.ipynb) | Full pipeline, with outputs |
| Dataset generator | [src/generate_dataset.py](src/generate_dataset.py) | 500 rows, seed 42 |
| Training and evaluation | [src/train.py](src/train.py) | Preprocess, KNN, GaussianNB, grid search |
| Predict a new reading | [src/predict.py](src/predict.py) | Reuses the saved scaler and class names |
| Dataset | [data/synthetic_robotic_dataset.csv](data/synthetic_robotic_dataset.csv) | The 500-row table |
| Screenshots | [screenshots/](screenshots/) | The captures used below |

---

## Dataset

500 rows, 125 per class, no missing values. Seed 42.

| Class | Distance | Size | Colour intensity | Weight | Sensor voltage |
|-------|----------|------|------------------|--------|----------------|
| metal_can | 15–90 | 8–20 | 120–220 | 250–500 | 3.0–5.0 |
| plastic_bottle | 15–90 | 18–40 | 60–180 | 30–150 | 0.5–2.0 |
| wooden_block | 15–90 | 10–30 | 40–130 | 150–320 | 1.5–3.2 |
| rubber_ball | 15–90 | 6–18 | 150–255 | 80–220 | 1.0–2.8 |

Feature means on the generated table:

| Class | Distance | Size | Colour | Weight | Voltage |
|-------|----------|------|--------|--------|---------|
| metal_can | 51.56 | 14.20 | 169.66 | 380.92 | 4.05 |
| plastic_bottle | 49.95 | 29.04 | 123.47 | 88.77 | 1.25 |
| rubber_ball | 52.29 | 12.04 | 203.02 | 151.99 | 1.86 |
| wooden_block | 51.94 | 20.51 | 84.52 | 237.70 | 2.35 |

Weight and voltage are the strongest separators. Wooden block and rubber ball overlap more than the others, which is where the one default-KNN mistake happened.

---

## Models and Results

80/20 split, stratified, seed 42: 400 training rows and 100 test rows, 25 per class. Features are scaled with `StandardScaler` fitted on the training set only. Class names follow `LabelEncoder` order: `metal_can`, `plastic_bottle`, `rubber_ball`, `wooden_block`.

### Default models

| Model | Test accuracy | Notes |
|-------|----------------|-------|
| KNN (k=3) | 99% | One wooden block predicted as rubber_ball |
| GaussianNB | 100% | All 100 test rows correct |

KNN (k=3) per class on the test set:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| metal_can | 1.0000 | 1.0000 | 1.0000 |
| plastic_bottle | 1.0000 | 1.0000 | 1.0000 |
| rubber_ball | 0.9615 | 1.0000 | 0.9804 |
| wooden_block | 1.0000 | 0.9600 | 0.9796 |

### After grid search

| Model | Best parameters | 5-fold CV accuracy | Test accuracy |
|-------|-----------------|--------------------|---------------|
| KNN | metric=`manhattan`, k=11, weights=`distance` | 99.25% | 100% |
| GaussianNB | `var_smoothing=1e-09` | 99.75% | 100% |

On this test set the two tuned models tie at 100%. GaussianNB was already perfect with its defaults; the search kept the default smoothing and the 5-fold score is 99.75%. I would pick GaussianNB if I had to choose one here — it is already at 100% without extra knobs, and it is cheaper to evaluate than a 11-neighbour KNN.

`src/predict.py` applies the scaler that was fitted at training time. Fitting a new scaler on a single reading would collapse every feature to zero, so that is not used. A few checks:

```
python src/predict.py --distance 40 --size 12 --color 180 --weight 400 --voltage 4.2 --model knn
# metal_can

python src/predict.py --distance 40 --size 30 --color 100 --weight 80 --voltage 1.0 --model knn
# plastic_bottle

python src/predict.py --distance 50 --size 30 --color 120 --weight 300 --voltage 1.5 --model knn
# wooden_block
```

The fitted KNN, GaussianNB, scaler and label encoder are written together to `models/object_classifiers.joblib` when you run `src/train.py`. That file is left out of git on purpose — it is rebuilt in a few seconds.

---

## Folder Structure

```
Object_Classification/
├── assignment/
│   └── assignment-04.pdf
├── report/
│   ├── report.pdf
│   ├── report.docx
│   └── notebook-colab-export.pdf
├── notebooks/
│   └── object_classification.ipynb
├── src/
│   ├── generate_dataset.py
│   ├── train.py
│   └── predict.py
├── data/
│   └── synthetic_robotic_dataset.csv
├── models/                         # created by src/train.py
├── screenshots/
├── assets/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── index.html
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── .gitattributes
└── .nojekyll
```

---

## How to Run

Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
```

On Windows: `.venv\Scripts\activate`  
On Linux or macOS: `source .venv/bin/activate`

```bash
pip install -r requirements.txt
python src/generate_dataset.py
python src/train.py
python src/predict.py --distance 40 --size 12 --color 180 --weight 400 --voltage 4.2 --model knn
```

`generate_dataset.py` is optional if you just want to reuse the CSV already in `data/`. `train.py` will also generate the table if the CSV is missing.

The notebook is [`notebooks/object_classification.ipynb`](notebooks/object_classification.ipynb). Open it from the repository root, or from `notebooks/` — it looks for `src/` and `data/` in both places.

To preview the website locally, open `index.html` or serve the folder:

```bash
python -m http.server 8080
```

then visit [http://localhost:8080](http://localhost:8080).

---

## Screenshots

### 1. Repository Folder Structure

![Repository folder structure](screenshots/01-repo-structure.png)

### 2. Assignment Brief

![Assignment brief](screenshots/02-assignment-brief.png)

### 3. Dataset Sample and Class Counts

![Dataset sample and class counts](screenshots/03-dataset-sample.png)

### 4. Feature Means by Class

![Feature means by class](screenshots/04-feature-means.png)

### 5. Weight and Voltage by Class

![Weight and voltage scatter](screenshots/05-weight-voltage-scatter.png)

### 6. Feature Ranges by Class

![Feature box plots](screenshots/06-feature-boxplots.png)

### 7. KNN Default Classification Report

![KNN default report](screenshots/07-knn-default-report.png)

### 8. KNN Default Confusion Matrix

![KNN default confusion matrix](screenshots/08-knn-default-cm.png)

### 9. GaussianNB Classification Report

![GaussianNB report](screenshots/09-nb-default-report.png)

### 10. GaussianNB Confusion Matrix

![GaussianNB confusion matrix](screenshots/10-nb-default-cm.png)

### 11. Grid Search and Tuned Comparison

![Grid search and tuned comparison](screenshots/11-grid-search-comparison.png)

### 12. Classifying a New Sensor Reading

![Predict a new reading](screenshots/12-predict-new-reading.png)

### 13. Notebook Walkthrough

![Notebook walkthrough](screenshots/13-notebook-walkthrough.png)

### 14. Website Home (Dark Theme)

![Website home, dark theme](screenshots/14-website-home-dark.png)

### 15. Website Home (Light Theme)

![Website home, light theme](screenshots/15-website-home-light.png)

### 16. Website on a Phone

![Website on a phone](screenshots/16-website-phone.png)

### 17. Results Section on the Website

![Results section on the website](screenshots/17-website-results.png)

### 18. How to Run on the Website

![How to run on the website](screenshots/18-website-how-to-run.png)

---

## License

This project is released under the [MIT License](LICENSE). The assignment brief in `assignment/` and the report in `report/` remain course material and are included as I received or submitted them.

---

## Author

**Mohammad Rohaan**  
FAST-NUCES · 22I-2327  
[github.com/rohaan2802](https://github.com/rohaan2802)
