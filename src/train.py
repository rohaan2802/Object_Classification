"""Preprocess the synthetic table, train KNN and GaussianNB, and tune both.

The fitted scaler, label encoder and the two tuned models are written
together to models/object_classifiers.joblib so predict.py can reuse
the same scaling and class names.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from generate_dataset import FEATURES, default_csv_path, generate


def default_model_path():
    return Path(__file__).resolve().parents[1] / "models" / "object_classifiers.joblib"


def load_data(path=None):
    path = Path(path) if path else default_csv_path()
    if path.exists():
        return pd.read_csv(path)
    df, path = __import__("generate_dataset").main(path)
    return df


def preprocess(df, test_size=0.2, random_state=42):
    X = df[FEATURES].copy()
    y_raw = df["label"].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURES, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURES, index=X_test.index
    )
    return X_train_s, X_test_s, y_train, y_test, scaler, encoder


def train_defaults(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return knn, nb


def tune_knn(X_train, y_train):
    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid={
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def tune_nb(X_train, y_train):
    grid = GridSearchCV(
        GaussianNB(),
        param_grid={"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def report(name, y_test, y_pred, class_names):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"{name} accuracy: {acc * 100:.2f}%")
    print("Confusion matrix (rows = actual, columns = predicted)")
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df.to_string())
    print()
    print(cr)
    return acc, cm, cr


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows")
    print("Missing values:", int(df.isnull().sum().sum()))
    print()

    X_train, X_test, y_train, y_test, scaler, encoder = preprocess(df)
    class_names = list(encoder.classes_)
    print(f"Train rows: {len(X_train)}    Test rows: {len(X_test)}")
    print("Class order (LabelEncoder):", class_names)
    print("Test class counts:", {class_names[i]: int((y_test == i).sum()) for i in range(len(class_names))})
    print()

    knn, nb = train_defaults(X_train, y_train)
    print("=" * 60)
    print("Default models  (KNN k=3, GaussianNB defaults)")
    print("=" * 60)
    knn_acc, knn_cm, knn_cr = report("KNN (k=3)", y_test, knn.predict(X_test), class_names)
    nb_acc, nb_cm, nb_cr = report("GaussianNB", y_test, nb.predict(X_test), class_names)

    print("=" * 60)
    print("Grid search")
    print("=" * 60)
    knn_grid = tune_knn(X_train, y_train)
    print("Best KNN params:", knn_grid.best_params_,
          f"  (cv accuracy {knn_grid.best_score_ * 100:.2f}%)")
    best_knn = knn_grid.best_estimator_
    best_knn_acc, best_knn_cm, best_knn_cr = report(
        "Tuned KNN", y_test, best_knn.predict(X_test), class_names
    )

    nb_grid = tune_nb(X_train, y_train)
    print("Best GaussianNB params:", nb_grid.best_params_,
          f"  (cv accuracy {nb_grid.best_score_ * 100:.2f}%)")
    best_nb = nb_grid.best_estimator_
    best_nb_acc, best_nb_cm, best_nb_cr = report(
        "Tuned GaussianNB", y_test, best_nb.predict(X_test), class_names
    )

    print("=" * 60)
    print("Tuned comparison")
    print("=" * 60)
    print(f"KNN        {best_knn_acc * 100:.2f}%")
    print(f"GaussianNB {best_nb_acc * 100:.2f}%")
    if best_knn_acc > best_nb_acc:
        winner = "KNN"
    elif best_nb_acc > best_knn_acc:
        winner = "GaussianNB"
    else:
        winner = "tie"
    print(f"Higher test accuracy: {winner}")

    bundle = {
        "scaler": scaler,
        "encoder": encoder,
        "features": FEATURES,
        "knn": best_knn,
        "nb": best_nb,
        "knn_params": knn_grid.best_params_,
        "nb_params": nb_grid.best_params_,
    }
    out = default_model_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"\nSaved models, scaler and encoder to {out}")
    return bundle, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
