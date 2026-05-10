"""
train.py
--------
Step 4: Train a Random Forest classifier on the cleaned dataset,
        evaluate it, and save the model artefacts.

Settings tuned for large datasets (150k rows, 320 features, 587 classes)
that must fit in limited RAM while keeping accuracy high:

  - No cross-validation  : CV trains the model N extra times = N× RAM.
                           The held-out test set (20%) gives the same info.
  - max_depth=30         : Deep enough to separate 587 classes, bounded
                           enough to not blow up memory (None = OOM).
  - max_samples=0.8      : Each tree sees 80% of rows. Tiny accuracy cost,
                           large RAM saving vs None (100%).
  - n_estimators=100     : Solid ensemble. Diminishing returns after ~150.
  - n_jobs=1             : No parallel workers. Each worker copies the
                           dataset into its own memory space.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "..", "data")
CLEAN_PATH   = os.path.join(DATA_DIR, "disease_dataset_clean.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "..", "models")
GRAPH_DIR    = os.path.join(BASE_DIR, "..", "graphs")
MODEL_PATH   = os.path.join(MODEL_DIR, "model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
N_ESTIMATORS     = 150    # trees in the forest
MAX_DEPTH        = 50     # deep enough for 587 classes; bounded for RAM
MAX_FEATURES     = "sqrt" # sqrt(320) ≈ 18 features per split
MAX_SAMPLES      = 0.8    # each tree sees 80% of training rows
MIN_SAMPLES_LEAF = 1      # allow pure leaves — disease data is deterministic
MIN_CLASS_SIZE   = 10     # drop classes with fewer samples than this
MAX_ROWS         = 150_000
TOP_N_FEATURES   = 20


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data(path: str) -> tuple:
    print("[ train ] Loading cleaned dataset ...")
    df = pd.read_csv(path)

    target_candidates = [c for c in df.columns if "disease" in c or "prognosis" in c]
    if not target_candidates:
        raise ValueError("No disease/prognosis column found. Run clean.py first.")

    target_col = target_candidates[0]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(str).str.strip()

    print(f"[ train ] {len(df):,} samples | {X.shape[1]} features | {y.nunique()} classes")
    return X, y


# ── Filter rare classes ───────────────────────────────────────────────────────
def filter_rare_classes(X: pd.DataFrame, y: pd.Series, min_size: int) -> tuple:
    counts  = y.value_counts()
    keep    = counts[counts >= min_size].index
    dropped = counts[counts < min_size]

    if len(dropped):
        print(f"[ train ] Removing {len(dropped)} rare class(es) "
              f"({dropped.sum():,} rows) with < {min_size} samples each")

    mask = y.isin(keep)
    X_f  = X[mask].reset_index(drop=True)
    y_f  = y[mask].reset_index(drop=True)
    print(f"[ train ] After class filter: {len(X_f):,} rows | {y_f.nunique()} classes")
    return X_f, y_f


# ── Smart downsample ──────────────────────────────────────────────────────────
def smart_downsample(X: pd.DataFrame, y: pd.Series, max_rows: int) -> tuple:
    if len(X) <= max_rows:
        return X, y

    ratio   = max_rows / len(X)
    rng     = np.random.default_rng(RANDOM_STATE)
    indices = []
    for cls in y.unique():
        cls_idx = y[y == cls].index.tolist()
        n_keep  = max(1, int(len(cls_idx) * ratio))
        sampled = rng.choice(cls_idx, size=n_keep, replace=False)
        indices.extend(sampled.tolist())

    indices = sorted(indices)
    X_d = X.loc[indices].reset_index(drop=True)
    y_d = y.loc[indices].reset_index(drop=True)
    print(f"[ train ] Downsampled {len(X):,} -> {len(X_d):,} rows "
          f"(class distribution preserved)")
    return X_d, y_d


# ── Feature selection ─────────────────────────────────────────────────────────
def select_features(X: pd.DataFrame) -> pd.DataFrame:
    selector = VarianceThreshold(threshold=0.0)
    X_sel    = selector.fit_transform(X)
    kept     = X.columns[selector.get_support()].tolist()
    removed  = X.shape[1] - len(kept)

    if removed:
        print(f"[ train ] Removed {removed} zero-variance features "
              f"({len(kept)} remain)")
    else:
        print(f"[ train ] All {len(kept)} features have variance > 0")

    return pd.DataFrame(X_sel, columns=kept)


# ── Train ─────────────────────────────────────────────────────────────────────
def train(X: pd.DataFrame, y_enc: np.ndarray) -> tuple:

    # Split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
            stratify     = y_enc,
        )
        print("[ train ] Stratified split: OK")
    except ValueError:
        print("[ train ] WARNING: stratified split failed -> using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size    = TEST_SIZE,
            random_state = RANDOM_STATE,
        )

    print(f"[ train ] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # Model
    model = RandomForestClassifier(
        n_estimators     = N_ESTIMATORS,
        max_depth        = MAX_DEPTH,
        max_features     = MAX_FEATURES,
        max_samples      = MAX_SAMPLES,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        random_state     = RANDOM_STATE,
        class_weight     = "balanced_subsample",
        n_jobs           = 1,
    )

    print(f"\n[ train ] Fitting {N_ESTIMATORS} trees ...")
    print(f"          max_depth    = {MAX_DEPTH}")
    print(f"          max_features = {MAX_FEATURES}")
    print(f"          max_samples  = {MAX_SAMPLES}  (per tree)")
    print()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    print("[ train ] Training complete.")
    y_pred = model.predict(X_test)
    return model, X_test, y_test, y_pred


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(
    model  : RandomForestClassifier,
    X_test : pd.DataFrame,
    y_test : np.ndarray,
    y_pred : np.ndarray,
    le     : LabelEncoder,
) -> dict:
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    if   acc >= 0.90: grade = "Excellent"
    elif acc >= 0.75: grade = "Good"
    elif acc >= 0.60: grade = "Fair"
    else:             grade = "Poor - check dataset quality"

    print("\n" + "=" * 50)
    print("  MODEL PERFORMANCE  (held-out test set)")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}  ({acc * 100:.1f}%)  [{grade}]")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("=" * 50 + "\n")

    target_names = [str(c) for c in le.classes_]
    print("[ train ] Per-class report:")
    print(classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0,
    ))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ── Save plots ────────────────────────────────────────────────────────────────
def save_plots(model, X_test: pd.DataFrame, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Random Forest - Model Evaluation", fontsize=14, fontweight="bold")

    # Metrics bar chart
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["precision"],
              metrics["recall"],   metrics["f1"]]
    colors = ["#4C9BE8", "#63C87A", "#F5A623", "#E86B4C"]
    bars   = axes[0].bar(labels, values, color=colors, edgecolor="white")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_title("Aggregate Metrics (test set)")
    axes[0].set_ylabel("Score")
    axes[0].axhline(y=0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    axes[0].text(3.6, 0.91, "90%", color="gray", fontsize=8)
    for bar, v in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.015, f"{v:.3f}",
            ha="center", fontsize=10, fontweight="bold",
        )

    # Feature importance chart
    importances = model.feature_importances_
    feat_names  = list(X_test.columns)
    idx_sorted  = np.argsort(importances)[-TOP_N_FEATURES:]
    axes[1].barh(
        [feat_names[i].replace("_", " ") for i in idx_sorted],
        importances[idx_sorted],
        color="#4C9BE8", edgecolor="white",
    )
    axes[1].set_title(f"Top {TOP_N_FEATURES} Most Predictive Symptoms")
    axes[1].set_xlabel("Feature Importance")

    plt.tight_layout()
    graph_path = os.path.join(GRAPH_DIR, "model_evaluation.png")
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ train ] Graph saved -> {graph_path}")


# ── Save artefacts ────────────────────────────────────────────────────────────
def save_artefacts(model, le: LabelEncoder, columns: list) -> None:
    joblib.dump(model,   MODEL_PATH)
    joblib.dump(le,      ENCODER_PATH)
    joblib.dump(columns, COLUMNS_PATH)
    print(f"[ train ] Model   -> {MODEL_PATH}")
    print(f"[ train ] Encoder -> {ENCODER_PATH}")
    print(f"[ train ] Columns -> {COLUMNS_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    if not os.path.exists(CLEAN_PATH):
        print("[ERROR] Cleaned dataset not found. Run clean.py first.")
        return

    X, y = load_data(CLEAN_PATH)
    X, y = filter_rare_classes(X, y, MIN_CLASS_SIZE)
    X, y = smart_downsample(X, y, MAX_ROWS)
    X    = select_features(X)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    model, X_test, y_test, y_pred = train(X, y_enc)
    metrics = evaluate(model, X_test, y_test, y_pred, le)
    save_plots(model, X_test, metrics)
    save_artefacts(model, le, list(X.columns))

    print("\n[ train ] All done.")


if __name__ == "__main__":
    main()
