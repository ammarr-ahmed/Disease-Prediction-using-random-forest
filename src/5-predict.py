"""
predict.py
----------
Step 5: Load saved model artefacts, build an input vector from the
        user-selected symptoms, and print the top-N disease predictions
        with confidence percentages.

Requires:
  • models/model.joblib
  • models/encoder.joblib
  • models/columns.joblib
  • data/input_vector.csv   ← produced by selector.py
"""

import os
import sys
import csv
import joblib
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH   = os.path.join(MODEL_DIR, "model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.joblib")
INPUT_PATH   = os.path.join(BASE_DIR, "..", "data", "input_vector.csv")

# ── Config ───────────────────────────────────────────────────────────────────
TOP_N = 5       # how many top diseases to display


# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_name(name: str) -> str:
    """Match the column normalisation used during training."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def check_files() -> None:
    missing = [p for p in [MODEL_PATH, ENCODER_PATH, COLUMNS_PATH] if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing model artefacts:")
        for p in missing:
            print(f"        {p}")
        print("        Run train.py first.")
        sys.exit(1)

    if not os.path.exists(INPUT_PATH):
        print("[ERROR] input_vector.csv not found.")
        print("        Run selector.py first.")
        sys.exit(1)


def load_selected_symptoms(path: str) -> set[str]:
    selected: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                selected.add(clean_name(row[0]))
    return selected


# ── Main ─────────────────────────────────────────────────────────────────────
def predict() -> None:
    check_files()

    # ── Load artefacts ────────────────────────────────────────────────────────
    model   = joblib.load(MODEL_PATH)
    le      = joblib.load(ENCODER_PATH)
    columns = joblib.load(COLUMNS_PATH)   # list[str] — same order as training

    # ── Load user input ───────────────────────────────────────────────────────
    selected = load_selected_symptoms(INPUT_PATH)

    if not selected:
        print("[ERROR] input_vector.csv is empty. Select symptoms in selector.py.")
        sys.exit(1)

    # ── Build binary feature vector ───────────────────────────────────────────
    input_row = np.array(
        [1 if col in selected else 0 for col in columns],
        dtype="int8",
    ).reshape(1, -1)

    matched = sum(input_row[0])
    print(f"\n[ predict ] {matched} / {len(columns)} symptom(s) recognised")

    if matched == 0:
        print("[WARNING] None of the selected symptoms matched training columns.")
        print("          Check that selector.py and the dataset are in sync.")
        sys.exit(1)

    # ── Predict ───────────────────────────────────────────────────────────────
    probs      = model.predict_proba(input_row)[0]
    top_n_idx  = np.argsort(probs)[::-1][:TOP_N]
    top_n_prob = probs[top_n_idx]
    top_n_name = le.inverse_transform(top_n_idx)

    # ── Display ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("  PREDICTION RESULTS")
    print("=" * 45)

    for rank, (disease, confidence) in enumerate(zip(top_n_name, top_n_prob), start=1):
        bar_len   = int(confidence * 30)
        bar       = "█" * bar_len + "░" * (30 - bar_len)
        prefix    = "▶" if rank == 1 else " "
        print(f"  {prefix} #{rank}  {disease}")
        print(f"       [{bar}]  {confidence * 100:.1f}%")
        if rank < TOP_N:
            print()

    print("=" * 45)
    print(f"\n  Most likely: {top_n_name[0]}  ({top_n_prob[0] * 100:.1f}% confidence)")

    low_confidence = top_n_prob[0] < 0.40
    if low_confidence:
        print("\n  ⚠  Confidence is below 40 %. Consider selecting more symptoms.")

    print("\n  ⚠  This tool is for educational purposes only.")
    print("     Always consult a qualified medical professional.\n")


if __name__ == "__main__":
    predict()
