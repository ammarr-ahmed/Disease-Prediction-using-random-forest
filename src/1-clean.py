"""
clean.py
--------
Step 1: Load the raw dataset, clean column names and values,
        drop duplicates/nulls, and save a cleaned version.
"""

import os
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
RAW_PATH    = os.path.join(DATA_DIR, "disease_dataset.csv")
CLEAN_PATH  = os.path.join(DATA_DIR, "disease_dataset_clean.csv")


def clean_col(name: str) -> str:
    """Normalise a column name to snake_case, lowercase."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def clean_dataset(raw_path: str, clean_path: str) -> pd.DataFrame:
    print("[ clean ] Loading raw dataset …")
    df = pd.read_csv(raw_path)

    # ── Drop unnamed / all-null columns ──────────────────────────────────────
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    df.dropna(axis=1, how="all", inplace=True)

    # ── Normalise column names ────────────────────────────────────────────────
    df.columns = [clean_col(c) for c in df.columns]

    # ── Identify the target column ────────────────────────────────────────────
    target_candidates = [c for c in df.columns if "disease" in c or "prognosis" in c]
    if not target_candidates:
        raise ValueError("Could not find a disease / prognosis column.")
    target_col = target_candidates[0]

    # ── Clean feature columns: coerce to int, fill missing with 0 ────────────
    feature_cols = [c for c in df.columns if c != target_col]
    df[feature_cols] = (
        df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("int8")
    )

    # ── Clean target: strip whitespace ───────────────────────────────────────
    df[target_col] = df[target_col].astype(str).str.strip()

    # ── Drop duplicate rows ───────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[ clean ] Dropped {dropped} duplicate rows.")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    df.to_csv(clean_path, index=False)

    print(f"[ clean ] Done → {len(df)} rows, {len(df.columns)} columns")
    print(f"[ clean ] Saved → {clean_path}")
    return df


if __name__ == "__main__":
    clean_dataset(RAW_PATH, CLEAN_PATH)
