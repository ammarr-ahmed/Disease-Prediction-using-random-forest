"""
preprocess.py
-------------
Step 2: Read the cleaned dataset, group symptom columns by body system,
        and write symptom_groups.csv for the selector.
"""

import os
import csv
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")   # src/ → project root → data/
CLEAN_PATH  = os.path.join(DATA_DIR, "disease_dataset_clean.csv")
GROUPS_PATH = os.path.join(DATA_DIR, "symptom_groups.csv")

# ── Group keyword map (order matters — first match wins) ─────────────────────
GROUP_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Head & Neurological", ["head", "eye", "vision", "migraine", "dizzy",
                             "blur", "memory", "concentrat", "unconscious",
                             "convuls", "altered"]),
    ("Respiratory",         ["cough", "breath", "lung", "throat", "nose",
                             "cold", "phlegm", "mucus", "wheez", "sputum",
                             "chest_tight", "runny"]),
    ("Digestive",           ["stomach", "vomit", "nausea", "diarrhea",
                             "abdomen", "bowel", "indigest", "constipat",
                             "bloat", "acidity", "ulcer", "gastro"]),
    ("Skin",                ["skin", "rash", "itch", "acne", "redness",
                             "blister", "peel", "discharg", "sweat",
                             "yellowing", "pale", "bruising"]),
    ("Pain & Musculoskeletal", ["pain", "joint", "muscle", "back", "neck",
                                "cramp", "swell", "stiff", "tender",
                                "inflammation", "ache"]),
    ("General / Systemic",  ["fever", "chill", "fatigue", "weak", "weight",
                             "appetite", "lethargy", "malaise", "dehydr",
                             "fluid", "urin", "thirst"]),
    ("Cardiovascular",      ["heart", "palpitat", "pulse", "blood pressure",
                             "bp", "varicose", "chest_pain"]),
    ("Mental / Mood",       ["anxiety", "depress", "mood", "irritab",
                             "restless", "stress", "sleep", "insomnia"]),
]
OTHER_GROUP = "Other"


def assign_group(symptom: str) -> str:
    for group_name, keywords in GROUP_KEYWORDS:
        if any(kw in symptom for kw in keywords):
            return group_name
    return OTHER_GROUP


def build_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {group_name: [symptom, …]} from dataframe columns."""
    target_candidates = [c for c in df.columns if "disease" in c or "prognosis" in c]
    target_col = target_candidates[0] if target_candidates else df.columns[-1]

    symptom_cols = [c for c in df.columns if c != target_col]

    # Human-readable display names (spaces instead of underscores)
    groups: dict[str, list[str]] = {}
    for col in symptom_cols:
        display = col.replace("_", " ").strip()
        group   = assign_group(col)
        groups.setdefault(group, [])
        if display not in groups[group]:        # dedup display names
            groups[group].append(display)

    # Sort symptoms within each group
    for g in groups:
        groups[g].sort()

    return groups


def save_groups(groups: dict[str, list[str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for group, symptoms in groups.items():
            if symptoms:
                writer.writerow([group] + symptoms)
    print(f"[ preprocess ] Saved → {path}  ({len(groups)} groups)")


if __name__ == "__main__":
    print("[ preprocess ] Loading cleaned dataset …")
    df = pd.read_csv(CLEAN_PATH)
    groups = build_groups(df)

    total = sum(len(v) for v in groups.values())
    print(f"[ preprocess ] {total} symptoms → {len(groups)} groups")
    for name, syms in groups.items():
        print(f"    {name:30s} {len(syms):3d} symptoms")

    save_groups(groups, GROUPS_PATH)
    print("[ preprocess ] Done.")
