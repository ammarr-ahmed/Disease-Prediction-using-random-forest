"""
selector.py
-----------
Step 3: Interactive CLI to browse symptom groups and build the
        input_vector.csv that the prediction step reads.

Controls
--------
  Main menu   →  enter group number, or 0 to finish
  Group view  →  enter numbers to toggle (e.g. "1 3 5" or "2,4")
                 "c"  clear all selections in this group
                 "b"  go back to main menu
"""

import os
import csv
import sys

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
GROUPS_PATH = os.path.join(DATA_DIR, "symptom_groups.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "input_vector.csv")


# ── Helpers ──────────────────────────────────────────────────────────────────
def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def load_groups(path: str) -> tuple[dict[int, str], dict[int, list[str]]]:
    group_names: dict[int, str]        = {}
    group_symptoms: dict[int, list[str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            group_names[idx]    = row[0].strip()
            group_symptoms[idx] = [s.strip() for s in row[1:] if s.strip()]

    return group_names, group_symptoms


def save_vector(selected: set[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sym in sorted(selected):
            writer.writerow([sym, 1])


def parse_numbers(text: str, max_idx: int) -> list[int]:
    """Parse '1,3 5' style input into a list of valid 1-based indices."""
    tokens = text.replace(",", " ").split()
    result = []
    for t in tokens:
        if t.isdigit():
            n = int(t)
            if 1 <= n <= max_idx:
                result.append(n)
    return result


# ── Views ────────────────────────────────────────────────────────────────────
def show_main_menu(
    group_names: dict[int, str],
    group_symptoms: dict[int, list[str]],
    selected: set[str],
) -> None:
    total_selected = len(selected)
    print("=" * 50)
    print("  SYMPTOM SELECTOR")
    print(f"  {total_selected} symptom(s) selected so far")
    print("=" * 50)
    for i, name in group_names.items():
        count_in_group = sum(1 for s in group_symptoms[i] if s in selected)
        tag = f"  [{count_in_group} selected]" if count_in_group else ""
        print(f"  {i:>2}. {name}{tag}")
    print("   0. ✓  Done — run prediction")
    print("-" * 50)


def show_group(
    name: str,
    symptoms: list[str],
    selected: set[str],
) -> None:
    print("=" * 50)
    print(f"  {name.upper()}")
    print("=" * 50)
    for i, sym in enumerate(symptoms, start=1):
        marker = "●" if sym in selected else "○"
        print(f"  {i:>2}. {marker} {sym}")
    print("-" * 50)
    print("  Enter numbers to toggle  |  c = clear group  |  b = back")
    print("-" * 50)


# ── Main loop ────────────────────────────────────────────────────────────────
def run() -> None:
    if not os.path.exists(GROUPS_PATH):
        print(f"[ERROR] Groups file not found: {GROUPS_PATH}")
        print("        Run preprocess.py first.")
        sys.exit(1)

    group_names, group_symptoms = load_groups(GROUPS_PATH)
    selected: set[str] = set()

    # Pre-load any existing selections
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 1:
                    selected.add(row[0].strip())
        if selected:
            print(f"  Loaded {len(selected)} previously selected symptom(s).")

    while True:
        clear()
        show_main_menu(group_names, group_symptoms, selected)
        raw = input("  Select: ").strip()

        if not raw.isdigit():
            continue

        choice = int(raw)

        if choice == 0:
            break

        if choice not in group_symptoms:
            input("  Invalid choice. Press Enter …")
            continue

        # ── Group sub-loop ────────────────────────────────────────────────────
        while True:
            clear()
            symptoms = group_symptoms[choice]
            show_group(group_names[choice], symptoms, selected)
            cmd = input("  > ").strip().lower()

            if cmd == "b":
                break

            if cmd == "c":
                removed = sum(1 for s in symptoms if s in selected)
                for s in symptoms:
                    selected.discard(s)
                print(f"  Cleared {removed} symptom(s).")
                input("  Press Enter …")
                continue

            indices = parse_numbers(cmd, len(symptoms))
            if not indices:
                input("  No valid numbers found. Press Enter …")
                continue

            for idx in indices:
                sym = symptoms[idx - 1]
                if sym in selected:
                    selected.discard(sym)
                else:
                    selected.add(sym)

            save_vector(selected, OUTPUT_PATH)
            # Brief flash of confirmation
            toggled = [symptoms[i - 1] for i in indices]
            print(f"  Toggled: {', '.join(toggled)}")
            input("  Press Enter …")

    # ── Summary ───────────────────────────────────────────────────────────────
    clear()
    if not selected:
        print("  No symptoms selected — nothing saved.")
        sys.exit(0)

    save_vector(selected, OUTPUT_PATH)
    print("=" * 50)
    print(f"  {len(selected)} symptom(s) saved to input_vector.csv")
    print("=" * 50)
    for s in sorted(selected):
        print(f"    • {s}")
    print()
    print("  Run train.py (first time) or predict.py to get a prediction.")


if __name__ == "__main__":
    run()
