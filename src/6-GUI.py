"""
gui.py
------
Disease Prediction GUI — built with CustomTkinter.

Install dependency first:
    pip install customtkinter

Run:
    python src/gui.py

Requirements:
    • data/symptom_groups.csv   (run preprocess.py)
    • models/model.joblib       (run train.py)
    • models/encoder.joblib
    • models/columns.joblib
"""

import os
import csv
import sys
import threading
import tkinter as tk
from tkinter import messagebox

try:
    import customtkinter as ctk
except ImportError:
    print("[ERROR] customtkinter not installed.")
    print("        Run:  pip install customtkinter")
    sys.exit(1)

try:
    import numpy as np
    import joblib
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("        Run:  pip install numpy joblib scikit-learn")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR    = os.path.join(BASE_DIR, "..", "models")
GROUPS_PATH  = os.path.join(DATA_DIR,  "symptom_groups.csv")
MODEL_PATH   = os.path.join(MODEL_DIR, "model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.joblib")
COLUMNS_PATH = os.path.join(MODEL_DIR, "columns.joblib")

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Colour palette
BG_DEEP    = "#0D1117"   # darkest background
BG_CARD    = "#161B22"   # card / panel background
BG_SURFACE = "#21262D"   # elevated surface
ACCENT     = "#2F81F7"   # blue accent
ACCENT2    = "#3FB950"   # green (good result)
ACCENT3    = "#F78166"   # red/orange (warning)
TEXT_PRI   = "#E6EDF3"   # primary text
TEXT_SEC   = "#8B949E"   # secondary / muted text
BORDER     = "#30363D"   # subtle border


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def load_groups() -> tuple[dict, dict]:
    group_names: dict[int, str]         = {}
    group_symptoms: dict[int, list[str]] = {}
    if not os.path.exists(GROUPS_PATH):
        return group_names, group_symptoms
    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.reader(f), start=1):
            if not row:
                continue
            group_names[idx]    = row[0].strip()
            group_symptoms[idx] = [s.strip() for s in row[1:] if s.strip()]
    return group_names, group_symptoms


def load_model():
    model   = joblib.load(MODEL_PATH)
    le      = joblib.load(ENCODER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, le, columns


def predict(selected: set[str], model, le, columns) -> list[tuple[str, float]]:
    import pandas as pd
    input_row = pd.DataFrame(
        [[1 if col in selected else 0 for col in columns]],
        columns=columns,
    ).astype("int8")
    probs    = model.predict_proba(input_row)[0]
    top5_idx = np.argsort(probs)[::-1][:5]
    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top5_idx]


# ── Main App ──────────────────────────────────────────────────────────────────
class DiseaseApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Disease Predictor")
        self.geometry("1200x750")
        self.minsize(1000, 650)
        self.configure(fg_color=BG_DEEP)

        # State
        self.group_names: dict    = {}
        self.group_symptoms: dict = {}
        self.selected: set[str]   = set()
        self.model                = None
        self.le                   = None
        self.columns              = None
        self.active_group: int    = 1
        self.checkboxes: list     = []   # (CTkCheckBox, symptom_name)

        self._load_data()
        self._build_ui()
        self._load_group(self.active_group)

    # ── Data loading ──────────────────────────────────────────────────────────
    def _load_data(self):
        self.group_names, self.group_symptoms = load_groups()

        if not self.group_names:
            messagebox.showerror(
                "Missing File",
                f"symptom_groups.csv not found.\n\nRun preprocess.py first.\n\n{GROUPS_PATH}"
            )

        models_exist = all(os.path.exists(p) for p in [MODEL_PATH, ENCODER_PATH, COLUMNS_PATH])
        if models_exist:
            try:
                self.model, self.le, self.columns = load_model()
            except Exception as e:
                messagebox.showwarning("Model Load Failed", str(e))
        else:
            messagebox.showwarning(
                "Model Not Found",
                "No trained model found.\n\nRun train.py first, then restart the app."
            )

    # ── UI Construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Top header bar ────────────────────────────────────────────────────
        header = ctk.CTkFrame(self, fg_color=BG_CARD, height=60, corner_radius=0)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="⚕  Disease Predictor",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color=TEXT_PRI,
        ).pack(side="left", padx=24, pady=14)

        # Selected count badge
        self.badge_var = tk.StringVar(value="0 symptoms selected")
        ctk.CTkLabel(
            header,
            textvariable=self.badge_var,
            font=ctk.CTkFont(size=13),
            text_color=TEXT_SEC,
        ).pack(side="right", padx=24)

        # ── Main body ─────────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=BG_DEEP)
        body.pack(fill="both", expand=True, padx=0, pady=0)
        body.columnconfigure(0, weight=0)   # sidebar
        body.columnconfigure(1, weight=1)   # symptom panel
        body.columnconfigure(2, weight=0)   # results
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_symptom_panel(body)
        self._build_results_panel(body)

    def _build_sidebar(self, parent):
        sidebar = ctk.CTkFrame(
            parent, fg_color=BG_CARD, width=210,
            corner_radius=0,
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="SYMPTOM GROUPS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_SEC,
        ).pack(pady=(20, 8), padx=16, anchor="w")

        self.group_btn_frames = {}
        for idx, name in self.group_names.items():
            btn = ctk.CTkButton(
                sidebar,
                text=name,
                font=ctk.CTkFont(size=13),
                anchor="w",
                fg_color="transparent",
                text_color=TEXT_SEC,
                hover_color=BG_SURFACE,
                height=38,
                corner_radius=6,
                command=lambda i=idx: self._load_group(i),
            )
            btn.pack(fill="x", padx=10, pady=2)
            self.group_btn_frames[idx] = btn

        # Divider + Clear button at bottom
        ctk.CTkFrame(sidebar, fg_color=BORDER, height=1).pack(
            fill="x", padx=16, pady=(16, 8)
        )
        ctk.CTkButton(
            sidebar,
            text="✕  Clear All",
            font=ctk.CTkFont(size=13),
            fg_color="transparent",
            text_color=ACCENT3,
            hover_color=BG_SURFACE,
            height=36,
            corner_radius=6,
            command=self._clear_all,
        ).pack(fill="x", padx=10, pady=2)

    def _build_symptom_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_DEEP)
        panel.grid(row=0, column=1, sticky="nsew", padx=1)

        # Group title
        self.group_title = ctk.CTkLabel(
            panel,
            text="",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRI,
            anchor="w",
        )
        self.group_title.pack(fill="x", padx=24, pady=(20, 4))

        self.group_subtitle = ctk.CTkLabel(
            panel,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SEC,
            anchor="w",
        )
        self.group_subtitle.pack(fill="x", padx=24, pady=(0, 12))

        # Search box
        search_frame = ctk.CTkFrame(panel, fg_color=BG_SURFACE, corner_radius=8, height=36)
        search_frame.pack(fill="x", padx=24, pady=(0, 12))
        search_frame.pack_propagate(False)

        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._filter_symptoms())

        ctk.CTkLabel(
            search_frame, text="🔍", font=ctk.CTkFont(size=13),
            text_color=TEXT_SEC, width=30,
        ).pack(side="left", padx=(10, 0))

        ctk.CTkEntry(
            search_frame,
            textvariable=self.search_var,
            placeholder_text="Search symptoms ...",
            font=ctk.CTkFont(size=13),
            border_width=0,
            fg_color="transparent",
            text_color=TEXT_PRI,
        ).pack(side="left", fill="both", expand=True, padx=4)

        # Scrollable checkbox area
        self.checkbox_scroll = ctk.CTkScrollableFrame(
            panel,
            fg_color=BG_CARD,
            corner_radius=10,
            scrollbar_button_color=BG_SURFACE,
        )
        self.checkbox_scroll.pack(fill="both", expand=True, padx=24, pady=(0, 16))

        # Predict button at bottom
        ctk.CTkButton(
            panel,
            text="  Run Prediction  →",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT,
            hover_color="#1A6ED8",
            height=44,
            corner_radius=10,
            command=self._run_prediction,
        ).pack(fill="x", padx=24, pady=(0, 20))

    def _build_results_panel(self, parent):
        panel = ctk.CTkFrame(
            parent, fg_color=BG_CARD, width=320, corner_radius=0,
        )
        panel.grid(row=0, column=2, sticky="nsew")
        panel.pack_propagate(False)

        ctk.CTkLabel(
            panel,
            text="PREDICTION RESULTS",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_SEC,
        ).pack(pady=(20, 16), padx=20, anchor="w")

        self.result_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.result_frame.pack(fill="both", expand=True, padx=16)

        self._show_placeholder()

        # Disclaimer
        ctk.CTkFrame(panel, fg_color=BORDER, height=1).pack(
            fill="x", padx=16, pady=(8, 8)
        )
        ctk.CTkLabel(
            panel,
            text="⚠  For educational purposes only.\nAlways consult a doctor.",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SEC,
            justify="left",
        ).pack(padx=20, pady=(0, 16), anchor="w")

    # ── Group loading ─────────────────────────────────────────────────────────
    def _load_group(self, idx: int):
        # Highlight active button
        for i, btn in self.group_btn_frames.items():
            if i == idx:
                btn.configure(fg_color=BG_SURFACE, text_color=TEXT_PRI)
            else:
                btn.configure(fg_color="transparent", text_color=TEXT_SEC)

        self.active_group = idx
        name     = self.group_names.get(idx, "")
        symptoms = self.group_symptoms.get(idx, [])

        self.group_title.configure(text=name)
        self.search_var.set("")
        self._render_checkboxes(symptoms)

        selected_in_group = sum(1 for s in symptoms if s in self.selected)
        self.group_subtitle.configure(
            text=f"{len(symptoms)} symptoms   •   {selected_in_group} selected in this group"
        )

    def _render_checkboxes(self, symptoms: list[str]):
        # Clear old
        for widget in self.checkbox_scroll.winfo_children():
            widget.destroy()
        self.checkboxes = []

        for sym in symptoms:
            var = tk.BooleanVar(value=(sym in self.selected))
            cb  = ctk.CTkCheckBox(
                self.checkbox_scroll,
                text=sym.title(),
                variable=var,
                font=ctk.CTkFont(size=13),
                text_color=TEXT_PRI,
                fg_color=ACCENT,
                hover_color="#1A6ED8",
                checkmark_color="white",
                corner_radius=4,
                command=lambda s=sym, v=var: self._toggle(s, v),
            )
            cb.pack(anchor="w", padx=16, pady=5)
            self.checkboxes.append((cb, sym, var))

    def _filter_symptoms(self):
        query    = self.search_var.get().lower().strip()
        symptoms = self.group_symptoms.get(self.active_group, [])
        filtered = [s for s in symptoms if query in s.lower()] if query else symptoms
        self._render_checkboxes(filtered)

    def _toggle(self, symptom: str, var: tk.BooleanVar):
        if var.get():
            self.selected.add(symptom)
        else:
            self.selected.discard(symptom)
        self._update_badge()
        self._update_subtitle()

    def _update_badge(self):
        n = len(self.selected)
        self.badge_var.set(f"{n} symptom{'s' if n != 1 else ''} selected")

    def _update_subtitle(self):
        symptoms = self.group_symptoms.get(self.active_group, [])
        selected_in_group = sum(1 for s in symptoms if s in self.selected)
        self.group_subtitle.configure(
            text=f"{len(symptoms)} symptoms   •   {selected_in_group} selected in this group"
        )

    def _clear_all(self):
        self.selected.clear()
        self._load_group(self.active_group)
        self._update_badge()
        self._show_placeholder()

    # ── Prediction ────────────────────────────────────────────────────────────
    def _run_prediction(self):
        if not self.model:
            messagebox.showerror("No Model", "Train the model first (run train.py).")
            return
        if not self.selected:
            messagebox.showwarning("No Symptoms", "Please select at least one symptom.")
            return

        # Clean selected names to match column format
        cleaned = {clean_name(s) for s in self.selected}
        results = predict(cleaned, self.model, self.le, self.columns)
        self._show_results(results)

    def _show_placeholder(self):
        for w in self.result_frame.winfo_children():
            w.destroy()
        ctk.CTkLabel(
            self.result_frame,
            text="Select symptoms\nthen click\n'Run Prediction'",
            font=ctk.CTkFont(size=14),
            text_color=TEXT_SEC,
            justify="center",
        ).pack(expand=True, pady=60)

    def _show_results(self, results: list[tuple[str, float]]):
        for w in self.result_frame.winfo_children():
            w.destroy()

        for rank, (disease, confidence) in enumerate(results, start=1):
            is_top = rank == 1

            card = ctk.CTkFrame(
                self.result_frame,
                fg_color=BG_SURFACE if is_top else "transparent",
                corner_radius=10,
            )
            card.pack(fill="x", pady=(0, 10))

            # Rank + disease name
            top_frame = ctk.CTkFrame(card, fg_color="transparent")
            top_frame.pack(fill="x", padx=14, pady=(12, 4))

            rank_color = ACCENT2 if is_top else TEXT_SEC
            ctk.CTkLabel(
                top_frame,
                text=f"#{rank}",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=rank_color,
                width=28,
            ).pack(side="left")

            ctk.CTkLabel(
                top_frame,
                text=disease,
                font=ctk.CTkFont(size=13, weight="bold" if is_top else "normal"),
                text_color=TEXT_PRI,
                wraplength=220,
                justify="left",
            ).pack(side="left", padx=(4, 0))

            # Confidence bar
            bar_frame = ctk.CTkFrame(card, fg_color="transparent")
            bar_frame.pack(fill="x", padx=14, pady=(2, 12))

            pct = confidence * 100
            bar_color = ACCENT2 if pct >= 60 else ACCENT if pct >= 30 else ACCENT3

            bar_bg = ctk.CTkFrame(bar_frame, fg_color=BG_DEEP, height=8, corner_radius=4)
            bar_bg.pack(fill="x", side="left", expand=True, padx=(0, 10))
            bar_bg.pack_propagate(False)

            fill_w = max(4, int(confidence * 200))
            bar_fill = ctk.CTkFrame(bar_bg, fg_color=bar_color, height=8, corner_radius=4, width=fill_w)
            bar_fill.place(x=0, y=0, relheight=1)

            ctk.CTkLabel(
                bar_frame,
                text=f"{pct:.1f}%",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=bar_color,
                width=48,
            ).pack(side="left")

        # Low confidence warning
        if results and results[0][1] < 0.40:
            warn = ctk.CTkFrame(self.result_frame, fg_color="#2D1B00", corner_radius=8)
            warn.pack(fill="x", pady=(4, 0))
            ctk.CTkLabel(
                warn,
                text="⚠  Low confidence.\nTry selecting more symptoms.",
                font=ctk.CTkFont(size=11),
                text_color="#F0A500",
                justify="left",
            ).pack(padx=12, pady=10)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = DiseaseApp()
    app.mainloop()