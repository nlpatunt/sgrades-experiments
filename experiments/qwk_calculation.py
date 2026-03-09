"""
compute_aggregated_qwk.py

Computes QWK (numeric) and F1 (categorical) using mean prediction
from 3-call stability CSVs, matched against HuggingFace ground truth by ID.

Run:
    python compute_aggregated_qwk.py
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from huggingface_hub import login
from datasets import load_dataset

login(token=os.getenv("HF_TOKEN", ""))

# ── PATHS ─────────────────────────────────────────────────────────────────────
MODEL_DIRS = {
    "GPT-4o-mini":      "/home/ts1506.UNT/Desktop/Work/sgrades-experiments/gpt4_mini",
    "Gemini-2.5-Flash": "/home/ts1506.UNT/Desktop/Work/sgrades-experiments/gemini_flash",
    "LLaMA-4-Scout":    "/home/ts1506.UNT/Desktop/Work/sgrades-experiments/generalization/lama_exp",
}

STRATEGY_FOLDERS = [
    "abductive_3call_predictions",
    "deductive_3call_predictions",
    "inductive_3call_predictions",
    "deductive_abductive_3call_predictions",
    "inductive_deductive_3call_predictions",
    "inductive_abductive_3call_predictions",
]

STRATEGY_LABELS = {
    "abductive_3call_predictions":           "Abductive",
    "deductive_3call_predictions":           "Deductive",
    "inductive_3call_predictions":           "Inductive",
    "deductive_abductive_3call_predictions": "Ded-Abd",
    "inductive_deductive_3call_predictions": "Ind-Ded",
    "inductive_abductive_3call_predictions": "Ind-Abd",
}

# ── DATASET CONFIG ─────────────────────────────────────────────────────────────
# (hf_id, load_kwargs, id_col, score_col, type, csv_suffix)
DATASETS = {
    "ASAP-AES":              ("nlpatunt/ASAP-AES",       {},                                        "essay_id",      "domain1_score",        "numeric",     "D_ASAP-AES"),
    "ASAP2":                 ("nlpatunt/ASAP2",          {},                                        "essay_id",      "score",                "numeric",     "D_ASAP2"),
    "ASAP-SAS":              ("nlpatunt/ASAP-SAS",       {"data_files": {"test": "test.csv"}},      "Id",            "Score1",               "numeric",     "D_ASAP-SAS"),
    "ASAP_plus_plus":        ("nlpatunt/ASAP_plus_plus", {},                                        "essay_id",      "overall_score",        "numeric",     "D_ASAP_plus_plus"),
    "CSEE":                  ("nlpatunt/CSEE",           {},                                        "index",         "overall_score",        "numeric",     "D_CSEE"),
    "persuade_2":            ("nlpatunt/persuade_2",     {},                                        "essay_id_comp", "holistic_essay_score", "numeric",     "D_persuade_2"),
    "Ielts_Writing_Dataset": ("nlpatunt/Ielts_Writing_Dataset",      {},                            "ID",            "Overall_Score",        "numeric",     "D_Ielts_Writing_Dataset"),
    "Ielts_Writing_Task_2":  ("nlpatunt/Ielts_Writing_Task_2_Dataset", {},                          "ID",            "band_score",           "numeric",     "D_Ielts_Writing_Task_2_Dataset"),
    "Mohlar":                ("nlpatunt/Mohlar",         {"data_files": {"test": "test.csv"}},      "ID",            "grade",                "numeric",     "D_Mohlar"),
    "Regrading_Dataset_J2C": ("nlpatunt/Regrading_Dataset_J2C", {},                                "ID",            "grade",                "numeric",     "D_Regrading_Dataset_J2C"),
    "BEEtlE_2way":           ("nlpatunt/BEEtlE",        {"name": "2way"},                          "ID",            "label",                "categorical", "D_BEEtlE_2way"),
    "BEEtlE_3way":           ("nlpatunt/BEEtlE",        {"name": "3way"},                          "ID",            "label",                "categorical", "D_BEEtlE_3way"),
    "SciEntSBank_2way":      ("nlpatunt/SciEntSBank",   {"name": "2way"},                          "ID",            "label",                "categorical", "D_SciEntSBank_2way"),
    "SciEntSBank_3way":      ("nlpatunt/SciEntSBank",   {"name": "3way"},                          "ID",            "label",                "categorical", "D_SciEntSBank_3way"),
    "Rice_Chem_Q1":          ("nlpatunt/Rice_Chem",     {"data_files": {"test": "Q1/test.csv"}},   "sis_id",        "Score",                "numeric",     "D_Rice_Chem_Q1"),
    "Rice_Chem_Q2":          ("nlpatunt/Rice_Chem",     {"data_files": {"test": "Q2/test.csv"}},   "sis_id",        "Score",                "numeric",     "D_Rice_Chem_Q2"),
    "Rice_Chem_Q3":          ("nlpatunt/Rice_Chem",     {"data_files": {"test": "Q3/test.csv"}},   "sis_id",        "Score",                "numeric",     "D_Rice_Chem_Q3"),
    "Rice_Chem_Q4":          ("nlpatunt/Rice_Chem",     {"data_files": {"test": "Q4/test.csv"}},   "sis_id",        "Score",                "numeric",     "D_Rice_Chem_Q4"),
    "OS_Dataset_q1":         ("nlpatunt/OS_Dataset",    {"data_files": {"test": "q1/test.csv"}},   "ID",            "score_1",              "numeric",     "D_OS_Dataset_q1"),
    "OS_Dataset_q2":         ("nlpatunt/OS_Dataset",    {"data_files": {"test": "q2/test.csv"}},   "ID",            "score_1",              "numeric",     "D_OS_Dataset_q2"),
    "OS_Dataset_q3":         ("nlpatunt/OS_Dataset",    {"data_files": {"test": "q3/test.csv"}},   "ID",            "score_1",              "numeric",     "D_OS_Dataset_q3"),
    "OS_Dataset_q4":         ("nlpatunt/OS_Dataset",    {"data_files": {"test": "q4/test.csv"}},   "ID",            "score_1",              "numeric",     "D_OS_Dataset_q4"),
    "OS_Dataset_q5":         ("nlpatunt/OS_Dataset",    {"data_files": {"test": "q5/test.csv"}},   "ID",            "score_1",              "numeric",     "D_OS_Dataset_q5"),
}

SUBQ_GROUPS = {
    "Rice_Chem": ["Rice_Chem_Q1", "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"],
    "OS_Dataset": ["OS_Dataset_q1", "OS_Dataset_q2", "OS_Dataset_q3", "OS_Dataset_q4", "OS_Dataset_q5"],
}

# ── LOAD GROUND TRUTH ─────────────────────────────────────────────────────────
print("Loading ground truth from HuggingFace...")
gt_cache = {}
for ds_name, (hf_id, load_kwargs, id_col, score_col, dtype, _) in DATASETS.items():
    try:
        ds = load_dataset(hf_id, split="test", trust_remote_code=True, **load_kwargs)
        df = ds.to_pandas()[[id_col, score_col]]
        df[id_col] = df[id_col].astype(str)
        gt_cache[ds_name] = df
        print(f"  ✅ {ds_name}: {len(df)} rows")
    except Exception as e:
        print(f"  ❌ {ds_name}: {e}")

# ── METRIC FUNCTIONS ──────────────────────────────────────────────────────────
def compute_qwk(y_true, y_pred):
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    mask   = ~(y_true.isna() | y_pred.isna())
    if mask.sum() == 0:
        return np.nan
    y_true = np.round(y_true[mask]).astype(int)
    y_pred = np.round(y_pred[mask]).astype(int)
    try:
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")
    except Exception:
        return np.nan

def compute_f1(y_true, y_pred):
    try:
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    except Exception:
        return np.nan

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
results = []

for model_name, model_dir in MODEL_DIRS.items():
    for strat_folder in STRATEGY_FOLDERS:
        strat_label = STRATEGY_LABELS[strat_folder]
        strat_path  = os.path.join(model_dir, strat_folder)

        if not os.path.isdir(strat_path):
            print(f"  ⚠️  Missing dir: {strat_path}")
            continue

        for ds_name, (hf_id, load_kwargs, id_col, score_col, dtype, csv_suffix) in DATASETS.items():
            if ds_name not in gt_cache:
                continue

            matches = glob.glob(os.path.join(strat_path, f"*{csv_suffix}_3call.csv"))
            if not matches:
                continue
            csv_path = matches[0]

            try:
                pred_df = pd.read_csv(csv_path)
                pred_df[id_col] = pred_df[id_col].astype(str)

                gt_df  = gt_cache[ds_name]
                merged = pred_df.merge(gt_df, on=id_col, how="inner", suffixes=("_pred", "_gt"))

                if len(merged) == 0:
                    print(f"  ⚠️  No ID matches: {model_name}/{strat_label}/{ds_name}")
                    continue

                gt_col   = f"{score_col}_gt"
                pred_col = f"{score_col}_pred"

                if pred_col not in merged.columns:
                    print(f"  ⚠️  Missing pred column '{pred_col}': {list(merged.columns)}")
                    continue

                if dtype == "numeric":
                    metric      = compute_qwk(merged[gt_col], merged[pred_col])
                    metric_name = "QWK"
                else:
                    metric      = compute_f1(merged[gt_col], merged[pred_col])
                    metric_name = "F1"

                results.append({
                    "model":    model_name,
                    "strategy": strat_label,
                    "dataset":  ds_name,
                    "metric":   metric_name,
                    "value":    round(metric, 4),
                    "n":        len(merged),
                })

            except Exception as e:
                print(f"  ❌ {model_name}/{strat_label}/{ds_name}: {e}")

# ── AGGREGATE SUB-QUESTIONS ───────────────────────────────────────────────────
results_df = pd.DataFrame(results)

extras = []
for group_name, subqs in SUBQ_GROUPS.items():
    sub = results_df[results_df["dataset"].isin(subqs)]
    if sub.empty:
        continue
    grouped = sub.groupby(["model", "strategy", "metric"])["value"].mean().reset_index()
    grouped["dataset"] = group_name
    extras.append(grouped)

if extras:
    results_df = pd.concat([results_df, pd.concat(extras)], ignore_index=True)
    all_subqs  = [q for qs in SUBQ_GROUPS.values() for q in qs]
    results_df = results_df[~results_df["dataset"].isin(all_subqs)]

# ── SAVE ──────────────────────────────────────────────────────────────────────
results_df.to_excel("aggregated_qwk_results.xlsx", index=False)
print("\n✅ Saved: aggregated_qwk_results.xlsx")

for model_name in results_df["model"].unique():
    model_df  = results_df[results_df["model"] == model_name]
    safe_name = model_name.replace(" ", "_").replace("/", "_")
    fname     = f"aggregated_qwk_{safe_name}.xlsx"
    model_df.to_excel(fname, index=False)
    print(f"✅ Saved: {fname}")

print(results_df.groupby(["model", "metric"])["value"].describe().round(3))