#!/usr/bin/env python3
"""
Stability Analysis for S-GRADES 3-call Predictions
====================================================
For each model folder and each strategy subfolder:
  - Finds all *_FULL.csv files
  - Per essay: std(ddof=1) across prediction_1/2/3 (numeric)
               agreement rate across prediction_1/2/3 (categorical)
  - Per dataset: mean_std / mean_agreement
  - Sub-question pooling (correct weighted pooling, NOT averaging of per-question means):
      OS_Dataset : q1-q5 essay stds pooled -> one combined row
      Rice_Chem  : Q1-Q4 essay stds pooled -> one combined row
  - Original sub-question rows kept in Appendix sheet only
  - Combined rows used in Main sheet + pooled overall (no double counting)

Saves per strategy subfolder:
  essay_level/essay_stability_<dataset>.csv
  stability_summary_<strategy>_<ts>.xlsx  (Main / Appendix / Numeric Only / Categorical Only)
  stability_summary_<strategy>_<ts>.json

Usage:
    python compute_stability.py
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================
MODEL_FOLDERS = [
    "/home/ts1506.UNT/Desktop/Work/sgrades-experiments/gemini_flash",
    "/home/ts1506.UNT/Desktop/Work/sgrades-experiments/gpt4_mini",
    "/home/ts1506.UNT/Desktop/Work/sgrades-experiments/generalization/lama_exp",
]

STRATEGY_SUBFOLDERS = [
    "abductive_3call_predictions",
    "deductive_3call_predictions",
    "inductive_3call_predictions",
    "deductive_abductive_3call_predictions",
    "inductive_deductive_3call_predictions",
    "inductive_abductive_3call_predictions",
]

CATEGORICAL_DATASETS = {
    "BEEtlE_2way", "BEEtlE_3way",
    "SciEntSBank_2way", "SciEntSBank_3way",
    "D_BEEtlE_2way", "D_BEEtlE_3way",
    "D_SciEntSBank_2way", "D_SciEntSBank_3way",
}

# Sub-question groups to pool into one combined row
# key = combined name, value = list of dataset name substrings to match
POOL_GROUPS = {
    "D_OS_Dataset": ["D_OS_Dataset_q1", "D_OS_Dataset_q2", "D_OS_Dataset_q3",
                     "D_OS_Dataset_q4", "D_OS_Dataset_q5"],
    "D_Rice_Chem":  ["D_Rice_Chem_Q1", "D_Rice_Chem_Q2",
                     "D_Rice_Chem_Q3", "D_Rice_Chem_Q4"],
}

PRED_COLS = ["prediction_1", "prediction_2", "prediction_3"]

COL_ORDER = [
    "dataset", "type", "n_essays", "n_valid",
    "mean_std", "median_std", "max_std", "pct_std_gt1",
    "mean_agreement", "perfect_agree_pct", "partial_agree_pct", "no_agree_pct",
]

# ============================================================================
# HELPERS
# ============================================================================

def load_csv(path):
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", engine="python")
        except Exception:
            continue
    return None


def extract_dataset_name(filename):
    name = filename.replace(".csv", "")
    if "_D_" not in name or "_3call_FULL" not in name:
        return None

    start = name.index("_D_") + 1
    end = name.index("_3call_FULL")
    return name[start:end]


def is_categorical(dataset_name):
    clean = dataset_name.replace("D_", "")
    return clean in {d.replace("D_", "") for d in CATEGORICAL_DATASETS}


def per_essay_std(row):
    vals = pd.to_numeric(row[PRED_COLS], errors="coerce").dropna()
    if len(vals) < 2:
        return np.nan
    return float(vals.std(ddof=1))


def per_essay_agreement(row):
    vals = [v for v in row[PRED_COLS]
            if pd.notna(v) and str(v).strip() not in ("", "nan", "none")]
    if len(vals) == 0:
        return np.nan
    return Counter(vals).most_common(1)[0][1] / len(vals)


def majority_label(row):
    vals = [v for v in row[PRED_COLS]
            if pd.notna(v) and str(v).strip() not in ("", "nan", "none")]
    if len(vals) == 0:
        return "unknown"
    return Counter(vals).most_common(1)[0][0]


# ============================================================================
# ANALYZE ONE _FULL.csv
# Returns (summary_dict, essay_df) or (None, None)
# ============================================================================

def analyze_full_csv(path, dataset_name):
    df = load_csv(path)
    if df is None:
        print(f"    x Could not load: {path.name}")
        return None, None

    missing = [c for c in PRED_COLS if c not in df.columns]
    if missing:
        print(f"    x Missing columns {missing} in {path.name}")
        return None, None

    n = len(df)
    categorical = is_categorical(dataset_name)

    if categorical:
        for c in PRED_COLS:
            df[c] = df[c].astype(str).str.strip().str.lower()
            df[c] = df[c].replace({"nan": np.nan, "none": np.nan, "": np.nan})

        df["agreement_rate"] = df.apply(per_essay_agreement, axis=1)
        df["majority_label"] = df.apply(majority_label, axis=1)
        df["is_unstable"]    = df["agreement_rate"] < 1.0

        valid = df["agreement_rate"].dropna()
        mean_agreement    = float(valid.mean())
        perfect_pct       = float((valid == 1.0).sum()                    / len(valid) * 100) if len(valid) > 0 else np.nan
        partial_pct       = float(((valid >= 2/3) & (valid < 1.0)).sum() / len(valid) * 100) if len(valid) > 0 else np.nan
        no_agree_pct      = float((valid < 2/3).sum()                    / len(valid) * 100) if len(valid) > 0 else np.nan

        summary = {
            "dataset":           dataset_name,
            "type":              "categorical",
            "n_essays":          n,
            "n_valid":           int(len(valid)),
            "mean_agreement":    mean_agreement,
            "perfect_agree_pct": perfect_pct,
            "partial_agree_pct": partial_pct,
            "no_agree_pct":      no_agree_pct,
            "_essay_agreements": valid.tolist(),
            "mean_std":          None,
            "median_std":        None,
            "max_std":           None,
            "pct_std_gt1":       None,
        }

    else:
        for c in PRED_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["essay_std"]  = df.apply(per_essay_std, axis=1)
        df["essay_mean"] = df[PRED_COLS].mean(axis=1)
        df["is_high_var"] = df["essay_std"] > 1.0

        valid = df["essay_std"].dropna()
        if len(valid) == 0:
            print(f"    x No valid numeric predictions in {path.name}")
            return None, None

        summary = {
            "dataset":           dataset_name,
            "type":              "numeric",
            "n_essays":          n,
            "n_valid":           int(len(valid)),
            "mean_std":          float(valid.mean()),
            "median_std":        float(valid.median()),
            "max_std":           float(valid.max()),
            "pct_std_gt1":       float((valid > 1.0).sum() / len(valid) * 100),
            "_essay_stds":       valid.tolist(),
            "mean_agreement":    None,
            "perfect_agree_pct": None,
            "partial_agree_pct": None,
            "no_agree_pct":      None,
        }

    return summary, df.copy()


# ============================================================================
# POOL SUB-QUESTION GROUPS  (OS_Dataset q1-q5, Rice_Chem Q1-Q4)
# ============================================================================

def build_combined_rows(dataset_results):
    """
    For each group in POOL_GROUPS, collect all essay stds/agreements from
    the matching sub-question summaries and compute one pooled combined row.

    Returns:
      combined_rows   : list of summary dicts for combined datasets
      subquestion_names: set of dataset names that were pooled (to exclude from Main)
    """
    combined_rows    = []
    subquestion_names = set()

    for combined_name, members in POOL_GROUPS.items():
        matching = [r for r in dataset_results if r["dataset"] in members]
        if not matching:
            continue

        # Mark all matched sub-questions
        for r in matching:
            subquestion_names.add(r["dataset"])

        # All must be numeric (OS_Dataset and Rice_Chem are both numeric)
        all_stds = []
        for r in matching:
            if "_essay_stds" in r:
                all_stds.extend(r["_essay_stds"])

        if not all_stds:
            continue

        arr = np.array(all_stds)
        combined_rows.append({
            "dataset":           combined_name,
            "type":              "numeric",
            "n_essays":          len(arr),
            "n_valid":           len(arr),
            "mean_std":          float(arr.mean()),
            "median_std":        float(np.median(arr)),
            "max_std":           float(arr.max()),
            "pct_std_gt1":       float((arr > 1.0).sum() / len(arr) * 100),
            "_essay_stds":       all_stds,   # keep for pooled overall
            "mean_agreement":    None,
            "perfect_agree_pct": None,
            "partial_agree_pct": None,
            "no_agree_pct":      None,
        })

        n_sub = len(matching)
        print(f"    [pool] {combined_name:30s} <- {n_sub} sub-questions, "
              f"n={len(arr)} essays, mean_std={arr.mean():.4f}")

    return combined_rows, subquestion_names


# ============================================================================
# BUILD SUMMARY DATAFRAMES
# ============================================================================

def build_summary_df(dataset_results):
    """
    dataset_results : all summary dicts including sub-questions
    Returns (main_df, appendix_df, overall_df, combined_df_for_excel)

    main_df     : combined rows replace sub-questions (use for heatmap/plots)
    appendix_df : all original sub-question rows (for reference)
    overall_df  : pooled overall rows (numeric + categorical), no double counting
    """
    # --- build combined rows and identify sub-questions ---
    combined_rows, subquestion_names = build_combined_rows(dataset_results)

    # --- main rows: non-subquestion originals + combined rows ---
    main_results = [r for r in dataset_results if r["dataset"] not in subquestion_names]
    main_results = main_results + combined_rows

    # --- appendix rows: only the sub-question originals ---
    appendix_results = [r for r in dataset_results if r["dataset"] in subquestion_names]

    def to_df(results):
        rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    main_df     = to_df(main_results)
    appendix_df = to_df(appendix_results)

    # --- pooled overall: use main_results (no sub-questions) to avoid double counting ---
    all_numeric_stds = []
    all_cat_agree    = []
    for r in main_results:
        if r["type"] == "numeric" and "_essay_stds" in r:
            all_numeric_stds.extend(r["_essay_stds"])
        if r["type"] == "categorical" and "_essay_agreements" in r:
            all_cat_agree.extend(r["_essay_agreements"])

    overall_rows = []

    if all_numeric_stds:
        arr = np.array(all_numeric_stds)
        overall_rows.append({
            "dataset":           "POOLED OVERALL (numeric)",
            "type":              "numeric",
            "n_essays":          len(arr),
            "n_valid":           len(arr),
            "mean_std":          float(arr.mean()),
            "median_std":        float(np.median(arr)),
            "max_std":           float(arr.max()),
            "pct_std_gt1":       float((arr > 1.0).sum() / len(arr) * 100),
            "mean_agreement":    None,
            "perfect_agree_pct": None,
            "partial_agree_pct": None,
            "no_agree_pct":      None,
        })

    if all_cat_agree:
        arr = np.array(all_cat_agree)
        overall_rows.append({
            "dataset":           "POOLED OVERALL (categorical)",
            "type":              "categorical",
            "n_essays":          len(arr),
            "n_valid":           len(arr),
            "mean_std":          None,
            "median_std":        None,
            "max_std":           None,
            "pct_std_gt1":       None,
            "mean_agreement":    float(arr.mean()),
            "perfect_agree_pct": float((arr == 1.0).sum()                    / len(arr) * 100),
            "partial_agree_pct": float(((arr >= 2/3) & (arr < 1.0)).sum()   / len(arr) * 100),
            "no_agree_pct":      float((arr < 2/3).sum()                    / len(arr) * 100),
        })

    overall_df = pd.DataFrame(overall_rows)

    # combined_df for excel: main + overall stacked
    combined_df = pd.concat([main_df, overall_df], ignore_index=True)

    return main_df, appendix_df, overall_df, combined_df, main_results


# ============================================================================
# SAVE ESSAY-LEVEL CSVs
# ============================================================================

def save_essay_level_csvs(results_with_dfs, strategy_path):
    essay_level_dir = strategy_path / "essay_level"
    essay_level_dir.mkdir(exist_ok=True)
    for summary, essay_df in results_with_dfs:
        if essay_df is None:
            continue
        clean = summary["dataset"].replace("D_", "")
        out_path = essay_level_dir / f"essay_stability_{clean}.csv"
        essay_df.to_csv(out_path, index=False)
        print(f"      essay CSV -> {out_path.name}  ({len(essay_df)} rows)")


# ============================================================================
# SAVE EXCEL
# ============================================================================

def save_excel(combined_df, main_df, appendix_df, out_path):
    cols = [c for c in COL_ORDER if c in combined_df.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Main sheet: combined rows (OS_Dataset, Rice_Chem) + pooled overall
        combined_df[cols].to_excel(writer, sheet_name="Main", index=False)

        # Appendix sheet: original sub-question rows
        if not appendix_df.empty:
            app_cols = [c for c in COL_ORDER if c in appendix_df.columns]
            appendix_df[app_cols].to_excel(writer, sheet_name="Appendix (sub-questions)", index=False)

        # Numeric-only view (from main)
        num = combined_df[combined_df["type"] == "numeric"][cols]
        if not num.empty:
            num.to_excel(writer, sheet_name="Numeric Only", index=False)

        # Categorical-only view (from main)
        cat = combined_df[combined_df["type"] == "categorical"][cols]
        if not cat.empty:
            cat.to_excel(writer, sheet_name="Categorical Only", index=False)

    print(f"    -> Excel: {out_path.name}")


# ============================================================================
# SAVE JSON
# ============================================================================

def save_json(main_results, overall_df, out_path):
    def clean(v):
        return None if (isinstance(v, float) and np.isnan(v)) else v

    overall = [{k: clean(v) for k, v in row.items()} for _, row in overall_df.iterrows()]

    payload = {
        "generated_at": datetime.now().isoformat(),
        "note": (
            "OS_Dataset and Rice_Chem rows are pooled from sub-questions "
            "(correct weighted pooling, not average of per-question means). "
            "Pooled overall excludes sub-question rows to avoid double counting."
        ),
        "per_dataset_main": [
            {k: clean(v) for k, v in r.items() if not k.startswith("_")}
            for r in main_results
        ],
        "pooled_overall": overall,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"    -> JSON:  {out_path.name}")


# ============================================================================
# CORE: process one strategy subfolder
# ============================================================================

def process_strategy_folder(strategy_path):
    full_files = sorted(strategy_path.glob("*_FULL.csv"))
    if not full_files:
        print(f"  [!] No _FULL.csv files found in {strategy_path.name}")
        return []

    print(f"\n  -- {strategy_path.name} ({len(full_files)} FULL files)")

    results_with_dfs = []

    for f in full_files:
        dname = extract_dataset_name(f.name)
        if dname is None:
            print(f"    [!] Could not parse dataset name from: {f.name}, skipping")
            continue

        summary, essay_df = analyze_full_csv(f, dname)

        if summary is not None:
            metric_str = ""
            if summary["mean_std"] is not None:
                metric_str = f" | mean_std={summary['mean_std']:.4f}"
            elif summary["mean_agreement"] is not None:
                metric_str = f" | mean_agr={summary['mean_agreement']:.4f}"
            print(f"    OK {dname:40s} | {summary['type']:11s} | n={summary['n_valid']}{metric_str}")
            results_with_dfs.append((summary, essay_df))

    if not results_with_dfs:
        return []

    # 1. Save essay-level CSVs (all sub-questions kept individually)
    save_essay_level_csvs(results_with_dfs, strategy_path)

    # 2. Build summary DataFrames (with sub-question pooling)
    dataset_results = [s for s, _ in results_with_dfs]
    main_df, appendix_df, overall_df, combined_df, main_results = build_summary_df(dataset_results)

    # 3. Save Excel + JSON
    ts = int(datetime.now().timestamp())
    excel_path = strategy_path / f"stability_summary_{strategy_path.name}_{ts}.xlsx"
    json_path  = strategy_path / f"stability_summary_{strategy_path.name}_{ts}.json"

    save_excel(combined_df, main_df, appendix_df, excel_path)
    save_json(main_results, overall_df, json_path)

    return dataset_results


# ============================================================================
# CORE: process one model folder
# ============================================================================

def process_model_folder(model_path):
    print(f"\n{'='*70}")
    print(f"MODEL FOLDER: {model_path}")
    print(f"{'='*70}")

    found_any = False
    for strategy_name in STRATEGY_SUBFOLDERS:
        strategy_path = model_path / strategy_name
        if not strategy_path.exists():
            print(f"  [!] Not found: {strategy_path.name}, skipping")
            continue
        process_strategy_folder(strategy_path)
        found_any = True

    if not found_any:
        print(f"  [x] No strategy subfolders found under {model_path.name}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    print("\n" + "="*70)
    print("S-GRADES STABILITY ANALYSIS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    print("Methodology:")
    print("  Numeric     : per-essay std(ddof=1) across 3 calls")
    print("                per-dataset -> mean_std, median_std, max_std, pct_std>1")
    print("                pooled overall -> mean std pooled across ALL essays")
    print("  Categorical : per-essay agreement rate (fraction matching majority label)")
    print("                per-dataset -> mean agreement, perfect/partial/no %")
    print("                pooled overall -> mean agreement pooled across ALL essays")
    print()
    print("Sub-question pooling (weighted, not averaged):")
    print("  OS_Dataset  : q1+q2+q3+q4+q5 essay stds pooled -> one D_OS_Dataset row")
    print("  Rice_Chem   : Q1+Q2+Q3+Q4 essay stds pooled    -> one D_Rice_Chem row")
    print("  Sub-question rows preserved in Appendix sheet of Excel")
    print()
    print("Output structure per strategy subfolder:")
    print("  essay_level/essay_stability_<dataset>.csv")
    print("  stability_summary_<strategy>_<ts>.xlsx  (Main / Appendix / Numeric / Categorical)")
    print("  stability_summary_<strategy>_<ts>.json")
    print()

    missing = [p for p in MODEL_FOLDERS if not os.path.isdir(p)]
    if missing:
        print("[!] These model folders were NOT found (will be skipped):")
        for p in missing:
            print(f"    {p}")
        print()

    for model_folder in MODEL_FOLDERS:
        if os.path.isdir(model_folder):
            process_model_folder(Path(model_folder))
        else:
            print(f"\n[x] Skipping (not found): {model_folder}")

    print(f"\n{'='*70}")
    print(f"DONE -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()