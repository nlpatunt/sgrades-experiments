#!/usr/bin/env python3
"""
Stability Analysis for S-GRADES 3-call Predictions
====================================================
For each model folder and each strategy subfolder:
  - Finds all *_FULL.csv files
  - Per essay: computes std across prediction_1/2/3 (numeric datasets)
               computes agreement rate across prediction_1/2/3 (categorical datasets)
  - Per dataset: reports mean std (or mean agreement rate)
  - Pooled overall: pools ALL essays across ALL datasets → single overall mean std / agreement rate
  - Saves per-strategy outputs INSIDE each strategy subfolder:
      essay_level/  ← one CSV per dataset with essay-level stats
      stability_summary_<strategy>_<ts>.xlsx
      stability_summary_<strategy>_<ts>.json

Categorical datasets: BEEtlE_2way, BEEtlE_3way, SciEntSBank_2way, SciEntSBank_3way
All others: numeric (regression)

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
# CONFIG — edit these paths if needed
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

PRED_COLS = ["prediction_1", "prediction_2", "prediction_3"]

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
    """
    Extract dataset name from filenames like:
      gpt-4o-mini_D_ASAP_plus_plus_3call_FULL.csv
      llama-4-scout_D_BEEtlE_2way_3call_FULL.csv
      gemini-2.5-flash_D_SciEntSBank_3way_3call_FULL.csv
    Returns e.g. D_ASAP_plus_plus
    """
    name = filename.replace(".csv", "")
    m = re.search(r"_(D_[A-Za-z0-9_]+?)_3call", name)
    if m:
        return m.group(1)
    return None


def is_categorical(dataset_name):
    clean = dataset_name.replace("D_", "")
    return clean in {d.replace("D_", "") for d in CATEGORICAL_DATASETS}


def per_essay_std(row):
    """Std (ddof=1) across 3 numeric predictions for one essay."""
    vals = pd.to_numeric(row[PRED_COLS], errors="coerce").dropna()
    if len(vals) < 2:
        return np.nan
    return float(vals.std(ddof=1))


def per_essay_agreement(row):
    """
    Agreement rate for one essay (categorical).
    Fraction of 3 calls matching the majority label.
    e.g. [correct, correct, incorrect] -> 2/3 ~ 0.667
    """
    vals = [v for v in row[PRED_COLS]
            if pd.notna(v) and str(v).strip() not in ("", "nan", "none")]
    if len(vals) == 0:
        return np.nan
    counts = Counter(vals)
    majority_count = counts.most_common(1)[0][1]
    return majority_count / len(vals)


def majority_label(row):
    """Return the majority label across 3 categorical predictions."""
    vals = [v for v in row[PRED_COLS]
            if pd.notna(v) and str(v).strip() not in ("", "nan", "none")]
    if len(vals) == 0:
        return "unknown"
    return Counter(vals).most_common(1)[0][0]


# ============================================================================
# ANALYZE ONE _FULL.csv
# Returns (summary_dict, essay_level_df) or (None, None) on failure.
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
        # Normalise to lowercase strings
        for c in PRED_COLS:
            df[c] = df[c].astype(str).str.strip().str.lower()
            df[c] = df[c].replace({"nan": np.nan, "none": np.nan, "": np.nan})

        df["agreement_rate"] = df.apply(per_essay_agreement, axis=1)
        df["majority_label"] = df.apply(majority_label, axis=1)
        # is_unstable: any disagreement among the 3 calls
        df["is_unstable"]    = df["agreement_rate"] < 1.0

        valid = df["agreement_rate"].dropna()
        mean_agreement    = float(valid.mean())
        perfect_pct       = float((valid == 1.0).sum()              / len(valid) * 100) if len(valid) > 0 else np.nan
        partial_pct       = float(((valid >= 2/3) & (valid < 1.0)).sum() / len(valid) * 100) if len(valid) > 0 else np.nan
        no_agree_pct      = float((valid < 2/3).sum()               / len(valid) * 100) if len(valid) > 0 else np.nan

        summary = {
            "dataset":           dataset_name,
            "type":              "categorical",
            "n_essays":          n,
            "n_valid":           int(len(valid)),
            "mean_agreement":    mean_agreement,
            "perfect_agree_pct": perfect_pct,
            "partial_agree_pct": partial_pct,
            "no_agree_pct":      no_agree_pct,
            # private keys for pooling (stripped before saving)
            "_essay_agreements": valid.tolist(),
            # numeric fields -> None
            "mean_std":          None,
            "median_std":        None,
            "max_std":           None,
            "pct_std_gt1":       None,
        }

    else:
        # Numeric
        for c in PRED_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["essay_std"]   = df.apply(per_essay_std, axis=1)
        df["essay_mean"]  = df[PRED_COLS].mean(axis=1)
        # is_high_var: std > 1.0 flags notable instability
        df["is_high_var"] = df["essay_std"] > 1.0

        valid = df["essay_std"].dropna()
        if len(valid) == 0:
            print(f"    x No valid numeric predictions in {path.name}")
            return None, None

        mean_std   = float(valid.mean())
        median_std = float(valid.median())
        max_std    = float(valid.max())
        pct_gt1    = float((valid > 1.0).sum() / len(valid) * 100)

        summary = {
            "dataset":           dataset_name,
            "type":              "numeric",
            "n_essays":          n,
            "n_valid":           int(len(valid)),
            "mean_std":          mean_std,
            "median_std":        median_std,
            "max_std":           max_std,
            "pct_std_gt1":       pct_gt1,
            # private keys for pooling
            "_essay_stds":       valid.tolist(),
            # categorical fields -> None
            "mean_agreement":    None,
            "perfect_agree_pct": None,
            "partial_agree_pct": None,
            "no_agree_pct":      None,
        }

    return summary, df.copy()


# ============================================================================
# SAVE ESSAY-LEVEL CSVs  (one per dataset, inside essay_level/ subfolder)
# ============================================================================

def save_essay_level_csvs(results_with_dfs, strategy_path):
    """
    results_with_dfs: list of (summary_dict, essay_df)
    Saves to:  strategy_path/essay_level/essay_stability_<dataset>.csv
    Columns saved:
      Numeric    : all original cols + essay_std, essay_mean, is_high_var
      Categorical: all original cols + agreement_rate, majority_label, is_unstable
    """
    essay_level_dir = strategy_path / "essay_level"
    essay_level_dir.mkdir(exist_ok=True)

    for summary, essay_df in results_with_dfs:
        if essay_df is None:
            continue
        dname = summary["dataset"]
        clean = dname.replace("D_", "")
        out_path = essay_level_dir / f"essay_stability_{clean}.csv"
        essay_df.to_csv(out_path, index=False)
        print(f"      essay CSV -> {out_path.name}  ({len(essay_df)} rows)")


# ============================================================================
# BUILD SUMMARY DATAFRAMES
# ============================================================================

def build_summary_df(dataset_results):
    """
    dataset_results: list of summary dicts
    Returns (per_dataset_df, overall_df, combined_df)
    Pooled overall is computed by collecting every individual essay value,
    NOT by averaging dataset-level means.
    """
    rows             = []
    all_numeric_stds = []
    all_cat_agree    = []

    for r in dataset_results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        rows.append(row)
        if r["type"] == "numeric" and "_essay_stds" in r:
            all_numeric_stds.extend(r["_essay_stds"])
        if r["type"] == "categorical" and "_essay_agreements" in r:
            all_cat_agree.extend(r["_essay_agreements"])

    per_dataset_df = pd.DataFrame(rows)

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
            "perfect_agree_pct": float((arr == 1.0).sum()               / len(arr) * 100),
            "partial_agree_pct": float(((arr >= 2/3) & (arr < 1.0)).sum() / len(arr) * 100),
            "no_agree_pct":      float((arr < 2/3).sum()                / len(arr) * 100),
        })

    overall_df  = pd.DataFrame(overall_rows)
    combined_df = pd.concat([per_dataset_df, overall_df], ignore_index=True)

    return per_dataset_df, overall_df, combined_df


# ============================================================================
# SAVE SUMMARY EXCEL + JSON
# ============================================================================

COL_ORDER = [
    "dataset", "type", "n_essays", "n_valid",
    "mean_std", "median_std", "max_std", "pct_std_gt1",
    "mean_agreement", "perfect_agree_pct", "partial_agree_pct", "no_agree_pct",
]


def save_excel(combined_df, out_path):
    cols = [c for c in COL_ORDER if c in combined_df.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        combined_df[cols].to_excel(writer, sheet_name="Per-Dataset + Overall", index=False)

        num = combined_df[combined_df["type"] == "numeric"][cols]
        if not num.empty:
            num.to_excel(writer, sheet_name="Numeric Only", index=False)

        cat = combined_df[combined_df["type"] == "categorical"][cols]
        if not cat.empty:
            cat.to_excel(writer, sheet_name="Categorical Only", index=False)

    print(f"    -> Excel: {out_path.name}")


def save_json(dataset_results, combined_df, overall_df, out_path):
    overall = []
    for _, row in overall_df.iterrows():
        overall.append({
            k: (None if (isinstance(v, float) and np.isnan(v)) else v)
            for k, v in row.items()
        })

    payload = {
        "generated_at": datetime.now().isoformat(),
        "note": (
            "Pooled overall = all individual essay stds/agreements pooled together, "
            "NOT an average of dataset-level means. "
            "Per-dataset means are also reported for dataset-level variability."
        ),
        "per_dataset": [
            {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
             for k, v in r.items() if not k.startswith("_")}
            for r in dataset_results
        ],
        "pooled_overall": overall,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"    -> JSON:  {out_path.name}")


# ============================================================================
# CORE: process one strategy subfolder  (fully isolated, no cross-strategy mixing)
# ============================================================================

def process_strategy_folder(strategy_path):
    """
    Analyze all *_FULL.csv files in strategy_path.
    Saves:
      strategy_path/essay_level/essay_stability_<dataset>.csv  (one per dataset)
      strategy_path/stability_summary_<strategy>_<ts>.xlsx
      strategy_path/stability_summary_<strategy>_<ts>.json
    """
    full_files = sorted(strategy_path.glob("*_FULL.csv"))
    if not full_files:
        print(f"  [!] No _FULL.csv files found in {strategy_path.name}")
        return []

    print(f"\n  -- {strategy_path.name} ({len(full_files)} FULL files)")

    results_with_dfs = []  # list of (summary_dict, essay_df)

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

    # 1. Save essay-level CSVs
    save_essay_level_csvs(results_with_dfs, strategy_path)

    # 2. Build summary DataFrames
    dataset_results = [s for s, _ in results_with_dfs]
    per_dataset_df, overall_df, combined_df = build_summary_df(dataset_results)

    # 3. Save Excel + JSON
    ts = int(datetime.now().timestamp())
    excel_path = strategy_path / f"stability_summary_{strategy_path.name}_{ts}.xlsx"
    json_path  = strategy_path / f"stability_summary_{strategy_path.name}_{ts}.json"

    save_excel(combined_df, excel_path)
    save_json(dataset_results, combined_df, overall_df, json_path)

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
    print("                per-dataset -> mean agreement, perfect/partial/no agreement %")
    print("                pooled overall -> mean agreement pooled across ALL essays")
    print()
    print("Output structure per strategy subfolder:")
    print("  essay_level/")
    print("    essay_stability_<dataset>.csv   <- all original cols + std/agreement per essay")
    print("  stability_summary_<strategy>_<ts>.xlsx  <- per-dataset + pooled overall")
    print("  stability_summary_<strategy>_<ts>.json  <- same in JSON")
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