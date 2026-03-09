

#!/usr/bin/env python3
"""
Inductive reasoning evaluation on ASAP-AES only.
- Train + test from nlpatunt/D_ASAP-AES
- Ground truth metrics from nlpatunt/ASAP-AES test split
- 5 examples sampled GLOBALLY per seed (same examples for all essay sets)
- Seeds 42, 123, 456 control exemplar sampling only
- temperature=0
- Metrics: QWK per essay_set (rounded/clipped) -> macro average
           Pearson per essay_set (raw) -> macro average
           MAE per essay_set (raw) -> macro average
"""

import os
import sys
import json
import time
import re
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score
from huggingface_hub import login
from datasets import load_dataset

HF_TOKEN = "REMOVED_KEY"
API_KEY  =  "REMOVED_KEY"

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not set.")
if not API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not set.")

MODEL_CODE   = "openai/gpt-4o-mini"
MODEL_NAME   = "gpt-4o-mini"
SEEDS        = [42, 123, 456]
NUM_EXAMPLES = 5
BASE_OUT     = os.path.expanduser(
    "~/Desktop/Work/sgrades-experiments/new_run_2/asap_aes_inductive_gpt4omini_seeds"
)

SCORE_RANGES = {
    1: (2, 12), 2: (1, 6),  3: (0, 3),  4: (0, 3),
    5: (0, 4),  6: (0, 4),  7: (0, 30), 8: (0, 60),
}

# ============================================================================
# LOAD DATA
# ============================================================================
def load_asap_aes():
    login(token=HF_TOKEN)

    print("Loading train and test from nlpatunt/D_ASAP-AES...")
    ds_d     = load_dataset("nlpatunt/D_ASAP-AES", trust_remote_code=True)
    train_df = ds_d["train"].to_pandas()
    test_df  = ds_d["test"].to_pandas()
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    print("Loading ground truth from nlpatunt/ASAP-AES test split...")
    ds_gt = load_dataset("nlpatunt/ASAP-AES", trust_remote_code=True)
    gt_df = ds_gt["test"].to_pandas()
    print(f"  Ground truth: {len(gt_df)} rows")

    return train_df, test_df, gt_df

# ============================================================================
# SAMPLE EXAMPLES GLOBALLY
# 5 examples from full train set, same for all essay sets within a seed
# ============================================================================
def sample_examples_global(train_df: pd.DataFrame, seed: int) -> list:
    subset = train_df.dropna(subset=["domain1_score"])
    if len(subset) < NUM_EXAMPLES:
        raise ValueError(f"Training set too small: {len(subset)} < {NUM_EXAMPLES}")
    sampled  = subset.sample(n=NUM_EXAMPLES, random_state=seed)
    examples = [
        {
            "text":      str(row["essay"]),
            "score":     str(int(row["domain1_score"])),
            "essay_set": int(row["essay_set"]),
            "min_s":     SCORE_RANGES[int(row["essay_set"])][0],
            "max_s":     SCORE_RANGES[int(row["essay_set"])][1],
        }
        for _, row in sampled.iterrows()
    ]
    print(f"  Sampled {NUM_EXAMPLES} global examples (IDs: {sampled['essay_id'].tolist()})")
    return examples

# ============================================================================
# PROMPT
# ============================================================================
def build_prompt(essay_text: str, essay_set: int, examples: list) -> dict:
    min_s, max_s = SCORE_RANGES[essay_set]

    examples_str = ""
    for i, ex in enumerate(examples, 1):
        ex_min = ex["min_s"]
        ex_max = ex["max_s"]
        ex_set = ex["essay_set"]
        examples_str += f"\nEXAMPLE {i} (Essay Set {ex_set}, score range {ex_min}–{ex_max}):\n"
        examples_str += f"Essay: {ex['text']}\n"
        examples_str += f"Score: {ex['score']} (out of {ex_min}–{ex_max})\n"

    system_prompt = f"""You are an expert essay scorer using INDUCTIVE REASONING.

INDUCTIVE PROCESS:
1. Study the labeled examples below carefully — note each example shows its essay set and score range
2. Identify patterns that distinguish high scores from low scores
3. Apply those patterns to score the new essay

SCORED REFERENCE EXAMPLES (from various essay sets — each labeled with its own range):
{examples_str}

From these examples, learn:
- What content, quality, and completeness earns high scores relative to the range
- What weaknesses or gaps lead to lower scores
- The expected level of development for a given prompt"""

    user_prompt = f"""You must score the following essay.

TARGET ESSAY SET: {essay_set}
TARGET SCORE RANGE: {min_s} to {max_s}

ESSAY:
{essay_text}

Apply the patterns you learned from the reference examples above.
Scale your judgment to the target score range ({min_s} to {max_s}).

Your ENTIRE response must be EXACTLY one number between {min_s} and {max_s}.
Nothing else. Just the number."""

    return {"system": system_prompt, "user": user_prompt}

# ============================================================================
# VALIDATE
# ============================================================================
def validate_prediction(text: str, essay_set: int) -> dict:
    min_s, max_s = SCORE_RANGES[essay_set]
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text.strip())
    if not numbers:
        return {"valid": False, "extracted": None, "error": f"No number found: {text[:50]}"}
    score = float(numbers[0])
    if min_s <= score <= max_s:
        return {"valid": True, "extracted": score, "error": None}
    return {"valid": False, "extracted": score, "error": f"Out of range [{min_s},{max_s}]: {score}"}

# ============================================================================
# API CALL
# ============================================================================
def call_api(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_CODE,
                messages=messages,
                max_tokens=10,
                temperature=0,
                extra_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title":      "S-GRADES Inductive ASAP-AES"
                }
            )
            return {"success": True, "response": response}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
            time.sleep(2 ** attempt)


def get_prediction(client, essay_text, essay_set, examples, max_retries=5):
    text = None
    for attempt in range(max_retries):
        prompt   = build_prompt(essay_text, essay_set, examples)
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user",   "content": prompt["user"]}
        ]
        result = call_api(client, messages)
        if not result["success"]:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {"success": False, "error": result["error"]}

        text       = result["response"].choices[0].message.content.strip()
        validation = validate_prediction(text, essay_set)
        if validation["valid"]:
            return {"success": True, "prediction": validation["extracted"],
                    "raw": text, "attempts": attempt + 1}
        if attempt < max_retries - 1:
            time.sleep(1)

    return {"success": False,
            "error": f"Validation failed after {max_retries} attempts",
            "last_raw": text or ""}

# ============================================================================
# METRICS
# ============================================================================
def safe_macro(per_set_metrics: dict, metric: str) -> float:
    vals = [v[metric] for v in per_set_metrics.values() if not np.isnan(v[metric])]
    return round(float(np.mean(vals)), 4) if vals else float("nan")


def compute_metrics(predictions: dict, gt_df: pd.DataFrame) -> dict:
    pred_rows = [{"essay_id": int(k), "pred": float(v)} for k, v in predictions.items()]
    pred_df   = pd.DataFrame(pred_rows)
    merged    = pd.merge(
        gt_df[["essay_id", "essay_set", "domain1_score"]],
        pred_df, on="essay_id"
    ).dropna()

    # ------------------------------------------------------------------
    # PER-SET metrics -> macro average
    # ------------------------------------------------------------------
    per_set_metrics = {}
    for essay_set, group in merged.groupby("essay_set"):
        essay_set    = int(essay_set)
        min_s, max_s = SCORE_RANGES.get(essay_set, (0, 100))

        y_true     = group["domain1_score"].astype(float).values
        y_pred_raw = group["pred"].astype(float).values

        y_pred_qwk = np.clip(np.round(y_pred_raw), min_s, max_s).astype(int)
        y_true_i   = y_true.astype(int)

        try:
            qwk = float(cohen_kappa_score(y_true_i, y_pred_qwk, weights="quadratic"))
        except Exception:
            qwk = float("nan")

        try:
            pearson, _ = pearsonr(y_true, y_pred_raw)
            pearson    = float(pearson)
        except Exception:
            pearson = float("nan")

        mae = float(np.mean(np.abs(y_true - y_pred_raw)))

        per_set_metrics[essay_set] = {
            "QWK":     round(qwk,     4),
            "Pearson": round(pearson, 4),
            "MAE":     round(mae,     4),
            "n":       len(group),
        }
        print(f"    Set {essay_set}: QWK={qwk:.4f} Pearson={pearson:.4f} MAE={mae:.4f} n={len(group)}")

    macro = {
        "QWK":     safe_macro(per_set_metrics, "QWK"),
        "Pearson": safe_macro(per_set_metrics, "Pearson"),
        "MAE":     safe_macro(per_set_metrics, "MAE"),
        "n_sets":  len(per_set_metrics),
        "n_total": sum(v["n"] for v in per_set_metrics.values()),
    }

    # ------------------------------------------------------------------
    # GLOBAL metrics (all essay sets together, matches main results table)
    # ------------------------------------------------------------------
    y_true_all = merged["domain1_score"].astype(float).values
    y_pred_all = merged["pred"].astype(float).values

    # For global QWK: clip each prediction to its own essay_set range
    y_pred_qwk_all = np.array([
        int(np.clip(np.round(pred), SCORE_RANGES[int(es)][0], SCORE_RANGES[int(es)][1]))
        for pred, es in zip(merged["pred"].values, merged["essay_set"].values)
    ])
    y_true_int_all = y_true_all.astype(int)

    try:
        global_qwk = round(float(cohen_kappa_score(y_true_int_all, y_pred_qwk_all, weights="quadratic")), 4)
    except Exception:
        global_qwk = float("nan")

    try:
        global_pearson, _ = pearsonr(y_true_all, y_pred_all)
        global_pearson = round(float(global_pearson), 4)
    except Exception:
        global_pearson = float("nan")

    global_mae = round(float(np.mean(np.abs(y_true_all - y_pred_all))), 4)

    global_metrics = {
        "QWK":     global_qwk,
        "Pearson": global_pearson,
        "MAE":     global_mae,
        "n_total": len(merged),
    }
    print(f"    GLOBAL: QWK={global_qwk:.4f} Pearson={global_pearson:.4f} MAE={global_mae:.4f} n={len(merged)}")

    return {"per_set": per_set_metrics, "macro": macro, "global": global_metrics}

# ============================================================================
# RUN ONE SEED
# ============================================================================
def run_seed(seed: int, train_df: pd.DataFrame, test_df: pd.DataFrame,
             gt_df: pd.DataFrame, client: OpenAI):
    out_dir         = os.path.join(BASE_OUT, f"seed{seed}")
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")
    failed_path     = os.path.join(out_dir, "failed_ids.json")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"SEED {seed} — INDUCTIVE — ASAP-AES — temperature=0")
    print(f"{'='*65}")

    # Sample 5 global examples for this seed
    examples = sample_examples_global(train_df, seed)

    ex_path = os.path.join(out_dir, "examples_used.json")
    with open(ex_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"  ✓ Examples saved: {ex_path}")

    # Load checkpoint
    predictions = {}
    failed_ids  = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            ckpt        = json.load(f)
            predictions = ckpt.get("predictions", {})
            failed_ids  = ckpt.get("failed_ids",  [])
        print(f"  Resuming — {len(predictions)} predictions, {len(failed_ids)} failures")

    test_nogt = test_df.drop(columns=["domain1_score"], errors="ignore")
    total     = len(test_nogt)
    done_ids  = set(predictions.keys()) | {str(x) for x in failed_ids}

    valid_count   = len(predictions)
    invalid_count = len(failed_ids)

    for i, (_, row) in enumerate(test_nogt.iterrows(), 1):
        essay_id = str(int(row["essay_id"]))
        if essay_id in done_ids:
            continue

        essay_set  = int(row["essay_set"])
        essay_text = str(row["essay"])

        if i % 100 == 0:
            print(f"  Progress: {i}/{total} (✓{valid_count} ✗{invalid_count})")

        result = get_prediction(client, essay_text, essay_set, examples)

        if result["success"]:
            predictions[essay_id] = result["prediction"]
            valid_count += 1
        else:
            failed_ids.append(essay_id)
            invalid_count += 1

        if (valid_count + invalid_count) % 50 == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"predictions": predictions, "failed_ids": failed_ids}, f)

        time.sleep(1.5)

    with open(checkpoint_path, "w") as f:
        json.dump({"predictions": predictions, "failed_ids": failed_ids}, f)
    with open(failed_path, "w") as f:
        json.dump(failed_ids, f, indent=2)

    print(f"\n  ✓ Valid: {valid_count} | ✗ Invalid: {invalid_count}")
    if failed_ids:
        print(f"  ⚠️  Failed IDs saved: {failed_path}")

    pred_rows = []
    for _, row in test_df.iterrows():
        essay_id = str(int(row["essay_id"]))
        pred_rows.append({
            "essay_id":      int(essay_id),
            "essay_set":     int(row["essay_set"]),
            "domain1_score": predictions.get(essay_id, None),
        })
    csv_path = os.path.join(out_dir, f"{MODEL_NAME}_ASAP-AES_inductive.csv")
    pd.DataFrame(pred_rows).to_csv(csv_path, index=False)
    print(f"  ✓ Predictions saved: {csv_path}")

    print(f"\n  Metrics (per essay_set -> macro average):")
    metrics = compute_metrics(predictions, gt_df)

    print(f"\n  MACRO AVERAGE:")
    print(f"    QWK={metrics['macro']['QWK']} Pearson={metrics['macro']['Pearson']} MAE={metrics['macro']['MAE']}")

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return metrics

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

    print("INDUCTIVE EVALUATION — ASAP-AES ONLY")
    print(f"Model:         {MODEL_NAME}")
    print(f"Seeds:         {SEEDS}")
    print(f"Examples:      {NUM_EXAMPLES} (sampled globally per seed, same for all essay sets)")
    print(f"Temperature:   0")
    print(f"Output:        {BASE_OUT}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        sys.exit(0)

    os.makedirs(BASE_OUT, exist_ok=True)
    train_df, test_df, gt_df = load_asap_aes()

    assert "essay_set"     in train_df.columns
    assert "essay_set"     in test_df.columns
    assert "essay_set"     in gt_df.columns
    assert "domain1_score" in train_df.columns
    assert "domain1_score" in gt_df.columns

    print(f"\nTest size per essay_set:")
    for s, g in test_df.groupby("essay_set"):
        print(f"  Set {s}: {len(g)} essays")

    all_seed_metrics = {}
    for seed in SEEDS:
        all_seed_metrics[seed] = run_seed(seed, train_df, test_df, gt_df, client)

    print(f"\n{'='*65}")
    print("CROSS-SEED SUMMARY (macro averages across seeds 42/123/456)")
    print(f"{'='*65}")

    summary_macro  = {}
    summary_global = {}

    for metric in ["QWK", "Pearson", "MAE"]:
        # Per-set macro
        vals_macro = [all_seed_metrics[s]["macro"][metric] for s in SEEDS
                      if not np.isnan(all_seed_metrics[s]["macro"][metric])]
        summary_macro[metric] = {
            "mean":   round(float(np.mean(vals_macro)), 4),
            "std":    round(float(np.std(vals_macro)),  4),
            "values": vals_macro,
        }

        # Global
        vals_global = [all_seed_metrics[s]["global"][metric] for s in SEEDS
                       if not np.isnan(all_seed_metrics[s]["global"][metric])]
        summary_global[metric] = {
            "mean":   round(float(np.mean(vals_global)), 4),
            "std":    round(float(np.std(vals_global)),  4),
            "values": vals_global,
        }

    print("  [Per-set macro average]")
    for metric in ["QWK", "Pearson", "MAE"]:
        m = summary_macro[metric]
        print(f"  {metric}: mean={m['mean']} std={m['std']} values={m['values']}")

    print("\n  [Global (all sets combined)]")
    for metric in ["QWK", "Pearson", "MAE"]:
        m = summary_global[metric]
        print(f"  {metric}: mean={m['mean']} std={m['std']} values={m['values']}")

    summary_path = os.path.join(BASE_OUT, "cross_seed_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "seeds":          {str(k): v for k, v in all_seed_metrics.items()},
            "summary_macro":  summary_macro,
            "summary_global": summary_global,
        }, f, indent=2)
    print(f"\n✓ Cross-seed summary saved: {summary_path}")