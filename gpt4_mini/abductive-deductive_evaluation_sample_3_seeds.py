#!/usr/bin/env python3
"""
Deductive-Abductive evaluation on sampled datasets - Seeds 42, 123, 456
Reads from sam_datasets/ CSVs directly (no HuggingFace download)
Saves predictions to separate folders per seed
"""

import os
import json
import time
import sys
import re
from openai import OpenAI
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import sys as _sys
_sys.path.insert(0, os.path.expanduser("~/Desktop/Work/sgrades-experiments/gpt4_mini"))
from dataset_ranges import get_score_range_for_dataset

# ============================================================================
# CONFIGURATION
# ============================================================================
SAMPLED_DIR = os.path.expanduser("~/Desktop/Work/sampled_datasets/sam_datasets")
SEEDS       = [42, 123, 456]
MODEL_CODE  = "openai/gpt-4o-mini"
MODEL_NAME  = "gpt-4o-mini"
API_KEY     = "REMOVED_KEY"   # <-- update before running

# ============================================================================
# DATASET COLUMN CONFIG
# ============================================================================
def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    normalized = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    configs = {
        # "ASAP-AES":                    {"id": "essay_id",      "text": "essay",            "score": "domain1_score",        "question": "prompt_name",   "essay_set": "essay_set"},
        # "ASAP2":                       {"id": "essay_id",      "text": "full_text",         "score": "score",                "question": "assignment",    "essay_set": None},
        # "ASAP-SAS":                    {"id": "Id",            "text": "essay_text",        "score": "Score1",               "question": "prompt",        "essay_set": None},
        # "ASAP_plus_plus":              {"id": "essay_id",      "text": "essay",             "score": "overall_score",        "question": "prompt",        "essay_set": "essay_set"},
        # "BEEtlE_2way":                 {"id": "ID",            "text": "student_answer",    "score": "label",                "question": "question_text", "essay_set": None},
        # "BEEtlE_3way":                 {"id": "ID",            "text": "student_answer",    "score": "label",                "question": "question_text", "essay_set": None},
        # "SciEntSBank_2way":            {"id": "ID",            "text": "student_answer",    "score": "label",                "question": "question_text", "essay_set": None},
        # "SciEntSBank_3way":            {"id": "ID",            "text": "student_answer",    "score": "label",                "question": "question_text", "essay_set": None},
        "CSEE":                        {"id": "index",         "text": "essay",             "score": "overall_score",        "question": "prompt",        "essay_set": None},
        "Mohlar":                      {"id": "ID",            "text": "student_answer",    "score": "grade",                "question": "Question",      "essay_set": None},
        "Ielts_Writing_Dataset":       {"id": "ID",            "text": "Essay",             "score": "Overall_Score",        "question": "Question",      "essay_set": None},
        "Ielts_Writing_Task_2_Dataset":{"id": "ID",            "text": "essay",             "score": "band_score",           "question": "prompt",        "essay_set": None},
        "persuade_2":                  {"id": "essay_id_comp", "text": "full_text",         "score": "holistic_essay_score", "question": "assignment",    "essay_set": None},
        "Regrading_Dataset_J2C":       {"id": "ID",            "text": "student_answer",    "score": "grade",                "question": "Question",      "essay_set": None},
        "OS_Dataset_q1":               {"id": "ID",            "text": "answer",            "score": "score_1",              "question": "question",      "essay_set": None},
        "OS_Dataset_q2":               {"id": "ID",            "text": "answer",            "score": "score_1",              "question": "question",      "essay_set": None},
        "OS_Dataset_q3":               {"id": "ID",            "text": "answer",            "score": "score_1",              "question": "question",      "essay_set": None},
        "OS_Dataset_q4":               {"id": "ID",            "text": "answer",            "score": "score_1",              "question": "question",      "essay_set": None},
        "OS_Dataset_q5":               {"id": "ID",            "text": "answer",            "score": "score_1",              "question": "question",      "essay_set": None},
        "Rice_Chem_Q1":                {"id": "sis_id",        "text": "student_response",  "score": "Score",                "question": "Prompt",        "essay_set": None},
        "Rice_Chem_Q2":                {"id": "sis_id",        "text": "student_response",  "score": "Score",                "question": "Prompt",        "essay_set": None},
        "Rice_Chem_Q3":                {"id": "sis_id",        "text": "student_response",  "score": "Score",                "question": "Prompt",        "essay_set": None},
        "Rice_Chem_Q4":                {"id": "sis_id",        "text": "student_response",  "score": "Score",                "question": "Prompt",        "essay_set": None},
    }
    return configs.get(normalized, {"id": "ID", "text": "text", "score": "score", "question": "question", "essay_set": None})

# ============================================================================
# PROMPT
# ============================================================================
def create_deductive_abductive_prompt(essay_text: str, dataset_info: dict) -> dict:
    dataset_name = dataset_info.get('name', 'D_ASAP-AES')
    essay_set    = dataset_info.get('essay_set', 1)
    score_range  = get_score_range_for_dataset(dataset_name, essay_set)
    question     = dataset_info.get('question', '')

    is_3way = '3way' in dataset_name.lower()
    is_2way = '2way' in dataset_name.lower()

    if is_3way:
        system_prompt = f"""You are an expert evaluator using first DEDUCTIVE and then ABDUCTIVE REASONING.

PHASE 1 - DEDUCTIVE REASONING (Apply general principles FIRST):
Start with universal scientific principles:
- Different substances have different physical properties (solubility, density, etc.)
- Voltage measures electrical potential difference between two points
- Zero voltage indicates equal electrical potential
- Scientific statements must align with established laws

Apply these general principles to the specific answer.

PHASE 2 - ABDUCTIVE REASONING (Then infer best explanation):
Now infer the most likely explanation for this answer:
- OBSERVATION: What did the student write?
- POSSIBLE EXPLANATIONS: Why might they have written this?
  * Correct understanding and application
  * Misunderstanding of concepts
  * Direct contradiction of principles
  * Partial knowledge or confusion
- BEST EXPLANATION: Which explanation best accounts for the observation?

COMBINED APPROACH:
1. FIRST apply general scientific principles (deductive)
2. THEN infer the best explanation for the answer (abductive)
3. Classify based on both reasoning methods"""

        user_prompt = f"""Classify this answer using first DEDUCTIVE principles AND ABDUCTIVE REASONING:

QUESTION:
{question}

STUDENT ANSWER (OBSERVATION):
{essay_text}

Work through both phases in 2-3 concise sentences: apply the relevant scientific principles, then infer the best explanation for what the student wrote.

Then output your final classification on a new line in this exact format:
CLASSIFICATION: correct
or
CLASSIFICATION: contradictory
or
CLASSIFICATION: incorrect"""

    elif is_2way:
        system_prompt = f"""You are an expert evaluator using first DEDUCTIVE and then ABDUCTIVE REASONING.

PHASE 1 - DEDUCTIVE REASONING (Apply principles FIRST):
Apply general scientific principles:
- Scientific accuracy and completeness
- Proper application of concepts
- Logical consistency

PHASE 2 - ABDUCTIVE REASONING (Then infer explanation):
Infer the best explanation:
- What does this answer reveal about student understanding?
- Is this correct application or misunderstanding?

COMBINED: Principles first, then best explanation."""

        user_prompt = f"""Classify using first DEDUCTIVE and then ABDUCTIVE REASONING:

QUESTION:
{question}

STUDENT ANSWER (OBSERVATION):
{essay_text}

Work through both phases in 2-3 concise sentences: apply the relevant principles, then infer the best explanation for what the student wrote.

Then output your final classification on a new line in this exact format:
CLASSIFICATION: correct
or
CLASSIFICATION: incorrect"""

    else:
        system_prompt = f"""You are an expert scorer using first DEDUCTIVE and then ABDUCTIVE REASONING.

PHASE 1 - DEDUCTIVE REASONING (Apply rules FIRST):
Apply general scoring rules:
- Completeness and accuracy requirements
- Depth of reasoning expectations
- Clear communication standards
- Evidence-based conclusions

PHASE 2 - ABDUCTIVE REASONING (Then infer quality):
Infer the best explanation for the essay quality:
- What does this essay reveal about understanding?
- What's the most likely level of mastery?
- Which score best explains the observed quality?

SCORING RANGE: {score_range}
COMBINED: Apply rules first, then infer best score."""

        user_prompt = f"""Score this essay using first DEDUCTIVE and then ABDUCTIVE REASONING:

QUESTION/PROMPT:
{question}

ESSAY (OBSERVATION):
{essay_text}

Work through both phases in 2-3 concise sentences: apply the general scoring rules, then infer the best explanation for the essay quality.

Then output your final score on a new line in this exact format:
SCORE: [number between {score_range[0]} and {score_range[1]}]"""

    return {"system": system_prompt, "user": user_prompt}

# ============================================================================
# VALIDATE PREDICTION
# ============================================================================
def validate_prediction(prediction_text: str, dataset_name: str, essay_set: int = 1) -> Dict[str, Any]:
    normalized       = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    is_3way          = '3way' in normalized.lower()
    is_2way          = '2way' in normalized.lower()
    prediction_clean = prediction_text.strip().lower()

    if is_3way:
        m = re.search(r'classification:\s*(correct|contradictory|incorrect)', prediction_clean)
        if m:
            return {'valid': True, 'extracted': m.group(1), 'error': None, 'raw': prediction_text}
        if 'incorrect'     in prediction_clean: return {'valid': True, 'extracted': 'incorrect',     'error': None, 'raw': prediction_text}
        if 'contradictory' in prediction_clean: return {'valid': True, 'extracted': 'contradictory', 'error': None, 'raw': prediction_text}
        if 'correct'       in prediction_clean: return {'valid': True, 'extracted': 'correct',       'error': None, 'raw': prediction_text}
        return {'valid': False, 'extracted': None, 'error': f'Invalid 3way: {prediction_text[:50]}', 'raw': prediction_text}

    elif is_2way:
        m = re.search(r'classification:\s*(correct|incorrect)', prediction_clean)
        if m:
            return {'valid': True, 'extracted': m.group(1), 'error': None, 'raw': prediction_text}
        if 'incorrect' in prediction_clean: return {'valid': True, 'extracted': 'incorrect', 'error': None, 'raw': prediction_text}
        if 'correct'   in prediction_clean: return {'valid': True, 'extracted': 'correct',   'error': None, 'raw': prediction_text}
        return {'valid': False, 'extracted': None, 'error': f'Invalid 2way: {prediction_text[:50]}', 'raw': prediction_text}

    else:
        score_range = get_score_range_for_dataset(dataset_name, essay_set)
        m = re.search(r'score:\s*(\d+(?:\.\d+)?)', prediction_clean)
        if m:
            score = float(m.group(1))
            if score_range[0] <= score <= score_range[1]:
                return {'valid': True, 'extracted': score, 'error': None, 'raw': prediction_text}
            return {'valid': False, 'extracted': score, 'error': f'Out of range {score_range}', 'raw': prediction_text}
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', prediction_clean)
        if not numbers:
            return {'valid': False, 'extracted': None, 'error': f'No number: {prediction_text[:50]}', 'raw': prediction_text}
        score = float(numbers[0])
        if score < score_range[0] or score > score_range[1]:
            return {'valid': False, 'extracted': score, 'error': f'Out of range {score_range}', 'raw': prediction_text}
        return {'valid': True, 'extracted': score, 'error': None, 'raw': prediction_text}

# ============================================================================
# API CALL
# ============================================================================
def call_api(client, model_code, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_code,
                messages=messages,
                max_tokens=500,
                temperature=0.1,
                extra_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "S-GRADES Ded-Abd"}
            )
            return {
                'response': response,
                'tokens': {
                    'prompt':     response.usage.prompt_tokens     if response.usage else 0,
                    'completion': response.usage.completion_tokens if response.usage else 0,
                    'total':      (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
                },
                'success': True
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {'response': None, 'tokens': {}, 'success': False, 'error': str(e)}
            time.sleep(2 ** attempt)

def get_prediction_with_retry(client, model_code, essay_text, question,
                               dataset_name, score_range, essay_set=1, max_retries=5):
    for attempt in range(max_retries):
        dataset_info = {'name': dataset_name, 'essay_set': essay_set,
                        'question': question, 'description': f'{dataset_name} evaluation'}
        prompt_data  = create_deductive_abductive_prompt(essay_text, dataset_info)
        messages     = [
            {"role": "system", "content": prompt_data["system"]},
            {"role": "user",   "content": prompt_data["user"]}
        ]
        api_result = call_api(client, model_code, messages)

        if (not api_result['success'] or api_result.get('response') is None or
                not hasattr(api_result['response'], 'choices') or
                len(api_result['response'].choices) == 0 or
                api_result['response'].choices[0].message.content is None):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {'success': False, 'error': api_result.get('error', 'API failed'), 'attempts': max_retries}

        response_text = api_result['response'].choices[0].message.content.strip()
        validation    = validate_prediction(response_text, dataset_name, essay_set)

        if validation['valid']:
            return {
                'success':      True,
                'prediction':   validation['extracted'],
                'raw_response': validation.get('raw', response_text),
                'tokens':       api_result['tokens'],
                'attempts':     attempt + 1
            }
        if attempt < max_retries - 1:
            time.sleep(1)

    return {
        'success':       False,
        'error':         f"Validation failed after {max_retries} attempts: {validation['error']}",
        'last_response': response_text if 'response_text' in locals() else 'No response',
        'attempts':      max_retries
    }

# ============================================================================
# LOAD SAMPLED CSV
# ============================================================================
def load_sampled_csv(dataset_name: str, seed: int) -> Dict[str, Any]:
    csv_path = os.path.join(SAMPLED_DIR, f"{dataset_name}_test_21.csv")

    if not os.path.exists(csv_path):
        print(f"  ✗ Sampled file not found: {csv_path}")
        return {"status": "error", "error": "File not found"}

    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ Loaded: {len(df)} rows from {os.path.basename(csv_path)}")

        # Drop ground truth columns to avoid leakage
        cols      = get_dataset_columns(dataset_name)
        score_col = cols["score"]
        if score_col in df.columns and df[score_col].notna().sum() > 0:
            print(f"  ⚠️  Dropping ground truth column: {score_col}")
            df = df.drop(columns=[score_col])
        if 'label' in df.columns and df['label'].notna().sum() > 0:
            print(f"  ⚠️  Dropping ground truth column: label")
            df = df.drop(columns=['label'])

        # Shuffle with seed
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        return {"status": "success", "dataset": df}

    except Exception as e:
        print(f"  ✗ Error loading {csv_path}: {e}")
        return {"status": "error", "error": str(e)}

# ============================================================================
# SAVE PREDICTIONS CSV
# ============================================================================
def save_predictions_as_csv(test_df: pd.DataFrame, predictions_map: Dict,
                             raw_responses_map: Dict, dataset_name: str,
                             score_column: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_df = test_df.copy()

    id_col = get_dataset_columns(dataset_name)["id"]
    for idx in output_df.index:
        row_id = str(output_df.loc[idx, id_col])
        if row_id in predictions_map:
            output_df.loc[idx, score_column] = predictions_map[row_id]
        if row_id in raw_responses_map:
            output_df.loc[idx, 'reasoning'] = raw_responses_map[row_id]

    normalized   = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    csv_filename = f"{MODEL_NAME}_{normalized}_deductive_abductive.csv"
    csv_path     = os.path.join(output_dir, csv_filename)
    output_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    return csv_path

# ============================================================================
# DATASETS
# ============================================================================
DATASETS = [
    #"D_ASAP-AES", "D_ASAP2", "D_ASAP-SAS", "D_ASAP_plus_plus",
    #"D_BEEtlE_2way", "D_BEEtlE_3way", "D_SciEntSBank_2way", "D_SciEntSBank_3way",
    "D_CSEE", "D_Mohlar", "D_Ielts_Writing_Dataset", "D_Ielts_Writing_Task_2_Dataset",
    "D_persuade_2", "D_Regrading_Dataset_J2C",
    "D_OS_Dataset_q1", "D_OS_Dataset_q2", "D_OS_Dataset_q3", "D_OS_Dataset_q4", "D_OS_Dataset_q5",
    "D_Rice_Chem_Q1", "D_Rice_Chem_Q2", "D_Rice_Chem_Q3", "D_Rice_Chem_Q4",
]

# ============================================================================
# RUN ONE SEED
# ============================================================================
def run_seed(seed: int, client):
    output_dir      = os.path.expanduser(f"~/Desktop/Work/sgrades-experiments/gpt4_mini/seeds/deductive_abductive_seed{seed}_predictions")
    checkpoint_file = os.path.expanduser(f"~/Desktop/Work/sgrades-experiments/gpt4_mini/seeds/deductive_abductive_seed{seed}_checkpoint.json")

    print(f"\n{'='*70}")
    print(f"SEED {seed} — DEDUCTIVE-ABDUCTIVE EVALUATION — {MODEL_NAME}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # Load checkpoint
    completed = set()
    results   = {'seed': seed, 'model': MODEL_NAME, 'reasoning': 'deductive_abductive',
                 'datasets': [], 'timestamp': datetime.now().isoformat()}
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                results   = json.load(f)
            completed = set(d['dataset_name'] for d in results.get('datasets', []))
            print(f"  Resuming — {len(completed)} datasets already done")
        except:
            pass

    remaining = [d for d in DATASETS if d not in completed]
    print(f"  Remaining: {len(remaining)}/{len(DATASETS)} datasets\n")

    for idx, dataset_name in enumerate(remaining, 1):
        print(f"\n{'='*60}")
        print(f"[Seed {seed}] Dataset {idx}/{len(remaining)}: {dataset_name}")
        print(f"{'='*60}")

        load_result = load_sampled_csv(dataset_name, seed)
        if load_result["status"] != "success":
            print(f"  ✗ Skipping")
            continue

        test_df = load_result["dataset"]
        cols    = get_dataset_columns(dataset_name)

        predictions_map   = {}
        raw_responses_map = {}
        valid_count       = 0
        invalid_count     = 0

        for i, (_, row) in enumerate(test_df.iterrows(), 1):
            print(f"  [{i}/{len(test_df)}] Processing...", end="\r")
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(test_df)} (✓{valid_count} ✗{invalid_count})")

            essay_id   = str(row[cols["id"]])
            essay_text = row[cols["text"]]
            question   = str(row[cols["question"]]) if cols["question"] and cols["question"] in row else ""

            essay_set = 1
            if cols.get("essay_set") and cols["essay_set"] and cols["essay_set"] in row:
                try:
                    essay_set = int(row[cols["essay_set"]])
                except:
                    pass

            score_range = get_score_range_for_dataset(dataset_name, essay_set)

            result = get_prediction_with_retry(
                client, MODEL_CODE, essay_text, question,
                dataset_name, score_range, essay_set
            )

            if result['success']:
                predictions_map[essay_id]   = result['prediction']
                raw_responses_map[essay_id] = result.get('raw_response', '')
                valid_count += 1
            else:
                invalid_count += 1

            time.sleep(1.5)

        # Save CSV
        save_predictions_as_csv(test_df, predictions_map, raw_responses_map,
                                 dataset_name, cols["score"], output_dir)

        results['datasets'].append({
            'dataset_name': dataset_name,
            'valid':        valid_count,
            'invalid':      invalid_count,
            'total':        len(test_df)
        })
        completed.add(dataset_name)

        print(f"  ✓ Valid: {valid_count} | ✗ Invalid: {invalid_count}")

        # Save checkpoint
        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(results, f, indent=2)

        time.sleep(3)

    print(f"\n✓ Seed {seed} complete — predictions in: {output_dir}")

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return output_dir

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Please update API_KEY in the script before running")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

    print("DEDUCTIVE-ABDUCTIVE EVALUATION — SAMPLED DATASETS")
    print(f"Seeds: {SEEDS}")
    print(f"Sampled data from: {SAMPLED_DIR}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled")
        sys.exit(0)

    for seed in SEEDS:
        run_seed(seed, client)

    print("\n" + "="*70)
    print("ALL SEEDS COMPLETE")
    print("="*70)
    print("Prediction folders:")
    for seed in SEEDS:
        folder = os.path.expanduser(f"~/Desktop/Work/sgrades-experiments/gpt4_mini/seeds/deductive_abductive_seed{seed}_predictions")
        print(f"  Seed {seed}: {folder}")