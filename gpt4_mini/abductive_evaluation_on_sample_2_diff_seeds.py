#!/usr/bin/env python3
"""
Abductive evaluation on sampled datasets - Seeds 123 and 456
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
SAMPLED_DIR   = os.path.expanduser("~/Desktop/Work/sampled_datasets/sam_datasets")
SEEDS         = [42]
MODEL_CODE    = "openai/gpt-4o-mini"
MODEL_NAME    = "gpt-4o-mini"
API_KEY       = "REMOVED_KEY"

# ============================================================================
# DATASET COLUMN CONFIG
# ============================================================================
def get_dataset_columns(dataset_name: str) -> Dict[str, str]:
    normalized = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    configs = {
       # "ASAP-AES":                    {"id": "essay_id",      "text": "essay",            "score": "domain1_score",       "question": "prompt_name",   "essay_set": "essay_set"},
       # "ASAP2":                       {"id": "essay_id",      "text": "full_text",         "score": "score",               "question": "assignment",    "essay_set": None},
        #"ASAP-SAS":                    {"id": "Id",            "text": "essay_text",        "score": "Score1",              "question": "prompt",        "essay_set": None},
        #"ASAP_plus_plus":              {"id": "essay_id",      "text": "essay",             "score": "overall_score",       "question": "prompt",        "essay_set": "essay_set"},
        #"BEEtlE_2way":                 {"id": "ID",            "text": "student_answer",    "score": "label",               "question": "question_text", "essay_set": None},
        #"BEEtlE_3way":                 {"id": "ID",            "text": "student_answer",    "score": "label",               "question": "question_text", "essay_set": None},
        #"SciEntSBank_2way":            {"id": "ID",            "text": "student_answer",    "score": "label",               "question": "question_text", "essay_set": None},
        "SciEntSBank_3way":            {"id": "ID",            "text": "student_answer",    "score": "label",               "question": "question_text", "essay_set": None},
        #"CSEE":                        {"id": "index",         "text": "essay",             "score": "overall_score",       "question": "prompt",        "essay_set": None},
        #"Mohlar":                      {"id": "ID",            "text": "student_answer",    "score": "grade",               "question": "Question",      "essay_set": None},
        #"Ielts_Writing_Dataset":       {"id": "ID",            "text": "Essay",             "score": "Overall_Score",       "question": "Question",      "essay_set": None},
        #"Ielts_Writing_Task_2_Dataset":{"id": "ID",            "text": "essay",             "score": "band_score",          "question": "prompt",        "essay_set": None},
        "persuade_2":                  {"id": "essay_id_comp", "text": "full_text",         "score": "holistic_essay_score","question": "assignment",    "essay_set": None},
        "Regrading_Dataset_J2C":       {"id": "ID",            "text": "student_answer",    "score": "grade",               "question": "Question",      "essay_set": None},
        "OS_Dataset_q1":               {"id": "ID",            "text": "answer",            "score": "score_1",             "question": "question",      "essay_set": None},
        "OS_Dataset_q2":               {"id": "ID",            "text": "answer",            "score": "score_1",             "question": "question",      "essay_set": None},
        "OS_Dataset_q3":               {"id": "ID",            "text": "answer",            "score": "score_1",             "question": "question",      "essay_set": None},
        "OS_Dataset_q4":               {"id": "ID",            "text": "answer",            "score": "score_1",             "question": "question",      "essay_set": None},
        "OS_Dataset_q5":               {"id": "ID",            "text": "answer",            "score": "score_1",             "question": "question",      "essay_set": None},
        "Rice_Chem_Q1":                {"id": "sis_id",        "text": "student_response",  "score": "Score",               "question": "Prompt",        "essay_set": None},
        "Rice_Chem_Q2":                {"id": "sis_id",        "text": "student_response",  "score": "Score",               "question": "Prompt",        "essay_set": None},
        "Rice_Chem_Q3":                {"id": "sis_id",        "text": "student_response",  "score": "Score",               "question": "Prompt",        "essay_set": None},
        "Rice_Chem_Q4":                {"id": "sis_id",        "text": "student_response",  "score": "Score",               "question": "Prompt",        "essay_set": None},
    }
    return configs.get(normalized, {"id": "ID", "text": "text", "score": "score", "question": "question", "essay_set": None})

# ============================================================================
# PROMPT  (unchanged from original)
# ============================================================================
def create_abductive_prompt(essay_text: str, dataset_info: dict) -> dict:
    dataset_name = dataset_info.get('name', 'D_ASAP-AES')
    essay_set    = dataset_info.get('essay_set', 1)
    score_range  = get_score_range_for_dataset(dataset_name, essay_set)
    question     = dataset_info.get('question', '')

    is_3way = '3way' in dataset_name.lower()
    is_2way = '2way' in dataset_name.lower()

    if is_3way:
        examples = """
OBSERVATION 1: Student claims "all solids must have the same solubility"
POSSIBLE EXPLANATIONS:
- Student correctly understands solubility is universal → Would expect: "Yes, measurements correct"
- Student misunderstands solubility as substance-specific → Would expect: discussion of different properties
- Student contradicts basic chemistry principles → Would expect: incorrect claim about uniformity
BEST EXPLANATION: Student contradicts established principle (different substances have different properties)
CLASSIFICATION: contradictory

OBSERVATION 2: Student says terminals have "0V means terminals are the same"
POSSIBLE EXPLANATIONS:
- Student understands 0V = same potential/connected → Would expect: "same electrical state"
- Student confuses physical identity with electrical state → Would expect: vague "same" language
- Student correctly explains voltage reading → Would expect: mention of connection/continuity
BEST EXPLANATION: Ambiguous answer suggesting confusion about what "same" means (physical vs electrical)
CLASSIFICATION: contradictory

OBSERVATION 3: Student says "same electrical state" for 0V reading
POSSIBLE EXPLANATIONS:
- Student randomly guessed → Would expect: no technical terminology
- Student memorized without understanding → Would expect: definition without application
- Student understands voltage measures potential difference → Would expect: correct technical terms
BEST EXPLANATION: Student correctly understands 0V indicates equal electrical potential
CLASSIFICATION: correct
"""
        system_prompt = f"""You are an expert evaluator using ABDUCTIVE REASONING.

ABDUCTIVE PROCESS:
1. Observe what the student wrote
2. Generate possible explanations for why they wrote it
3. Identify which explanation best fits the evidence
4. Derive classification from best explanation

EXAMPLES OF ABDUCTIVE REASONING:
{examples}

Apply this process: Observation → Possible Explanations → Best Fit → Classification

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Use abductive reasoning to classify this answer:

QUESTION:
{question}

STUDENT ANSWER (OBSERVATION):
{essay_text}

Work through the abductive process in 2-3 concise sentences: observe the answer, generate possible explanations, identify the best explanation.

Then output your final classification on a new line in this exact format:
CLASSIFICATION: correct
or
CLASSIFICATION: contradictory
or
CLASSIFICATION: incorrect"""

    elif is_2way:
        examples = """
OBSERVATION 1: Student says "switch must be in path with the bulb"
POSSIBLE EXPLANATIONS:
- Student memorized without understanding → Would expect: no reasoning about why
- Student understands circuit paths → Would expect: mention of path/circuit
- Student randomly guessed correctly → Would expect: very brief answer
BEST EXPLANATION: Student demonstrates understanding that switches affect components in their path
CLASSIFICATION: correct

OBSERVATION 2: Student recommends "adding shelter" for territorial lizards
POSSIBLE EXPLANATIONS:
- Student thinks shelter helps any problem → Would expect: generic "make comfortable"
- Student understands territorial behavior needs → Would expect: shelter for separate spaces
- Student doesn't understand territoriality → Would expect: irrelevant suggestion
BEST EXPLANATION: Student correctly infers territorial animals need physical separation/hiding spots
CLASSIFICATION: correct

OBSERVATION 3: Student says "different solubilities mean measurements are wrong"
POSSIBLE EXPLANATIONS:
- Student thinks all substances should behave identically → Would expect: claim of error
- Student understands solubility is substance-specific → Would expect: "both can be correct"
- Student confuses variability with experimental error → Would expect: focus on measurement problems
BEST EXPLANATION: Student misunderstands that different solubilities are expected, not errors
CLASSIFICATION: incorrect
"""
        system_prompt = f"""You are an expert evaluator using ABDUCTIVE REASONING.

ABDUCTIVE PROCESS:
1. Observe what the student wrote
2. Generate possible explanations
3. Identify best explanation
4. Classify based on that

EXAMPLES OF ABDUCTIVE REASONING:
{examples}

TASK: {dataset_info.get('description', 'Answer classification')}"""

        user_prompt = f"""Use abductive reasoning to classify this answer:

QUESTION:
{question}

STUDENT ANSWER (OBSERVATION):
{essay_text}

Work through the abductive process in 2-3 concise sentences: observe the answer, generate possible explanations, identify the best explanation.

Then output your final classification on a new line in this exact format:
CLASSIFICATION: correct
or
CLASSIFICATION: incorrect"""

    else:
        examples = """
OBSERVATION 1: Student provides "amount of each sample, temperature, and rinsing duration"
POSSIBLE EXPLANATIONS:
- Student randomly listed items → Would expect: irrelevant details
- Student identified genuinely missing info → Would expect: procedurally important details
- Student didn't read procedure carefully → Would expect: things already specified
BEST EXPLANATION: Student correctly identified three critical missing procedural details
INFERENCE: Demonstrates good understanding, minor writing issues
SCORE: 2/3

OBSERVATION 2: Student wrote "It took 10 time units" with no explanation
POSSIBLE EXPLANATIONS:
- Student ran simulation but doesn't understand why → Would expect: correct number, no reasoning
- Student guessed lucky → Would expect: wrong answer or right answer with uncertainty
- Student understands deeply → Would expect: explanation of process interactions
BEST EXPLANATION: Student obtained correct result but cannot explain the underlying process
INFERENCE: Partial understanding - can execute but not explain
SCORE: 8/15

OBSERVATION 3: Student discusses "Coulomb's Law, electron-electron repulsion, 2p to 2s orbital jump"
POSSIBLE EXPLANATIONS:
- Student memorized terms without understanding → Would expect: misapplied concepts
- Student has partial understanding → Would expect: some correct, some errors
- Student fully understands → Would expect: all concepts correctly applied
BEST EXPLANATION: Student understands shell transitions and repulsion but incorrectly dismisses Zeff changes
INFERENCE: Solid grasp with one significant conceptual error
SCORE: 5/8
"""
        system_prompt = f"""You are an expert scorer using ABDUCTIVE REASONING.

ABDUCTIVE PROCESS:
1. Observe what the student wrote
2. Generate explanations for their knowledge state
3. Identify which explanation best fits the evidence
4. Infer appropriate score from that explanation

EXAMPLES OF ABDUCTIVE SCORING:
{examples}

SCORING RANGE: {score_range}
TASK: {dataset_info.get('description', 'Essay scoring')}"""

        user_prompt = f"""Use abductive reasoning to score:

QUESTION/PROMPT:
{question}

ESSAY (OBSERVATION):
{essay_text}

Work through the abductive process in 2-3 concise sentences: observe the response, generate possible explanations for the student's knowledge state, identify the best explanation.

Then output your final score on a new line in this exact format:
SCORE: [number between {score_range[0]} and {score_range[1]}]"""

    return {"system": system_prompt, "user": user_prompt}

# ============================================================================
# VALIDATE PREDICTION
# ============================================================================
def validate_prediction(prediction_text: str, dataset_name: str, essay_set: int = 1) -> Dict[str, Any]:
    normalized      = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    is_3way         = '3way' in normalized.lower()
    is_2way         = '2way' in normalized.lower()
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
                extra_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "S-GRADES Abductive"}
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
        prompt_data  = create_abductive_prompt(essay_text, dataset_info)
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
                time.sleep(2 ** attempt); continue
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
        'success': False,
        'error':   f"Validation failed after {max_retries} attempts: {validation['error']}",
        'last_response': response_text if 'response_text' in locals() else 'No response',
        'attempts': max_retries
    }

# ============================================================================
# LOAD SAMPLED CSV
# ============================================================================
def load_sampled_csv(dataset_name: str, seed: int) -> Dict[str, Any]:
    """Load from sam_datasets folder, sample with given seed if needed"""
    normalized = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
    csv_path   = os.path.join(SAMPLED_DIR, f"{dataset_name}_test_21.csv")

    if not os.path.exists(csv_path):
        print(f"  ✗ Sampled file not found: {csv_path}")
        return {"status": "error", "error": "File not found"}

    try:
        df = pd.read_csv(csv_path)
        print(f"  ✓ Loaded: {len(df)} rows from {os.path.basename(csv_path)}")

        # Drop ground truth columns to avoid leakage
        cols = get_dataset_columns(dataset_name)
        score_col = cols["score"]
        if score_col in df.columns and df[score_col].notna().sum() > 0:
            print(f"  ⚠️  Dropping ground truth column: {score_col}")
            df = df.drop(columns=[score_col])
        if 'label' in df.columns and df['label'].notna().sum() > 0:
            print(f"  ⚠️  Dropping ground truth column: label")
            df = df.drop(columns=['label'])

        # Shuffle with seed so each seed sees same rows in different order
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
    csv_filename = f"{MODEL_NAME}_{normalized}_abductive.csv"
    csv_path     = os.path.join(output_dir, csv_filename)
    output_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")
    return csv_path

# ============================================================================
# RUN ONE SEED
# ============================================================================
DATASETS = [
   # "D_ASAP-SAS", "D_ASAP_plus_plus",
    #"D_BEEtlE_2way", "D_BEEtlE_3way", "D_SciEntSBank_2way",
    "D_SciEntSBank_3way",
    #"D_CSEE", "D_Mohlar", "D_Ielts_Writing_Dataset", "D_Ielts_Writing_Task_2_Dataset",
    "D_persuade_2", "D_Regrading_Dataset_J2C",
    "D_OS_Dataset_q1", "D_OS_Dataset_q2", "D_OS_Dataset_q3", "D_OS_Dataset_q4", "D_OS_Dataset_q5",
    "D_Rice_Chem_Q1", "D_Rice_Chem_Q2", "D_Rice_Chem_Q3", "D_Rice_Chem_Q4",
]

def run_seed(seed: int, client):
    output_dir      = os.path.expanduser(f"~/Desktop/Work/sgrades-experiments/gpt4_mini/seeds/abductive_seed{seed}_predictions")
    checkpoint_file = os.path.expanduser(f"~/Desktop/Work/sgrades-experiments/gpt4_mini/seeds/abductive_seed{seed}_checkpoint.json")

    print(f"\n{'='*70}")
    print(f"SEED {seed} — ABDUCTIVE EVALUATION — {MODEL_NAME}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # Load checkpoint
    completed = set()
    results   = {'seed': seed, 'model': MODEL_NAME, 'datasets': [], 'timestamp': datetime.now().isoformat()}
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

        predictions_map    = {}
        raw_responses_map  = {}
        valid_count        = 0
        invalid_count      = 0

        for i, (_, row) in enumerate(test_df.iterrows(), 1):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(test_df)} (✓{valid_count} ✗{invalid_count})")

            essay_id   = str(row[cols["id"]])
            essay_text = row[cols["text"]]
            question   = str(row.get(cols["question"], "")) if cols["question"] in row else ""

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
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)

    print("ABDUCTIVE EVALUATION — SAMPLED DATASETS")
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
        folder = os.path.expanduser(f"~/Desktop/Work/sgrades-experiments/gpt4_mini/seeds/abductive_seed{seed}_predictions")
        print(f"  Seed {seed}: {folder}")