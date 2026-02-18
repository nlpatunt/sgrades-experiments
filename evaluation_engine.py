"""
evaluation_engine.py
Standalone evaluation engine extracted from S-GRADES website backend.
Place this file in mllm_evaluation/ directory alongside dataset_ranges.py
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import os
import requests
from dataset_ranges import get_score_range_for_dataset

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


# ─────────────────────────────────────────────
# Ground Truth Loader
# ─────────────────────────────────────────────

def download_ground_truth_private(dataset_name: str) -> Dict[str, Any]:
    normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name

    if not HF_DATASETS_AVAILABLE:
        return {"status": "error", "error": "HuggingFace datasets library not available"}

    id_columns_map = {
        "ASAP-AES": "essay_id",
        "ASAP2": "essay_id",
        "ASAP-SAS": "Id",
        "ASAP_plus_plus": "essay_id",
        "BEEtlE_2way": "ID",
        "BEEtlE_3way": "ID",
        "SciEntSBank_2way": "ID",
        "SciEntSBank_3way": "ID",
        "CSEE": "index",
        "Mohlar": "ID",
        "Ielts_Writing_Dataset": "ID",
        "Ielst_Writing_Task_2_Dataset": "ID",
        "persuade_2": "essay_id_comp",
        "Regrading_Dataset_J2C": "ID",
        "OS_Dataset_q1": "ID",
        "OS_Dataset_q2": "ID",
        "OS_Dataset_q3": "ID",
        "OS_Dataset_q4": "ID",
        "OS_Dataset_q5": "ID",
        "Rice_Chem_Q1": "sis_id",
        "Rice_Chem_Q2": "sis_id",
        "Rice_Chem_Q3": "sis_id",
        "Rice_Chem_Q4": "sis_id"
    }

    try:
        if normalized_name in ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way", "ASAP-SAS"]:
            from io import StringIO

            if "BEEtlE" in normalized_name:
                suffix = "2way" if "2way" in normalized_name else "3way"
                urls = [
                    f"https://huggingface.co/datasets/nlpatunt/BEEtlE/resolve/main/test_{suffix}.csv",
                    f"https://huggingface.co/datasets/nlpatunt/BEEtlE/raw/main/test_{suffix}.csv"
                ]
            elif "SciEntSBank" in normalized_name:
                suffix = "2way" if "2way" in normalized_name else "3way"
                urls = [
                    f"https://huggingface.co/datasets/nlpatunt/SciEntSBank/resolve/main/test_{suffix}.csv",
                    f"https://huggingface.co/datasets/nlpatunt/SciEntSBank/raw/main/test_{suffix}.csv"
                ]
            elif normalized_name == "ASAP-SAS":
                urls = [
                    "https://huggingface.co/datasets/nlpatunt/ASAP-SAS/resolve/main/test.csv",
                    "https://huggingface.co/datasets/nlpatunt/ASAP-SAS/raw/main/test.csv"
                ]

            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if not hf_token:
                try:
                    with open(os.path.expanduser("~/.cache/huggingface/token"), "r") as f:
                        hf_token = f.read().strip()
                except:
                    hf_token = None

            headers = {}
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            for url in urls:
                try:
                    response = requests.get(url, headers=headers, timeout=30)
                    response.raise_for_status()
                    df = pd.read_csv(StringIO(response.text))
                    columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
                    if columns_to_drop:
                        df = df.drop(columns=columns_to_drop)
                    return {"status": "success", "dataset": df, "rows": len(df), "columns": list(df.columns)}
                except Exception as url_error:
                    print(f"Failed to download from {url}: {url_error}")
                    continue

            return {"status": "error", "error": f"All download URLs failed for {normalized_name}"}

        elif normalized_name in ["Rice_Chem_Q1", "Rice_Chem_Q2", "Rice_Chem_Q3", "Rice_Chem_Q4"]:
            q_num = normalized_name.split("_")[-1]
            dataset = load_dataset("nlpatunt/Rice_Chem", data_files=f"{q_num}/test.csv")
            dataset = dataset["train"]
        elif normalized_name.startswith("OS_Dataset_q"):
            q_num = normalized_name.split("_q")[-1]
            dataset = load_dataset("nlpatunt/OS_Dataset", data_files=f"q{q_num}/test.csv", trust_remote_code=True)
            dataset = dataset["train"]
        elif normalized_name == "persuade_2":
            dataset = load_dataset("nlpatunt/persuade_2", data_files="test.csv")
            dataset = dataset["train"]
        elif normalized_name == "Mohlar":
            dataset = load_dataset("nlpatunt/Mohlar", data_files="test.csv")
            dataset = dataset["train"]
        else:
            try:
                dataset = load_dataset(f"nlpatunt/{normalized_name}", split="test", trust_remote_code=True)
            except:
                dataset = load_dataset(f"nlpatunt/{normalized_name}")
                if hasattr(dataset, 'keys'):
                    first_split = list(dataset.keys())[0]
                    dataset = dataset[first_split]

        if 'dataset' in locals():
            df = dataset.to_pandas()
        else:
            return {"status": "error", "error": f"No dataset loaded for {normalized_name}"}

        if normalized_name.startswith("OS_Dataset") and "ID" not in df.columns:
            df["ID"] = range(1, len(df) + 1)

        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)

        id_column = id_columns_map.get(normalized_name, "ID")
        if id_column in df.columns and (df[id_column] == df.index).all():
            df[id_column] = df.index + 1

        return {"status": "success", "dataset": df, "rows": len(df), "columns": list(df.columns)}

    except Exception as e:
        print(f"All loading methods failed: {str(e)}")
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────

class BaseValidator:
    def __init__(self, required_columns, primary_score_column, id_column=None, valid_labels=None):
        self.required_columns = required_columns
        self.primary_score_column = primary_score_column
        self.id_column = id_column or required_columns[0]
        self.valid_labels = valid_labels

    def clean_labels_with_fallback(self, df, handle_invalid='discard', testing_mode=False):
        if not self.valid_labels or self.primary_score_column not in df.columns:
            return df, []

        warnings = []
        df_clean = df.copy()
        original_count = len(df_clean)

        missing_mask = df_clean[self.primary_score_column].isna() | \
                       (df_clean[self.primary_score_column] == '') | \
                       (df_clean[self.primary_score_column].astype(str).str.strip() == '')
        missing_count = missing_mask.sum()

        if missing_count > 0:
            if handle_invalid == 'assign_fallback' or testing_mode:
                df_clean.loc[missing_mask, self.primary_score_column] = '5'
                warnings.append(f"Assigned fallback value 5 to {missing_count} missing labels")
            else:
                df_clean = df_clean[~missing_mask]
                warnings.append(f"Removed {missing_count} rows with missing labels")

        df_clean[self.primary_score_column] = df_clean[self.primary_score_column].astype(str).str.strip()

        label_mapping = {}
        for valid_label in self.valid_labels:
            label_mapping.update({
                valid_label.lower(): valid_label,
                valid_label.upper(): valid_label,
                valid_label.capitalize(): valid_label,
                valid_label: valid_label
            })

        if 'correct' in self.valid_labels:
            label_mapping.update({'1': 'correct', 'true': 'correct', 'yes': 'correct', 'right': 'correct'})
        if 'incorrect' in self.valid_labels:
            label_mapping.update({'0': 'incorrect', 'false': 'incorrect', 'no': 'incorrect', 'wrong': 'incorrect'})

        df_clean['_original_label'] = df_clean[self.primary_score_column].copy()
        df_clean[self.primary_score_column] = df_clean[self.primary_score_column].str.lower().map(label_mapping).fillna(df_clean[self.primary_score_column])

        valid_mask = df_clean[self.primary_score_column].isin(self.valid_labels)
        invalid_mask = ~valid_mask
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            invalid_examples = df_clean[invalid_mask]['_original_label'].unique()[:5].tolist()
            if handle_invalid == 'assign_fallback' or testing_mode:
                df_clean.loc[invalid_mask, self.primary_score_column] = '4'
                warnings.append(f"Assigned fallback value 4 to {invalid_count} invalid labels. Examples: {invalid_examples}")
            else:
                df_clean = df_clean[valid_mask]
                warnings.append(f"Removed {invalid_count} rows with invalid labels. Examples: {invalid_examples}")

        if '_original_label' in df_clean.columns:
            df_clean = df_clean.drop('_original_label', axis=1)

        return df_clean, warnings

    def validate(self, df, testing_mode=False):
        errors = []
        warnings = []

        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: {sorted(list(missing_cols))}")

        df_clean = df.copy()

        if self.id_column in df_clean.columns:
            duplicate_mask = df_clean[self.id_column].duplicated(keep='first')
            duplicate_count = duplicate_mask.sum()
            if duplicate_count > 0:
                if testing_mode:
                    df_clean = df_clean[~duplicate_mask].copy()
                    warnings.append(f"Removed {duplicate_count} duplicate {self.id_column} rows")
                else:
                    errors.append(f"Found {duplicate_count} duplicate {self.id_column} values")

        if self.primary_score_column not in df_clean.columns:
            errors.append(f"{self.primary_score_column} column is required")
            return {"valid": False, "errors": errors, "warnings": warnings, "primary_score_column": self.primary_score_column}

        missing_scores_mask = df_clean[self.primary_score_column].isna()
        missing_scores_count = missing_scores_mask.sum()
        if missing_scores_count > 0:
            if testing_mode:
                df_clean = df_clean[~missing_scores_mask].copy()
                warnings.append(f"Removed {missing_scores_count} rows with missing scores")
            else:
                warnings.append(f"{self.primary_score_column} has {missing_scores_count} missing values")

        if len(df_clean) == 0:
            errors.append("No valid rows remaining after cleanup")
            return {"valid": False, "errors": errors, "warnings": warnings, "primary_score_column": self.primary_score_column}

        if self.valid_labels:
            handle_mode = 'assign_fallback' if testing_mode else 'discard'
            df_clean, label_warnings = self.clean_labels_with_fallback(df_clean, handle_mode, testing_mode)
            warnings.extend(label_warnings)
            if len(df_clean) == 0:
                errors.append("No valid rows remaining after label validation")
        else:
            validator_class = self.__class__.__name__

            if validator_class in ["IELTSWritingValidator", "IELTSTask2Validator"]:
                df_clean[self.primary_score_column] = df_clean[self.primary_score_column].astype(str)
                less_than_pattern = df_clean[self.primary_score_column].str.match(r'^<(\d+\.?\d*)$')
                if less_than_pattern.any():
                    converted = df_clean.loc[less_than_pattern, self.primary_score_column].str.extract(r'^<(\d+\.?\d*)$')[0].astype(float) - 0.5
                    df_clean.loc[less_than_pattern, self.primary_score_column] = converted
                greater_than_pattern = df_clean[self.primary_score_column].str.match(r'^>(\d+\.?\d*)$')
                if greater_than_pattern.any():
                    converted = df_clean.loc[greater_than_pattern, self.primary_score_column].str.extract(r'^>(\d+\.?\d*)$')[0].astype(float) + 0.5
                    df_clean.loc[greater_than_pattern, self.primary_score_column] = converted
                df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')

            if validator_class == "MohlarValidator":
                df_clean[self.primary_score_column] = df_clean[self.primary_score_column].astype(str).str.strip()
                numeric_mask = df_clean[self.primary_score_column].str.match(r'^-?\d+\.?\d*$')
                non_numeric_count = (~numeric_mask).sum()
                if non_numeric_count > 0:
                    df_clean = df_clean[numeric_mask].copy()
                    warnings.append(f"Discarded {non_numeric_count} rows with non-numeric grades")
                df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')

            valid_scores = df_clean[self.primary_score_column].dropna()
            if len(valid_scores) > 0:
                numeric_scores = pd.to_numeric(valid_scores, errors='coerce')
                non_numeric_count = numeric_scores.isna().sum()
                if non_numeric_count > 0:
                    if testing_mode:
                        numeric_mask = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce').notna()
                        df_clean = df_clean[numeric_mask].copy()
                        warnings.append(f"Removed {non_numeric_count} non-numeric rows")
                    else:
                        errors.append(f"{self.primary_score_column} has {non_numeric_count} non-numeric values")
                else:
                    df_clean[self.primary_score_column] = pd.to_numeric(df_clean[self.primary_score_column], errors='coerce')

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "primary_score_column": self.primary_score_column,
            "cleaned_df": df_clean
        }


class ASAPAESValidator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id", "domain1_score"], "domain1_score", "essay_id")

class ASAP2Validator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id", "score"], "score", "essay_id")

class ASAPSASValidator(BaseValidator):
    def __init__(self):
        super().__init__(["Id", "Score1"], "Score1", "Id")

class ASAPPlusPlusValidator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id", "overall_score"], "overall_score", "essay_id")

class CSEEValidator(BaseValidator):
    def __init__(self):
        super().__init__(["index", "overall_score"], "overall_score", "index")

class Persuade2Validator(BaseValidator):
    def __init__(self):
        super().__init__(["essay_id_comp", "holistic_essay_score"], "holistic_essay_score", "essay_id_comp")

class RiceChemValidator(BaseValidator):
    def __init__(self, question_number):
        super().__init__(["sis_id", "Score"], "Score", "sis_id")
        self.question_number = question_number

class MohlarValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "grade"], "grade", "ID")

class IELTSWritingValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "Overall_Score"], "Overall_Score", "ID")

class IELTSTask2Validator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "band_score"], "band_score", "ID")

class RegradingDatasetJ2CValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "grade"], "grade", "ID")

class GradeLikeHumanValidator(BaseValidator):
    def __init__(self, question_number):
        super().__init__(["ID", "score_1"], "score_1", "ID")
        self.question_number = question_number

class BEEtlE2WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect"])

class BEEtlE3WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect", "contradictory"])

class SciEntSBank2WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect"])

class SciEntSBank3WayValidator(BaseValidator):
    def __init__(self):
        super().__init__(["ID", "label"], "label", "ID", valid_labels=["correct", "incorrect", "contradictory"])


def create_rice_chem_validators():
    return {
        "Rice_Chem_Q1": RiceChemValidator("Q1"),
        "Rice_Chem_Q2": RiceChemValidator("Q2"),
        "Rice_Chem_Q3": RiceChemValidator("Q3"),
        "Rice_Chem_Q4": RiceChemValidator("Q4")
    }

def create_OS_Dataset_validators():
    return {
        "OS_Dataset_q1": GradeLikeHumanValidator("1"),
        "OS_Dataset_q2": GradeLikeHumanValidator("2"),
        "OS_Dataset_q3": GradeLikeHumanValidator("3"),
        "OS_Dataset_q4": GradeLikeHumanValidator("4"),
        "OS_Dataset_q5": GradeLikeHumanValidator("5")
    }


class RealEvaluationEngine:
    def __init__(self):
        self.ground_truth_cache = {}

        rice_chem_validators = create_rice_chem_validators()
        OS_Dataset_validators = create_OS_Dataset_validators()

        self.validators = {
            "ASAP-AES": ASAPAESValidator(), "D_ASAP-AES": ASAPAESValidator(),
            "ASAP-AES_Set1": ASAPAESValidator(), "D_ASAP-AES_Set1": ASAPAESValidator(),
            "ASAP-AES_Set2_Domain1": ASAPAESValidator(), "D_ASAP-AES_Set2_Domain1": ASAPAESValidator(),
            "ASAP-AES_Set2_Domain2": ASAPAESValidator(), "D_ASAP-AES_Set2_Domain2": ASAPAESValidator(),
            "ASAP-AES_Set3": ASAPAESValidator(), "D_ASAP-AES_Set3": ASAPAESValidator(),
            "ASAP-AES_Set4": ASAPAESValidator(), "D_ASAP-AES_Set4": ASAPAESValidator(),
            "ASAP-AES_Set5": ASAPAESValidator(), "D_ASAP-AES_Set5": ASAPAESValidator(),
            "ASAP-AES_Set6": ASAPAESValidator(), "D_ASAP-AES_Set6": ASAPAESValidator(),
            "ASAP-AES_Set7": ASAPAESValidator(), "D_ASAP-AES_Set7": ASAPAESValidator(),
            "ASAP-AES_Set8": ASAPAESValidator(), "D_ASAP-AES_Set8": ASAPAESValidator(),
            "ASAP-SAS": ASAPSASValidator(), "D_ASAP-SAS": ASAPSASValidator(),
            "ASAP2": ASAP2Validator(), "D_ASAP2": ASAP2Validator(),
            "ASAP_plus_plus": ASAPPlusPlusValidator(), "D_ASAP_plus_plus": ASAPPlusPlusValidator(),
            "BEEtlE_2way": BEEtlE2WayValidator(), "D_BEEtlE_2way": BEEtlE2WayValidator(),
            "BEEtlE_3way": BEEtlE3WayValidator(), "D_BEEtlE_3way": BEEtlE3WayValidator(),
            "SciEntSBank_2way": SciEntSBank2WayValidator(), "D_SciEntSBank_2way": SciEntSBank2WayValidator(),
            "SciEntSBank_3way": SciEntSBank3WayValidator(), "D_SciEntSBank_3way": SciEntSBank3WayValidator(),
            "Mohlar": MohlarValidator(), "D_Mohlar": MohlarValidator(),
            "CSEE": CSEEValidator(), "D_CSEE": CSEEValidator(),
            "persuade_2": Persuade2Validator(), "D_persuade_2": Persuade2Validator(),
            "Regrading_Dataset_J2C": RegradingDatasetJ2CValidator(), "D_Regrading_Dataset_J2C": RegradingDatasetJ2CValidator(),
            "Ielts_Writing_Dataset": IELTSWritingValidator(), "D_Ielts_Writing_Dataset": IELTSWritingValidator(),
            "Ielts_Writing_Task_2_Dataset": IELTSTask2Validator(), "D_Ielts_Writing_Task_2_Dataset": IELTSTask2Validator(),
            **rice_chem_validators,
            **OS_Dataset_validators,
        }

        for q in ["q1", "q2", "q3", "q4", "q5"]:
            base_name = f"OS_Dataset_{q}"
            d_name = f"D_OS_Dataset_{q}"
            if base_name in self.validators:
                self.validators[d_name] = self.validators[base_name]

        for Q in ["Q1", "Q2", "Q3", "Q4"]:
            base_name = f"Rice_Chem_{Q}"
            d_name = f"D_Rice_Chem_{Q}"
            if base_name in self.validators:
                self.validators[d_name] = self.validators[base_name]

        self.SCORE_COLUMNS = {
            "ASAP-AES": "domain1_score", "ASAP2": "score", "ASAP-SAS": "Score1",
            "ASAP_plus_plus": "overall_score", "BEEtlE_2way": "label", "BEEtlE_3way": "label",
            "SciEntSBank_2way": "label", "SciEntSBank_3way": "label", "CSEE": "overall_score",
            "Mohlar": "grade", "Ielts_Writing_Dataset": "Overall_Score",
            "Ielts_Writing_Task_2_Dataset": "band_score", "persuade_2": "holistic_essay_score",
            "Regrading_Dataset_J2C": "grade",
            "OS_Dataset_q1": "score_1", "OS_Dataset_q2": "score_1", "OS_Dataset_q3": "score_1",
            "OS_Dataset_q4": "score_1", "OS_Dataset_q5": "score_1",
            "Rice_Chem_Q1": "Score", "Rice_Chem_Q2": "Score", "Rice_Chem_Q3": "Score", "Rice_Chem_Q4": "Score"
        }

        self.ID_COLUMNS = {
            "ASAP-AES": "essay_id", "ASAP2": "essay_id", "ASAP-SAS": "Id",
            "ASAP_plus_plus": "essay_id", "BEEtlE_2way": "ID", "BEEtlE_3way": "ID",
            "SciEntSBank_2way": "ID", "SciEntSBank_3way": "ID", "CSEE": "index",
            "Mohlar": "ID", "Ielts_Writing_Dataset": "ID", "Ielts_Writing_Task_2_Dataset": "ID",
            "persuade_2": "essay_id_comp", "Regrading_Dataset_J2C": "ID",
            "OS_Dataset_q1": "ID", "OS_Dataset_q2": "ID", "OS_Dataset_q3": "ID",
            "OS_Dataset_q4": "ID", "OS_Dataset_q5": "ID",
            "Rice_Chem_Q1": "sis_id", "Rice_Chem_Q2": "sis_id",
            "Rice_Chem_Q3": "sis_id", "Rice_Chem_Q4": "sis_id"
        }

    def get_ground_truth(self, dataset_name: str) -> Dict[str, Any]:
        if dataset_name not in self.ground_truth_cache:
            result = download_ground_truth_private(dataset_name)
            if result["status"] == "success":
                self.ground_truth_cache[dataset_name] = result["dataset"]
            return result
        return {"status": "success", "dataset": self.ground_truth_cache[dataset_name]}

    def get_score_column(self, dataset_name: str) -> str:
        return self.SCORE_COLUMNS.get(dataset_name, "score")

    def get_id_column(self, dataset_name: str) -> str:
        return self.ID_COLUMNS.get(dataset_name, "ID")

    def validate_full_structure(self, dataset_name, prediction_df, ground_truth_df, testing_mode=True):
        try:
            if dataset_name in self.validators:
                validator = self.validators[dataset_name]
                validation_result = validator.validate(prediction_df, testing_mode=testing_mode)
                if not validation_result["valid"]:
                    return {"valid": False, "errors": validation_result["errors"], "warnings": validation_result["warnings"]}
                return {"valid": True, "errors": [], "warnings": validation_result["warnings"],
                        "score_column": validation_result["primary_score_column"]}
            return {"valid": False, "errors": [f"No validator found for: {dataset_name}"], "warnings": []}
        except Exception as e:
            return {"valid": False, "errors": [f"Validation failed: {str(e)}"], "warnings": []}

    def match_predictions_to_ground_truth(self, dataset_name, prediction_df, ground_truth_df):
        normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name
        score_col = self.get_score_column(normalized_name)
        id_col = self.get_id_column(normalized_name)

        prediction_df[id_col] = prediction_df[id_col].astype(str)
        ground_truth_df[id_col] = ground_truth_df[id_col].astype(str)

        prediction_df_unique = prediction_df.drop_duplicates(subset=[id_col], keep='first')
        ground_truth_df_unique = ground_truth_df.drop_duplicates(subset=[id_col], keep='first')

        merged_df = prediction_df_unique.merge(
            ground_truth_df_unique[[id_col, score_col]],
            on=id_col, how="inner", suffixes=("_pred", "_true")
        )

        if len(merged_df) == 0:
            return {"status": "error", "error": f"No matching {id_col} found between predictions and ground truth"}

        score_pred_col = f"{score_col}_pred"
        score_true_col = f"{score_col}_true"

        classification_datasets = ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]

        if normalized_name in classification_datasets:
            pred_scores = merged_df[score_pred_col].values
            gt_scores = merged_df[score_true_col].values
            valid_mask = ~(pd.Series(pred_scores).isna() | pd.Series(gt_scores).isna())
        else:
            pred_scores_numeric = pd.to_numeric(merged_df[score_pred_col], errors='coerce')
            gt_scores_numeric = pd.to_numeric(merged_df[score_true_col], errors='coerce')
            valid_mask = ~(pred_scores_numeric.isna() | gt_scores_numeric.isna())
            pred_scores = pred_scores_numeric[valid_mask].values
            gt_scores = gt_scores_numeric[valid_mask].values

        essay_sets = None
        if normalized_name in ["ASAP-AES", "ASAP_plus_plus"]:
            if "essay_set" in merged_df.columns:
                essay_sets = merged_df["essay_set"].values

        return {
            "status": "success",
            "y_pred": pred_scores[valid_mask] if normalized_name in classification_datasets else pred_scores,
            "y_true": gt_scores[valid_mask] if normalized_name in classification_datasets else gt_scores,
            "essay_sets": essay_sets,
            "matched_count": int(valid_mask.sum()),
            "total_predictions": len(prediction_df),
            "total_ground_truth": len(ground_truth_df),
        }

    def calculate_mae_percentage(self, mae, dataset_name, essay_set=1):
        score_range = get_score_range_for_dataset(dataset_name, essay_set)
        range_size = score_range[1] - score_range[0]
        if range_size == 0:
            return 0.0
        return round((mae / range_size) * 100, 2)

    def calculate_metrics(self, y_true, y_pred):
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, f1_score,
            precision_score, recall_score, cohen_kappa_score, accuracy_score
        )
        from scipy.stats import pearsonr

        is_categorical = isinstance(y_true[0], str) if len(y_true) > 0 else False

        if is_categorical:
            unique_labels = sorted(list(set(list(y_true) + list(y_pred))))
            label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
            y_true_numeric = np.array([label_to_num[label] for label in y_true])
            y_pred_numeric = np.array([label_to_num[label] for label in y_pred])
            accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
            try:
                qwk = cohen_kappa_score(y_true_numeric, y_pred_numeric, weights="quadratic")
            except:
                qwk = cohen_kappa_score(y_true_numeric, y_pred_numeric)
            try:
                f1 = f1_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
                precision = precision_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
                recall = recall_score(y_true_numeric, y_pred_numeric, average="weighted", zero_division=0)
            except:
                f1 = precision = recall = accuracy
            correlation = accuracy
            mae = 1.0 - accuracy
            mse = mae ** 2
            rmse = np.sqrt(mse)
        else:
            y_true = np.array(y_true, dtype=np.float64)
            y_pred = np.array(y_pred, dtype=np.float64)
            if len(y_true) == 1:
                perfect = abs(y_true[0] - y_pred[0]) < 1e-10
                correlation = qwk = 1.0 if perfect else 0.0
                f1 = precision = recall = accuracy = 1.0 if perfect else 0.0
                mae = 0.0 if perfect else abs(y_true[0] - y_pred[0])
                mse = 0.0 if perfect else (y_true[0] - y_pred[0]) ** 2
                rmse = np.sqrt(mse)
            else:
                correlation, _ = pearsonr(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                y_true_class = np.round(y_true).astype(np.int64)
                y_pred_class = np.round(y_pred).astype(np.int64)
                accuracy = accuracy_score(y_true_class, y_pred_class)
                try:
                    qwk = cohen_kappa_score(y_true_class, y_pred_class, weights="quadratic")
                except:
                    qwk = 0.0
                try:
                    f1 = f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                    precision = precision_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                    recall = recall_score(y_true_class, y_pred_class, average="weighted", zero_division=0)
                except:
                    f1 = precision = recall = accuracy

        return {
            "quadratic_weighted_kappa": float(qwk) if not pd.isna(qwk) else 0.0,
            "pearson_correlation": float(correlation) if not pd.isna(correlation) else 0.0,
            "mean_absolute_error": float(mae) if not pd.isna(mae) else 0.0,
            "mean_squared_error": float(mse) if not pd.isna(mse) else 0.0,
            "root_mean_squared_error": float(rmse) if not pd.isna(rmse) else 0.0,
            "f1_score": float(f1) if not pd.isna(f1) else 0.0,
            "precision": float(precision) if not pd.isna(precision) else 0.0,
            "recall": float(recall) if not pd.isna(recall) else 0.0,
            "accuracy": float(accuracy) if not pd.isna(accuracy) else 0.0
        }

    def evaluate_submission(self, dataset_name: str, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        metrics = {
            'quadratic_weighted_kappa': 0.0, 'pearson_correlation': 0.0,
            'mean_absolute_error': 0.0, 'mean_squared_error': 0.0,
            'root_mean_squared_error': 0.0, 'f1_score': 0.0,
            'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0
        }

        try:
            gt_result = self.get_ground_truth(dataset_name)
            if gt_result["status"] != "success":
                return {"status": "error", "error": f"Failed to load ground truth: {gt_result.get('error')}"}

            ground_truth_df = gt_result["dataset"]
            normalized_name = dataset_name[2:] if dataset_name.startswith("D_") else dataset_name

            validation_result = self.validate_full_structure(dataset_name, predictions_df, ground_truth_df, testing_mode=True)
            if not validation_result["valid"]:
                return {"status": "error", "error": "Validation failed", "validation_details": validation_result}

            matching_result = self.match_predictions_to_ground_truth(dataset_name, predictions_df, ground_truth_df)
            if matching_result["status"] != "success":
                return {"status": "error", "error": "Matching failed", "matching_details": matching_result}

            y_pred = matching_result["y_pred"]
            y_true = matching_result["y_true"]

            if len(y_pred) == 0 or len(y_true) == 0:
                return {"status": "error", "error": "No valid score pairs found"}

            classification_datasets = ["BEEtlE_2way", "BEEtlE_3way", "SciEntSBank_2way", "SciEntSBank_3way"]
            if normalized_name in classification_datasets:
                label_map = (
                    {'correct': 2, 'Correct': 2, 'incorrect': 0, 'Incorrect': 0, 'contradictory': 1, 'Contradictory': 1,
                     '2': 2, '1': 1, '0': 0, 2: 2, 1: 1, 0: 0}
                    if "3way" in normalized_name else
                    {'correct': 1, 'Correct': 1, 'incorrect': 0, 'Incorrect': 0, '1': 1, '0': 0, 1: 1, 0: 0}
                )
                if len(y_pred) > 0 and isinstance(y_pred[0], str):
                    y_pred = pd.Series(y_pred).str.strip().map(label_map).fillna(0).astype(int).to_numpy()
                if len(y_true) > 0 and isinstance(y_true[0], str):
                    y_true = pd.Series(y_true).str.strip().map(label_map).fillna(0).astype(int).to_numpy()

            try:
                calculated_metrics = self.calculate_metrics(y_true, y_pred)
                if calculated_metrics:
                    metrics.update(calculated_metrics)
                if normalized_name not in ["ASAP-AES", "ASAP_plus_plus"]:
                    mae = metrics.get("mean_absolute_error", 0)
                    metrics["mae_percentage"] = self.calculate_mae_percentage(mae, dataset_name)
            except Exception as e:
                print(f"WARNING: Metrics calculation failed: {e}")

            return {
                "status": "success",
                "metrics": metrics,
                "evaluation_details": {
                    "dataset": dataset_name,
                    "matched_examples": int(len(y_pred)),
                    "total_predictions": int(matching_result["total_predictions"]),
                    "total_ground_truth": int(matching_result["total_ground_truth"]),
                    "score_column": self.get_score_column(dataset_name),
                    "id_column": self.get_id_column(dataset_name),
                    "validation_warnings": validation_result.get("warnings", [])
                }
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e), "dataset": dataset_name, "metrics": metrics}


# Singleton instance — import this in your experiment scripts
real_evaluation_engine = RealEvaluationEngine()