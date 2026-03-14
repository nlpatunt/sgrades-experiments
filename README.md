# S-GRADES Experiments

Evaluation experiments for the [S-GRADES benchmark](https://github.com/YOUR_USERNAME/sgrades) — a comprehensive benchmark for automated student response assessment spanning 23 datasets and multiple educational domains.

---

## Models

All experiments use the following three models via OpenRouter:

| Model | Code |
|-------|------|
| GPT-4o-mini | `openai/gpt-4o-mini` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |
| Llama 4 Scout | `meta-llama/llama-4-scout` |

---

## Experiments

### 1. Reasoning Strategy Evaluation

Each model is evaluated across 7 reasoning approaches on all 23 datasets:

| # | Approach |
|---|----------|
| 1 | Inductive |
| 2 | Deductive |
| 3 | Abductive |
| 4 | Inductive + Deductive |
| 5 | Inductive + Abductive |
| 6 | Deductive + Abductive |

**Total conditions:** 3 models × 6 reasoning approaches = 18 experimental runs

Scripts: `run_all_models_sequential.py`, `run_models.py`

---

### 2. Cross-Dataset Generalization

Tests whether reasoning strategies learned from one dataset transfer to another. Two generalization tracks:

**Track A — Within AES (Essay Scoring)**
- Training examples drawn from: `ASAP-AES`
- Test dataset: other AES datasets

**Track B — Cross-Domain (AES → ASAP Short Answer)**
- Training examples drawn from: `ASAP-AES` or `CSEE`
- Test dataset: `ASAP-SAS` (limited to 1,200 samples)

**Approach used:** Inductive + Abductive

**Example configurations:**
```
Training: ASAP-AES  →  Test: ASAP-SAS   | Model: GPT-4o-mini
Training: CSEE      →  Test: ASAP-SAS   | Model: Llama 4 Scout
```

Scripts: `lama_exp/`

---

### 3. Randomization / Consistency Experiment *(Appendix)*

Investigates whether model outputs are consistent across runs by repeating the same experiment with different random seeds and non-zero temperature.

**Setup:**
- Dataset: `ASAP-AES`
- Training examples per run: 5 (sampled randomly)
- Seeds: `42`, `123`, `456`
- Temperature: non-zero
- Model: `meta-llama/llama-4-scout`

**Purpose:** Quantify how much score predictions vary when the prompt examples change, helping establish confidence intervals around the main results.

Scripts: `vizualize_random_sampling.py`, `calculate_SD.py`, `calculate_SD_2.py`

---

## Datasets

23 datasets across traditional essay scoring, short answer grading, and domain-specific assessments.

| Dataset | Task Type | Test Size |
|---------|-----------|-----------|
| ASAP-AES | Essay Scoring | 1,298 |
| ASAP2 | Essay Scoring | 4,946 |
| ASAP-SAS | Short Answer | 3,409 |
| ASAP_plus_plus | Essay Scoring | 1,069 |
| BEEtlE_2way | Short Answer | 1,258 |
| BEEtlE_3way | Short Answer | 1,258 |
| SciEntSBank_2way | Short Answer | 4,969 |
| SciEntSBank_3way | Short Answer | 4,969 |
| CSEE | Essay Scoring | 2,654 |
| persuade_2 | Essay Scoring | 2,600 |
| Mohlar | Short Answer | 455 |
| Ielts_Writing_Dataset | Essay Scoring | 144 |
| Ielst_Writing_Task_2_Dataset | Essay Scoring | 491 |
| Regrading_Dataset_J2C | Short Answer | 198 |
| Rice_Chem_Q1–Q4 | Domain-Specific | 60–66 each |
| OS_Dataset_q1–q5 | Domain-Specific | 8 each |

**Total test examples:** 35,873

---

## Setup

```bash
pip install -r requirements.txt
```

Required environment variables:
```bash
export OPENROUTER_API_KEY="<YOUR_OPENROUTER_API_KEY>"
export HF_TOKEN="your_huggingface_token"
```

---

## Related

- S-GRADES Platform: [link to website repo]
- Paper: *S-GRADES: Studying Generalization of Student Response Assessments in Diverse Evaluative Settings* — LREC-COLING 2026
