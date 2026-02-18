import numpy as np

# AES Data - Model scores across 6 datasets and 6 strategies
# Format: [Ind, Ded, Abd, Ind+Abd, Ind+Ded, Ded+Abd]

gpt_4o_mini = {
    'ASAP-AES': [0.92, 0.95, 0.95, 0.92, 0.95, 0.92],
    'ASAP2': [0.29, 0.17, 0.16, 0.28, 0.25, 0.11],
    'ASAP++': [0.19, 0.19, 0.19, 0.19, 0.21, 0.15],
    'Persuade-2': [0.67, 0.53, 0.49, 0.67, 0.67, 0.54],
    'IELTS_Writing': [0.44, 0.29, 0.28, 0.35, 0.41, 0.27],
    'IELTS_Task_2': [0.19, 0.23, 0.24, 0.19, 0.18, 0.19]
}

gemini_2_5_flash = {
    'ASAP-AES': [0.60, 0.88, 0.88, 0.72, 0.82, 0.81],
    'ASAP2': [0.33, 0.24, 0.20, 0.38, 0.35, 0.18],
    'ASAP++': [0.30, 0.17, 0.14, 0.29, 0.23, 0.14],
    'Persuade-2': [0.78, 0.70, 0.68, 0.77, 0.77, 0.59],
    'IELTS_Writing': [0.49, 0.37, 0.43, 0.47, 0.44, 0.22],
    'IELTS_Task_2': [0.20, 0.19, 0.17, 0.20, 0.19, 0.14]
}

llama_4_scout = {
    'ASAP-AES': [0.62, 0.86, 0.91, 0.89, 0.94, 0.92],
    'ASAP2': [0.41, 0.15, 0.18, 0.34, 0.19, 0.16],
    'ASAP++': [0.24, 0.22, 0.22, 0.23, 0.23, 0.24],
    'Persuade-2': [0.30, 0.47, 0.38, 0.26, 0.32, 0.37],
    'IELTS_Writing': [0.37, 0.55, 0.31, 0.27, 0.64, 0.50],
    'IELTS_Task_2': [0.18, 0.22, 0.22, 0.12, 0.26, 0.25]
}

def calculate_strategy_variance(model_data, model_name):
    """
    Calculate strategy variance: average SD across strategies for each dataset
    """
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    dataset_sds = []
    
    for dataset, scores in model_data.items():
        sd = np.std(scores, ddof=1)  # Sample standard deviation
        dataset_sds.append(sd)
        print(f"{dataset:20s}: scores={scores}")
        print(f"{' '*20}  Mean={np.mean(scores):.3f}, SD={sd:.4f}")
    
    avg_sd = np.mean(dataset_sds)
    print(f"\n{model_name} Average Strategy Variance: σ ≈ {avg_sd:.4f}")
    print(f"Individual dataset SDs: {[f'{sd:.4f}' for sd in dataset_sds]}")
    
    return avg_sd

# Calculate for all models
print("\n" + "="*60)
print("STRATEGY VARIANCE CALCULATION FOR AES DATASETS")
print("="*60)

gpt_sd = calculate_strategy_variance(gpt_4o_mini, "GPT-4o-mini")
gemini_sd = calculate_strategy_variance(gemini_2_5_flash, "Gemini-2.5-Flash")
llama_sd = calculate_strategy_variance(llama_4_scout, "LLaMA-4-Scout")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"GPT-4o-mini:        σ ≈ {gpt_sd:.4f}")
print(f"Gemini-2.5-Flash:   σ ≈ {gemini_sd:.4f}")
print(f"LLaMA-4-Scout:      σ ≈ {llama_sd:.4f}")
print("\nRanking (smallest to largest variance):")

models = [
    ("GPT-4o-mini", gpt_sd),
    ("Gemini-2.5-Flash", gemini_sd),
    ("LLaMA-4-Scout", llama_sd)
]
models.sort(key=lambda x: x[1])

for i, (name, sd) in enumerate(models, 1):
    print(f"{i}. {name:20s} σ = {sd:.4f}")