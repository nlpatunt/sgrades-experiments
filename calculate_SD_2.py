import numpy as np

# ASAG Regression Data extracted from heatmap (Ind, Ded, Abd, Ind+Abd, Ind+Ded, Ded+Abd)
# Source: ASAG_regression_heatmaps_final.pdf

gpt_4o_mini_asag = {
    'ASAP-SAS': [0.64, 0.55, 0.60, 0.66, 0.65, 0.49],
    'CSEE': [0.52, 0.51, 0.48, 0.51, 0.55, 0.43],
    'Mohlar': [0.03, 0.02, 0.02, 0.01, -0.01, -0.01],
    'Regrading_J2C': [0.25, 0.27, 0.18, 0.26, 0.24, 0.36],
    'OS_Dataset': [0.04, -0.02, -0.04, -0.05, 0.00, -0.10],
    'Rice_Chem': [0.51, 0.19, 0.14, 0.49, 0.44, 0.24]
}

gemini_asag = {
    'ASAP-SAS': [0.60, 0.65, 0.63, 0.60, 0.60, 0.65],
    'CSEE': [0.62, 0.51, 0.57, 0.64, 0.64, 0.33],
    'Mohlar': [0.13, 0.15, 0.18, 0.26, 0.19, 0.07],
    'Regrading_J2C': [-0.20, 0.26, 0.19, 0.17, 0.22, 0.18],
    'OS_Dataset': [-0.04, 0.01, -0.04, -0.02, 0.01, -0.11],
    'Rice_Chem': [0.64, 0.39, 0.31, 0.56, 0.57, 0.44]
}

llama_asag = {
    'ASAP-SAS': [0.51, 0.61, 0.50, 0.48, 0.50, 0.36],
    'CSEE': [0.62, 0.59, 0.62, 0.61, 0.66, 0.58],
    'Mohlar': [0.02, 0.07, 0.01, -0.04, 0.13, -0.06],
    'Regrading_J2C': [0.20, 0.44, 0.31, 0.32, 0.40, 0.41],
    'OS_Dataset': [-0.02, -0.04, 0.14, 0.09, 0.17, 0.08],
    'Rice_Chem': [0.52, 0.34, 0.23, 0.37, 0.45, 0.28]
}


def calculate_strategy_variance(model_data, model_name):
    """
    Calculates variance across strategies for each dataset,
    then averages to get overall model strategy variance.
    """
    print(f"\n{'='*60}")
    print(f"{model_name} — Strategy Variance (ASAG Regression)")
    print(f"{'='*60}")

    dataset_sds = []

    for dataset, scores in model_data.items():
        sd = np.std(scores, ddof=1)  # sample standard deviation
        dataset_sds.append(sd)
        print(f"{dataset:20s}: {scores}")
        print(f"{' '*20}Mean={np.mean(scores):.3f}, SD={sd:.4f}")

    avg_sd = np.mean(dataset_sds)

    print(f"\n{model_name} Average Variance: σ ≈ {avg_sd:.4f}")
    print(f"Individual SDs: {[f'{sd:.4f}' for sd in dataset_sds]}")
    
    return avg_sd


# Run for each model
gpt_sd = calculate_strategy_variance(gpt_4o_mini_asag, "GPT-4o-mini")
gemini_sd = calculate_strategy_variance(gemini_asag, "Gemini-2.5-Flash")
llama_sd = calculate_strategy_variance(llama_asag, "LLaMA-4-Scout")

print("\n" + "="*60)
print("ASAG Regression Strategy Variance — Summary")
print("="*60)

models = [
    ("GPT-4o-mini", gpt_sd),
    ("Gemini-2.5-Flash", gemini_sd),
    ("LLaMA-4-Scout", llama_sd),
]

models.sort(key=lambda x: x[1])

for i, (name, sd) in enumerate(models, 1):
    print(f"{i}. {name:20s} σ = {sd:.4f}")
