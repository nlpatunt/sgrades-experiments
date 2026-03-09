import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
#   AES DATA
# =========================
aes_data = {
    "ASAP-AES": {
        "GPT-4o-mini":      [0.9207, 0.9523, 0.9542, 0.9151, 0.9500, 0.9159],
        "Gemini-2.5-Flash": [0.6046, 0.8835, 0.8751, 0.7177, 0.8182, 0.8112],
        "LLaMA-4-Scout":    [0.6209, 0.8620, 0.9112, 0.8863, 0.9371, 0.9189],
    },
    "ASAP2": {
        "GPT-4o-mini":      [0.2889, 0.1650, 0.1595, 0.2827, 0.2500, 0.1051],
        "Gemini-2.5-Flash": [0.3338, 0.2389, 0.2009, 0.3806, 0.3455, 0.1824],
        "LLaMA-4-Scout":    [0.4128, 0.1456, 0.1761, 0.3355, 0.1937, 0.1557],
    },
    "ASAP++": {
        "GPT-4o-mini":      [0.1914, 0.1856, 0.1914, 0.1931, 0.2100, 0.1514],
        "Gemini-2.5-Flash": [0.3033, 0.1704, 0.1407, 0.2864, 0.2303, 0.1379],
        "LLaMA-4-Scout":    [0.2423, 0.2233, 0.2198, 0.2321, 0.2292, 0.2390],
    },
    "Persuade-2": {
        "GPT-4o-mini":      [0.6727, 0.5304, 0.4856, 0.6667, 0.6700, 0.5410],
        "Gemini-2.5-Flash": [0.7754, 0.7003, 0.6805, 0.7721, 0.7671, 0.5875],
        "LLaMA-4-Scout":    [0.2962, 0.4688, 0.3814, 0.2606, 0.3181, 0.3733],
    },
    "IELTS Task 1": {
        "GPT-4o-mini":      [0.4418, 0.2857, 0.2820, 0.3529, 0.4100, 0.2667],
        "Gemini-2.5-Flash": [0.4855, 0.3691, 0.4346, 0.4737, 0.4419, 0.2168],
        "LLaMA-4-Scout":    [0.3662, 0.5455, 0.3144, 0.2730, 0.6429, 0.4961],
    },
    "IELTS Task 2": {
        "GPT-4o-mini":      [0.1901, 0.2340, 0.2374, 0.1856, 0.1800, 0.1940],
        "Gemini-2.5-Flash": [0.2030, 0.1876, 0.1659, 0.1983, 0.1903, 0.1394],
        "LLaMA-4-Scout":    [0.1790, 0.2202, 0.2219, 0.1162, 0.2584, 0.2519],
    },
}

# =========================
#   ASAG REGRESSION DATA
# =========================
asag_reg_data = {
    "ASAP-SAS": {
        "GPT-4o-mini":      [0.6431, 0.5512, 0.5973, 0.6558, 0.6459, 0.4914],
        "Gemini-2.5-Flash": [0.6025, 0.6470, 0.6262, 0.6020, 0.5953, 0.6520],
        "LLaMA-4-Scout":    [0.5063, 0.6077, 0.4991, 0.4758, 0.5030, 0.3565],
    },
    "CSEE": {
        "GPT-4o-mini":      [0.5226, 0.5112, 0.4764, 0.5050, 0.5500, 0.4254],
        "Gemini-2.5-Flash": [0.6226, 0.5053, 0.5682, 0.6419, 0.6433, 0.3341],
        "LLaMA-4-Scout":    [0.6216, 0.5902, 0.6215, 0.6133, 0.6600, 0.5823],
    },
    "Mohlar": {
        "GPT-4o-mini":      [0.0281, 0.0172, 0.0234, 0.0100, -0.0100, -0.0064],
        "Gemini-2.5-Flash": [0.1259, 0.1452, 0.1814, 0.2556, 0.1872, 0.0710],
        "LLaMA-4-Scout":    [0.0234, 0.0700, 0.0138, -0.0375, 0.1335, -0.0565],
    },
    "Regrading J2C": {
        "GPT-4o-mini":      [0.2463, 0.2746, 0.1805, 0.2553, 0.2393, 0.3569],
        "Gemini-2.5-Flash": [-0.2000, 0.2579, 0.1897, 0.1741, 0.2178, 0.1807],
        "LLaMA-4-Scout":    [0.2000, 0.4400, 0.3054, 0.3204, 0.3950, 0.4100],
    },
    "OS Dataset": {
        "GPT-4o-mini":      [0.0420, -0.0238, -0.0434, -0.0462, 0.0020, -0.0954],
        "Gemini-2.5-Flash": [-0.0382, 0.0088, -0.0400, -0.0200, 0.0080, -0.1100],
        "LLaMA-4-Scout":    [-0.0193, -0.0387, 0.1400, 0.0900, 0.1727, 0.0839],
    },
    "Rice Chem": {
        "GPT-4o-mini":      [0.5110, 0.1944, 0.1362, 0.4864, 0.4406, 0.2444],
        "Gemini-2.5-Flash": [0.6444, 0.3911, 0.3106, 0.5608, 0.5680, 0.4406],
        "LLaMA-4-Scout":    [0.5220, 0.3400, 0.2283, 0.3696, 0.4469, 0.2762],
    },
}

models = ["GPT-4o-mini", "Gemini-2.5-Flash", "LLaMA-4-Scout"]

# =========================
#   COMPUTE SIGMA
# =========================
def compute_sigma_table(data, task_name):
    rows = []
    for dataset, model_scores in data.items():
        row = {"Task": task_name, "Dataset": dataset}
        for model in models:
            sigma = np.std(model_scores[model])
            row[model] = round(sigma, 4)
        rows.append(row)
    return rows

aes_rows = compute_sigma_table(aes_data, "AES")
asag_rows = compute_sigma_table(asag_reg_data, "ASAG Regression")
all_rows = aes_rows + asag_rows

df = pd.DataFrame(all_rows)

# Print the table
print("\n" + "="*75)
print("STRATEGY VARIANCE (σ) PER DATASET AND MODEL")
print("σ = std across 6 reasoning strategies: Ind, Ded, Abd, Ind+Abd, Ind+Ded, Ded+Abd")
print("="*75)
print(f"{'Task':<15} {'Dataset':<20} {'GPT-4o-mini':>12} {'Gemini-2.5-Flash':>17} {'LLaMA-4-Scout':>14}")
print("-"*75)

for task in ["AES", "ASAG Regression"]:
    task_rows = df[df["Task"] == task]
    for _, row in task_rows.iterrows():
        print(f"{row['Task']:<15} {row['Dataset']:<20} {row['GPT-4o-mini']:>12.4f} {row['Gemini-2.5-Flash']:>17.4f} {row['LLaMA-4-Scout']:>14.4f}")
    # Average per task
    avg = task_rows[models].mean()
    print(f"{'':15} {'Average σ':<20} {avg['GPT-4o-mini']:>12.4f} {avg['Gemini-2.5-Flash']:>17.4f} {avg['LLaMA-4-Scout']:>14.4f}")
    print("-"*75)

# =========================
#   PRODUCE FIGURE
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.set_theme(style="whitegrid")

colors = {"GPT-4o-mini": "#4C72B0", "Gemini-2.5-Flash": "#DD8452", "LLaMA-4-Scout": "#55A868"}

for ax, task in zip(axes, ["AES", "ASAG Regression"]):
    task_df = df[df["Task"] == task].set_index("Dataset")[models]
    
    x = np.arange(len(task_df))
    width = 0.25

    for i, model in enumerate(models):
        bars = ax.bar(x + i*width, task_df[model], width,
                      label=model, color=colors[model], alpha=0.85, edgecolor="white")
        # Add value labels on bars
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.002,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xticks(x + width)
    ax.set_xticklabels(task_df.index, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel("Strategy Variance (σ)", fontsize=10)
    ax.set_title(f"{task} — Strategy Variance per Dataset", fontsize=11, weight="bold")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.legend(fontsize=9)

plt.suptitle(
    "Strategy Sensitivity: σ Across 6 Reasoning Strategies per Dataset",
    fontsize=13, weight="bold", y=1.02
)

plt.tight_layout()
output_path = "/home/ts1506.UNT/Desktop/Work/strategy_variance_figure.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved to: {output_path}")
plt.show()