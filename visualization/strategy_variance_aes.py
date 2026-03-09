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

models = ["GPT-4o-mini", "Gemini-2.5-Flash", "LLaMA-4-Scout"]

# =========================
#   COMPUTE SIGMA
# =========================
rows = []
for dataset, model_scores in aes_data.items():
    row = {"Dataset": dataset}
    for model in models:
        row[model] = round(np.std(model_scores[model]), 4)
    rows.append(row)

df = pd.DataFrame(rows).set_index("Dataset")

# Print table
print("\n" + "="*65)
print("AES STRATEGY VARIANCE (σ)")
print("σ = std across 6 strategies: Ind, Ded, Abd, Ind+Abd, Ind+Ded, Ded+Abd")
print("="*65)
print(f"{'Dataset':<20} {'GPT-4o-mini':>12} {'Gemini-2.5-Flash':>17} {'LLaMA-4-Scout':>14}")
print("-"*65)
for dataset, row in df.iterrows():
    print(f"{dataset:<20} {row['GPT-4o-mini']:>12.4f} {row['Gemini-2.5-Flash']:>17.4f} {row['LLaMA-4-Scout']:>14.4f}")
avg = df.mean()
print("-"*65)
print(f"{'Average σ':<20} {avg['GPT-4o-mini']:>12.4f} {avg['Gemini-2.5-Flash']:>17.4f} {avg['LLaMA-4-Scout']:>14.4f}")

# =========================
#   FIGURE
# =========================
sns.set_theme(style="whitegrid")
colors = {"GPT-4o-mini": "#4C72B0", "Gemini-2.5-Flash": "#DD8452", "LLaMA-4-Scout": "#55A868"}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df))
width = 0.25

for i, model in enumerate(models):
    bars = ax.bar(x + i*width, df[model], width,
                  label=model, color=colors[model], alpha=0.85, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.001,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

ax.set_xticks(x + width)
ax.set_xticklabels(df.index, rotation=35, ha='right', fontsize=10)
ax.set_ylabel("Strategy Variance (σ)", fontsize=11)
ax.set_title("AES — Strategy Variance (σ) Across 6 Reasoning Strategies per Dataset",
             fontsize=12, weight="bold")
ax.set_ylim(0, df.values.max() * 1.35)
ax.legend(fontsize=10)

plt.tight_layout()
output_path = "/home/ts1506.UNT/Desktop/Work/strategy_variance_AES.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved to: {output_path}")
plt.show()