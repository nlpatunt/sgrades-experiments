import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
#   ASAG REGRESSION DATA
# =========================
asag_data = {
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
rows = []
for dataset, model_scores in asag_data.items():
    row = {"Dataset": dataset}
    for model in models:
        row[model] = round(np.std(model_scores[model]), 4)
    rows.append(row)

df = pd.DataFrame(rows).set_index("Dataset")

# Print table
print("\n" + "="*65)
print("ASAG REGRESSION STRATEGY VARIANCE (σ)")
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
ax.set_title("ASAG Regression — Strategy Variance (σ) Across 6 Reasoning Strategies per Dataset",
             fontsize=12, weight="bold")
ax.set_ylim(0, df.values.max() * 1.35)
ax.legend(fontsize=10)

plt.tight_layout()
output_path = "/home/ts1506.UNT/Desktop/Work/strategy_variance_ASAG.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved to: {output_path}")
plt.show()