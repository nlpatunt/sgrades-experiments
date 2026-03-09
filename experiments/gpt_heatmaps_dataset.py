import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── FILE PATHS ────────────────────────────────────────────────────────────────
files = {
    "Abductive": "gpt4o/stability_summary_abductive_3call_predictions_1773036604.xlsx",
    "Deductive": "gpt4o/stability_summary_deductive_3call_predictions_1773036605.xlsx",
    "Inductive": "gpt4o/stability_summary_inductive_3call_predictions_1773036606.xlsx",
    "Ded-Abd":   "gpt4o/stability_summary_deductive_abductive_3call_predictions_1773036607.xlsx",
    "Ind-Ded":   "gpt4o/stability_summary_inductive_deductive_3call_predictions_1773036608.xlsx",
    "Ind-Abd":   "gpt4o/stability_summary_inductive_abductive_3call_predictions_1773036609.xlsx",
}

# AES vs ASAG split
AES_DATASETS  = ["D_ASAP-AES", "D_ASAP_plus_plus", "D_ASAP2", "D_persuade_2",
                 "D_Ielts_Writing_Dataset", "D_Ielts_Writing_Task_2_Dataset"]
ASAG_DATASETS = ["D_ASAP-SAS", "D_Regrading_Dataset_J2C", "D_CSEE",
                 "D_Mohlar", "D_OS_Dataset", "D_Rice_Chem"]

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
num_data = {}
cat_data = {}

for strat, fpath in files.items():
    df = pd.read_excel(fpath, sheet_name="Main")
    num_data[strat] = df[df["type"] == "numeric"].set_index("dataset")["mean_std"]
    cat_data[strat] = df[df["type"] == "categorical"].set_index("dataset")["mean_agreement"]

num_df = pd.DataFrame(num_data)
cat_df = pd.DataFrame(cat_data) * 100

# Drop pooled overall rows
num_df = num_df[~num_df.index.str.startswith("POOLED")]
cat_df = cat_df[~cat_df.index.str.startswith("POOLED")]

# Split numeric into AES and ASAG
aes_df  = num_df[num_df.index.isin(AES_DATASETS)]
asag_df = num_df[num_df.index.isin(ASAG_DATASETS)]

# Clean names for display
def clean_name(n):
    return n.replace("D_", "").replace("_", " ")

aes_df.index  = [clean_name(x) for x in aes_df.index]
asag_df.index = [clean_name(x) for x in asag_df.index]
cat_df.index  = [clean_name(x) for x in cat_df.index]

# ── PLOT: 3 panels ────────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16),
                                     gridspec_kw={"height_ratios": [len(aes_df),
                                                                     len(asag_df),
                                                                     len(cat_df)]})
fig.patch.set_facecolor("white")
fig.suptitle("GPT-4o-mini — Per-Dataset Prediction Stability",
             fontsize=13, fontweight="bold", y=1.01)

# (a) AES — Reds
sns.heatmap(aes_df, ax=ax1, cmap="Reds", annot=True, fmt=".2f",
            linewidths=0.4, linecolor="#dddddd",
            cbar_kws={"label": "Mean Std (↓ better)", "shrink": 0.8},
            annot_kws={"size": 8, "weight": "bold"})
ax1.set_title("(a) AES Datasets — Mean Std", fontsize=11, fontweight="bold", pad=8)
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.tick_params(axis="x", labelsize=9, rotation=0)
ax1.tick_params(axis="y", labelsize=8.5, rotation=0)

# (b) ASAG — Reds (independent colormap)
sns.heatmap(asag_df, ax=ax2, cmap="Reds", annot=True, fmt=".2f",
            linewidths=0.4, linecolor="#dddddd",
            cbar_kws={"label": "Mean Std (↓ better)", "shrink": 0.8},
            annot_kws={"size": 8, "weight": "bold"})
ax2.set_title("(b) ASAG Datasets — Mean Std", fontsize=11, fontweight="bold", pad=8)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.tick_params(axis="x", labelsize=9, rotation=0)
ax2.tick_params(axis="y", labelsize=8.5, rotation=0)

# (c) Categorical — Blues
sns.heatmap(cat_df, ax=ax3, cmap="Blues", annot=True, fmt=".1f",
            linewidths=0.4, linecolor="#dddddd",
            cbar_kws={"label": "Agreement % (↑ better)", "shrink": 0.8},
            annot_kws={"size": 8},
            vmin=93, vmax=100)
ax3.set_title("(c) Categorical Datasets — Mean Agreement (%)", fontsize=11, fontweight="bold", pad=8)
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.tick_params(axis="x", labelsize=9, rotation=0)
ax3.tick_params(axis="y", labelsize=8.5, rotation=0)

plt.tight_layout(h_pad=3)
plt.savefig("GPT4o_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig("GPT4o_heatmap.pdf", bbox_inches="tight")
print("Saved: GPT4o_heatmap.png and GPT4o_heatmap.pdf")