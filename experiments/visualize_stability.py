import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── MODEL COLORS ──────────────────────────────────────────────────────────────
COLORS = {
    "GPT-4o-mini":      "#2563EB",  # Blue
    "Gemini-2.5-Flash": "#16A34A",  # Green
    "LLaMA-4-Scout":    "#DC2626",  # Red
}

# ── DATA (pooled_overall values from all 18 JSONs) ────────────────────────────
strategies = ["Abductive", "Deductive", "Inductive", "Ded-Abd", "Ind-Ded", "Ind-Abd"]

# Numeric: pooled mean_std (lower is better)
gpt_num = [0.05147, 0.04730, 0.05122, 0.03170, 0.05324, 0.05323]
gem_num = [0.09824, 0.09360, 0.09308, 0.06600, 0.06482, 0.07953]
lla_num = [0.33739, 0.29689, 0.33060, 0.39096, 0.18584, 0.36368]

# Categorical: pooled mean_agreement as % (higher is better)
gpt_cat = [0.98858*100, 0.98735*100, 0.98925*100, 0.98623*100, 0.98909*100, 0.98668*100]
gem_cat = [0.99037*100, 0.98679*100, 0.99049*100, 0.99205*100, 0.99317*100, 0.99306*100]
lla_cat = [0.97011*100, 0.96709*100, 0.97649*100, 0.97079*100, 0.96855*100, 0.96463*100]

# ── SHARED SETTINGS ───────────────────────────────────────────────────────────
x        = np.arange(len(strategies))
w        = 0.25
offsets  = [-w, 0, w]
models   = ["GPT-4o-mini", "Gemini-2.5-Flash", "LLaMA-4-Scout"]
num_data = [gpt_num, gem_num, lla_num]
cat_data = [gpt_cat, gem_cat, lla_cat]

legend_handles = [mpatches.Patch(color=COLORS[m], label=m) for m in models]

def draw_panel(ax, data_list, ylabel, title, letter, fmt):
    for i, (model, data) in enumerate(zip(models, data_list)):
        bars = ax.bar(x + offsets[i], data, width=w - 0.02,
                      color=COLORS[model], alpha=0.88, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (0.002 if fmt == 'f' else 0.05),
                    f"{h:{fmt}}", ha='center', va='bottom',
                    fontsize=7, color='#333333')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"({letter}) {title}", fontsize=11, fontweight='bold', pad=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)

def make_legend_panel(ax, title):
    ax.set_axis_off()
    ax.text(0.5, 0.88, title,
            ha='center', va='top', fontsize=13, fontweight='bold',
            transform=ax.transAxes)
    positions = [0.18, 0.50, 0.82]
    for pos, model in zip(positions, models):
        ax.add_patch(mpatches.FancyBboxPatch(
            (pos - 0.10, 0.12), 0.20, 0.42,
            boxstyle="round,pad=0.02",
            facecolor=COLORS[model], alpha=0.88,
            transform=ax.transAxes, clip_on=False))
        ax.text(pos, 0.33, model,
                ha='center', va='center', fontsize=11,
                fontweight='bold', color='white',
                transform=ax.transAxes)

# ── FIGURE 1: Numeric Stability ───────────────────────────────────────────────
fig1, (ax_leg1, ax1) = plt.subplots(2, 1, figsize=(11, 7),
                                     gridspec_kw={'height_ratios': [1, 3.5]})
fig1.patch.set_facecolor("white")
make_legend_panel(ax_leg1, "Numeric Prediction Stability")
draw_panel(ax1, num_data,
           "Pooled Mean Std (↓ better)",
           "Numeric Stability", "a", ".3f")
plt.tight_layout(h_pad=2.0)
fig1.savefig("numeric_stability.png", dpi=150, bbox_inches='tight')
fig1.savefig("numeric_stability.pdf", bbox_inches='tight')

# ── FIGURE 2: Categorical Stability ──────────────────────────────────────────
fig2, (ax_leg2, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                     gridspec_kw={'height_ratios': [1, 3.5]})
fig2.patch.set_facecolor("white")
make_legend_panel(ax_leg2, "Categorical Prediction Stability")
draw_panel(ax2, cat_data,
           "Mean Agreement (%) (↑ better)",
           "Categorical Stability", "b", ".2f")
ax2.set_ylim(95, 100)
plt.tight_layout(h_pad=2.0)
fig2.savefig("categorical_stability.png", dpi=150, bbox_inches='tight')
fig2.savefig("categorical_stability.pdf", bbox_inches='tight')

print("Saved: numeric_stability.png/.pdf and categorical_stability.png/.pdf")