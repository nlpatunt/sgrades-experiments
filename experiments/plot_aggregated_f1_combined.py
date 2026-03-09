import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

df = pd.read_excel('aggregated_qwk_results.xlsx')
df_f1 = df[df['metric'] == 'F1'].copy()

# Average 2way and 3way into single BEEtlE and SciEntSBank rows
df_f1['dataset'] = df_f1['dataset'].str.replace('_2way', '').str.replace('_3way', '')
df_f1 = df_f1.groupby(['model', 'strategy', 'dataset'])['value'].mean().reset_index()

STRATEGY_ORDER  = ['Inductive', 'Deductive', 'Abductive', 'Ind-Ded', 'Ind-Abd', 'Ded-Abd']
STRATEGY_LABELS = ['Ind', 'Ded', 'Abd', 'Ind+Ded', 'Ind+Abd', 'Ded+Abd']
MODEL_ORDER     = ['GPT-4o-mini', 'Gemini-2.5-Flash', 'LLaMA-4-Scout']

DATASETS = {
    'BEEtlE':      'BEEtlE',
    'SciEntSBank': 'SciEntSBank',
}

def make_pivot(data, ds_key):
    sub = data[data['dataset'] == ds_key]
    pivot = sub.pivot(index='model', columns='strategy', values='value')
    pivot = pivot.reindex(index=MODEL_ORDER, columns=STRATEGY_ORDER)
    pivot.columns = STRATEGY_LABELS
    return pivot

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

vmin, vmax = 0.3, 0.8
cmap = 'YlGnBu'
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

for i, (ds_key, ds_label) in enumerate(DATASETS.items()):
    ax = axes[i]
    pivot = make_pivot(df_f1, ds_key)

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        annot=True,
        fmt='.2f',
        annot_kws={"size": 11, "weight": "bold"},
        linewidths=0.5,
        linecolor='white',
        cbar=False,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(ds_label, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=10, rotation=0)
    ax.tick_params(axis='y', labelsize=10, rotation=0)

cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('F1 Score', fontsize=11, rotation=270, labelpad=18)
cbar.ax.tick_params(labelsize=10)

fig.suptitle('ASAG Classification — F1 Score Performance (3-Call Mean)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0, 0, 0.92, 1])
plt.savefig('3call_aggregated_ASAG_F1_combined_heatmap.pdf', dpi=200, bbox_inches='tight')
plt.close()
print('✅ Saved: aggregated_ASAG_F1_combined_heatmap.pdf')