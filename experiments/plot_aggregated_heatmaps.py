import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

df = pd.read_excel('aggregated_qwk_results.xlsx')

# Strategy order and labels matching the PDFs
STRATEGY_ORDER = ['Inductive', 'Deductive', 'Abductive', 'Ind-Ded', 'Ind-Abd', 'Ded-Abd']
STRATEGY_LABELS = ['Ind', 'Ded', 'Abd', 'Ind+Ded', 'Ind+Abd', 'Ded+Abd']
MODEL_ORDER = ['GPT-4o-mini', 'Gemini-2.5-Flash', 'LLaMA-4-Scout']

# ── AES datasets (QWK) ───────────────────────────────────────────────────────
AES_DATASETS = {
    'ASAP-AES':              'ASAP-AES',
    'ASAP2':                 'ASAP2',
    'ASAP_plus_plus':        'ASAP++',
    'persuade_2':            'Persuade-2',
    'Ielts_Writing_Dataset': 'IELTS_Writing_Dataset',
    'Ielts_Writing_Task_2':  'IELTS_Writing_Task_2_Dataset',
}

# ── ASAG numeric datasets (QWK) ──────────────────────────────────────────────
ASAG_NUM_DATASETS = {
    'ASAP-SAS':             'ASAP-SAS',
    'CSEE':                 'CSEE',
    'Mohlar':               'Mohlar',
    'Regrading_Dataset_J2C':'Regrading_J2C',
    'Rice_Chem':            'Rice_Chem',
    'OS_Dataset':           'OS_Dataset',
}

# ── ASAG categorical datasets (F1) ──────────────────────────────────────────
ASAG_CAT_DATASETS = {
    'BEEtlE_2way':      'BEEtlE_2way',
    'BEEtlE_3way':      'BEEtlE_3way',
    'SciEntSBank_2way': 'SciEntSBank_2way',
    'SciEntSBank_3way': 'SciEntSBank_3way',
}

def make_pivot(data, ds_key, metric):
    sub = data[(data['dataset'] == ds_key) & (data['metric'] == metric)]
    pivot = sub.pivot(index='model', columns='strategy', values='value')
    pivot = pivot.reindex(index=MODEL_ORDER, columns=STRATEGY_ORDER)
    pivot.columns = STRATEGY_LABELS
    return pivot

def plot_heatmap_grid(dataset_dict, metric, title, filename, vmin, vmax, cmap, fmt='.2f'):
    datasets = list(dataset_dict.items())
    n = len(datasets)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = axes.flatten()

    cbar_label = 'Quadratic Weighted Kappa (QWK)' if metric == 'QWK' else 'F1 Score'
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for i, (ds_key, ds_label) in enumerate(datasets):
        ax = axes[i]
        pivot = make_pivot(df, ds_key, metric)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            annot=True,
            fmt=fmt,
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

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Shared colorbar on right
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=11, rotation=270, labelpad=18)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {filename}')

# ── Plot AES ─────────────────────────────────────────────────────────────────
plot_heatmap_grid(
    AES_DATASETS, 'QWK',
    title='AES — Model-wise QWK Across Reasoning Strategies per Dataset (3-Call Mean)',
    filename='aggregated_AES_heatmap.pdf',
    vmin=0.1, vmax=1.0, cmap='YlGnBu'
)

# ── Plot ASAG numeric ────────────────────────────────────────────────────────
plot_heatmap_grid(
    ASAG_NUM_DATASETS, 'QWK',
    title='ASAG — Model-wise QWK Across Reasoning Strategies per Dataset (3-Call Mean)',
    filename='aggregated_ASAG_QWK_heatmap.pdf',
    vmin=0.1, vmax=1.0, cmap='YlGnBu'
)

# ── Plot ASAG categorical ────────────────────────────────────────────────────
plot_heatmap_grid(
    ASAG_CAT_DATASETS, 'F1',
    title='ASAG Classification — F1 Score Performance (3-Call Mean)',
    filename='aggregated_ASAG_F1_heatmap.pdf',
    vmin=0.3, vmax=0.8, cmap='YlGnBu'
)