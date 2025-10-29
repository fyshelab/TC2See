import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import json
import traceback
from sklearn.linear_model import HuberRegressor
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
RDM_dir = dataset_root / "processed/predicted_embeddings_RDMs/"
visual_dir = results_dir / f'pred_embed_rsa/visuals/scatters/exp_vs_pred-trueCLIP_RSA'
visual_dir.mkdir(parents=True, exist_ok=True)

with open(RDM_dir / "per_sub_pred_v_CLIP_rsa_results.json", "r") as f:
    per_sub_rsa = json.load(f)

# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)


def compute_stats(title, values):
    """Return x, y, r, p for one condition"""
    x, y = [], []
    for pid, rsa_val in values.items():
        if pid in expertise_scores:
            x.append(expertise_scores[pid])
            y.append(rsa_val)
    if len(x) > 1:
        r, p = pearsonr(x, y)
    else:
        r, p = float("nan"), float("nan")
    return x, y, r, p


def plot_scatter(ax, x, y, r, p, p_corr, title):
    """Make one scatter plot with regression line and stats"""
    ax.scatter(x, y)
    if len(x) > 1:
        slope, intercept, _, _, _ = linregress(x, y)
        xs = [min(x), max(x)]
        ys = [slope * xi + intercept for xi in xs]
        ax.plot(xs, ys, "r--", linewidth=1)

    ax.set_xlim(30, 95)
    if any(roi in title for roi in ["V1", "V2", "V3", "V4"]):
        ax.set_ylim(-0.05, 0.25)
    else:
        ax.set_ylim(-0.05, 0.1)
    ax.set_xlabel("Expertise Score")
    ax.set_ylabel("RSA Value (predicted CLIP  ←→  true CLIP)")
    ax.set_title(f"{title}\nr={r:.2f}, p={p:.3g}, p_adj={p_corr:.3g}")


### ---- Figure 1: experiment-8.3 embeddings ----
# embedding_plots = []
# for model, model_data in per_sub_rsa["experiment-8.3"].items():
#     values = model_data["embedding"]
#     title = f"{model} / embedding"
#     x, y, r, p = compute_stats(title, values)
#     embedding_plots.append((title, x, y, r, p))

# # FDR correction
# pvals = [p for (_, _, _, _, p) in embedding_plots]
# _, corrected_pvals, _, _ = multipletests(pvals, method="fdr_bh")

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# for ax, (title, x, y, r, p), p_corr in zip(axes, embedding_plots, corrected_pvals):
#     plot_scatter(ax, x, y, r, p, p_corr, title)

# plt.tight_layout()
# plt.savefig(visual_dir / 'experiment-8.3_whole_brain.png', dpi=300)



### ---- Figure 2: experiment-8.4 ----
exp84_plots = []
for model in ["ViT-B=16-features_large"]:
    for region, values in per_sub_rsa["experiment-8.4"][model].items():
        title = f"{model} / {region}"
        x, y, r, p = compute_stats(title, values)
        exp84_plots.append((title, x, y, r, p))

# BH Correction
pvals = [p for (_, _, _, _, p) in exp84_plots]
_, corrected_pvals, _, _ = multipletests(pvals, method="fdr_bh")

# Determine number of subplots dynamically
n_plots = len(exp84_plots)
n_cols = 4  # keep 4 columns for readability
n_rows = int(np.ceil(n_plots / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = np.array(axes).flatten()

for ax, (title, x, y, r, p), p_corr in zip(axes, exp84_plots, corrected_pvals):
    plot_scatter(ax, x, y, r, p, p_corr, title)

# Hide any unused subplots
for ax in axes[len(exp84_plots):]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(visual_dir / 'experiment-8.4_expertise_ROIs.png', dpi=300)



# Load grouped averages
with open(RDM_dir / "grouped_avg_pred_v_CLIP_rsa_results.json", "r") as f:
    grouped_rsa = json.load(f)

# Categories in order
categories = [
    ("experiment-8.4", "ViT-B=16-features_large", "A1"),
    ("experiment-8.4", "ViT-B=16-features_large", "FFC"),
    ("experiment-8.4", "ViT-B=16-features_large", "LO1"),
    ("experiment-8.4", "ViT-B=16-features_large", "LO2"),
    ("experiment-8.4", "ViT-B=16-features_large", "PIT"),
    ("experiment-8.4", "ViT-B=16-features_large", "Pir"),
    ("experiment-8.4", "ViT-B=16-features_large", "V1"),
    ("experiment-8.4", "ViT-B=16-features_large", "V2"),
    ("experiment-8.4", "ViT-B=16-features_large", "V3"),
    ("experiment-8.4", "ViT-B=16-features_large", "V4"),
    ("experiment-8.4", "ViT-B=16-features_large", "V7"),
    ("experiment-8.4", "ViT-B=16-features_large", "V8"),
    ("experiment-8.4", "ViT-B=16-features_large", "VVC"),
]


# Labels for x-axis
labels = [
    "ViT-B=16 / A1",
    "ViT-B=16 / FFC",
    "ViT-B=16 / LO1",
    "ViT-B=16 / LO2",
    "ViT-B=16 / PIT",
    "ViT-B=16 / Pir",
    "ViT-B=16 / V1",
    "ViT-B=16 / V2",
    "ViT-B=16 / V3",
    "ViT-B=16 / V4",
    "ViT-B=16 / V7",
    "ViT-B=16 / V8",
    "ViT-B=16 / VVC",
]

expertise_groups = ["low", "mid", "high"]
colors = ["#6C7A8A", "#1b2a3b", "#7da4a8"] 

# Collect data for each expertise group
values_by_group = {g: [] for g in expertise_groups}

for exp, model, region in categories:
    for group in expertise_groups:
        score = grouped_rsa[group][exp][model][region]
        values_by_group[group].append(score)

# Plot
x = np.arange(len(labels))  # positions of groups
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(18, 8))

for i, group in enumerate(expertise_groups):
    ax.bar(x + i*width - width, values_by_group[group], width, label=group, color=colors[i])


ax.set_ylabel("RSA Value")
ax.set_title("Averaged RSA Values (predicted CLIP  ←→  true CLIP) Within Expertise Group - for Each Model/Region")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(title="Expertise Level")

plt.tight_layout()
plt.savefig(visual_dir / 'grouped_bar_RSA_all.png', dpi=300)


# Load between-group RSA results
with open(RDM_dir / "between_group_avg_rsa_results.json", "r") as f:
    between_rsa = json.load(f)

# Group pairs to compare
group_pairs = {
    "low_vs_mid":  "Low vs Mid",
    "low_vs_high": "Low vs High",
    "mid_vs_high": "Mid vs High"
}
pair_colors = ["#6C7A8A", "#1b2a3b", "#7da4a8"]

# Collect values for each group pair
values_by_pair = {pair: [] for pair in group_pairs}

for exp, model, region in categories:
    for pair in group_pairs.keys():
        score = between_rsa[exp][model][region][pair]
        values_by_pair[pair].append(score)

# Plot
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(18, 8))

for i, (pair, pretty_label) in enumerate(group_pairs.items()):
    ax.bar(x + i*width - width, values_by_pair[pair], width,
           label=pretty_label, color=pair_colors[i])

ax.set_ylabel("RSA Value")
ax.set_title("Between-Expertise-Group RSA Values (predicted CLIP Group i  ←→  predicted CLIP Group j)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(title="Group Comparison")

plt.tight_layout()
plt.savefig(visual_dir / 'between_group_bar_RSA.png', dpi=300)