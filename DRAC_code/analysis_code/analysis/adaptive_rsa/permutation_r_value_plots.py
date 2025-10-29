from random import sample
import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
import json
import traceback
from sklearn.linear_model import HuberRegressor
from matplotlib.ticker import FormatStrFormatter
import tqdm as tqdm

dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
sigma_val = 5
data_dir = results_dir / f"gaussian_rsa/sigma_{sigma_val}/data"
sigma_dir = results_dir / f'gaussian_rsa/sigma_{sigma_val}'

correlation_type = "spearman"
added_description = ""  # ""  or  "_no_sub_8"
ROIs = ["V1", "V2", "V3", "V4", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]


permutation_in_path_r = data_dir / f'sigma_{sigma_val}_{correlation_type}{added_description}_weighted_permutations_r.json'
with open(permutation_in_path_r, 'r') as f:
    permutation_results_r = json.load(f)

permutation_in_path_p = data_dir / f'sigma_{sigma_val}_{correlation_type}{added_description}_weighted_permutations_p.json'
with open(permutation_in_path_p, 'r') as f:
    permutation_results_p = json.load(f)
expertise_vs_rsa_corr_path = sigma_dir / f'data/{added_description}sigma_{sigma_val}_expertise_vs_rsa_correlation_similarity.json'
with open(expertise_vs_rsa_corr_path, 'r') as f:
    expertise_vs_rsa_corrs = json.load(f)


roi_names = list(permutation_results_r.keys())
averages_from_perm = np.array([
    np.mean(permutation_results_r[roi]) for roi in roi_names
])
std_from_perm = np.array([
    np.std(permutation_results_r[roi]) for roi in roi_names
])
averages = np.array([
    expertise_vs_rsa_corrs[roi][f"{correlation_type}_r"] for roi in roi_names
])
x_positions = np.arange(len(roi_names))

plt.figure(figsize=(10, 5))

plt.fill_between(
    x_positions,
    averages_from_perm,
    averages_from_perm + std_from_perm,
    color='lightblue',
    alpha=0.4,
    label='+1 SD (Permutation)'
)

# Smooth blue line for permutation means
plt.plot(
    x_positions,
    averages_from_perm,
    color='#4a90e2',  # Softer blue
    linewidth=2,
    label='Avg Permutation Correlation',
    zorder=3
)

# Greenish dots for actual PARSA results
plt.scatter(
    x_positions,
    averages,
    color='#2ca02c',
    edgecolor='black',
    linewidth=0.5,
    zorder=4,
    label='PARSA'
)

plt.xlabel('ROI')
plt.ylabel(f'Correlation ({correlation_type})')
plt.title('Avg Correlation Values From Permutations vs PARSA Correlation')
plt.xticks(x_positions, roi_names, rotation=45)
plt.ylim(-0.1, 0.7)
plt.legend()
plt.tight_layout()

plt.savefig(sigma_dir / f'visuals/permutation_test_plots/{correlation_type}_{added_description}sigma_{sigma_val}_avg_perm.png', dpi=300)
