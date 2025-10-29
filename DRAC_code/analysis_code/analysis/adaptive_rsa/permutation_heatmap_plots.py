import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from statsmodels.stats.multitest import multipletests


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

sigma_val = "5"
correlation_type = "pearson"
added_description = "" # "" or "_no_sub_8"
adjust_p_values = False # Multiple comparisons
adjustment_type = 'bh' # bonf or bh
data_dir = results_dir / f"gaussian_rsa/sigma_{sigma_val}/data"
plots_dir = results_dir / f"gaussian_rsa/sigma_{sigma_val}/visuals"
plots_dir.mkdir(parents=True, exist_ok=True)

ROIs = ["V1", "V2", "V3", "V4", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]

try:
    permutation_path = data_dir / f'test_sigma_{sigma_val}_{correlation_type}{added_description}_weighted_permutations_r.json'
    with open(permutation_path, 'r') as f:
        permutation_results = json.load(f)
    permutation_data_available = True
except FileNotFoundError:
    permutation_data_available = False
    print(f"Permutation results not available, skipping p-value analysis...")

# Load the correlation results for this sigma range
sigma_corr_results_path = data_dir / f'sigma_{sigma_val}{added_description}_corr_and_p_for_each_ROI.json'
with open(sigma_corr_results_path, 'r') as f:
    sigma_corr_results = json.load(f)



###############################################
# Get p-values for different ROIs at each sigma 
###############################################


if permutation_data_available:
    p_values = {roi: {sigma_val: None} for roi in ROIs}

    for roi in ROIs:
        observed_r = sigma_corr_results[sigma_val][roi][f'{correlation_type}_r']
        null_distribution = permutation_results[roi]

        p_val = (np.sum(np.array(null_distribution) >= observed_r) + 1) / (len(null_distribution) + 1)
        p_values[roi][sigma_val] = p_val


if adjust_p_values:
    raw_p_list = [p_values[roi][sigma_val] for roi in ROIs]
    method = 'fdr_bh' if adjustment_type == 'bh' else 'bonferroni'
    _, corrected_p_list, _, _ = multipletests(raw_p_list, alpha=0.05, method=method)
    _, corrected_p_list, _, _ = multipletests(raw_p_list, alpha=0.05, method=method)
    p_values = {roi: {sigma_val: corrected_p} for roi, corrected_p in zip(ROIs, corrected_p_list)}
    

##################################
# P-value and correlation heatmaps 
##################################
if permutation_data_available:
    # Create matrices for heatmap
    p_matrix = np.full((len(ROIs), len([1])), np.nan)
    r_matrix = np.full((len(ROIs), len([1])), np.nan)
    for i, roi in enumerate(ROIs):
        if p_values[roi][sigma_val] is not None:
            p_matrix[i, 0] = p_values[roi][sigma_val]
            r_matrix[i, 0] = sigma_corr_results[sigma_val][roi][f'{correlation_type}_r']

    # P-value heatmap
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    im1 = ax1.imshow(p_matrix, cmap='Blues_r', aspect='auto', vmin=0, vmax=0.05)
    ax1.set_xticks(range(len([1])))
    ax1.set_xticklabels(["5"])
    ax1.set_yticks(range(len(ROIs)))
    ax1.set_yticklabels(ROIs)
    ax1.set_xlabel('Sigma Value', fontweight='bold')
    ax1.set_ylabel('ROI', fontweight='bold')
    subject_description = "(no sub 8)" if added_description == "_no_sub_8" else ""
    p_desc = f"Adjusted ({adjustment_type}) " if adjust_p_values == True else ""
    ax1.set_title(f'{p_desc}P-values From Permutation Tests ({correlation_type}) {subject_description}', fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('P-value', fontweight='bold')
    # Add p-values as text and red overlay for non-significant cells
    for i in range(len(ROIs)):
        for j in range(len([1])):
            if not np.isnan(p_matrix[i, j]):
                p_val = p_matrix[i, j]
                ax1.text(j, i, f'{p_val:.5f}', ha='center', va='center',
                        color='white' if p_val < 0.025 else 'black', fontsize=9)
                if p_val > 0.05:
                    ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                color='red', alpha=0.6))
    plt.tight_layout()
    adj = "adj_" if adjust_p_values == True else ""
    plt.savefig(plots_dir / f'test_sigma_{sigma_val}_{correlation_type}_{adjustment_type}{added_description}_{adj}p_heatmap.png', 
                dpi=300, bbox_inches='tight')

    # Correlation heatmap 
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 8))
    im2 = ax2.imshow(r_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax2.set_xticks(range(len([1])))
    ax2.set_xticklabels(["05"])
    ax2.set_yticks(range(len(ROIs)))
    ax2.set_yticklabels(ROIs)
    ax2.set_xlabel('Sigma Value', fontweight='bold')
    ax2.set_ylabel('ROI', fontweight='bold')
    title_description = "(no sub 8)" if added_description == "_no_sub_8" else ""
    ax2.set_title(f'Actual Correlation ({correlation_type}) of Expertise and Gaussian RSA {title_description}', fontweight='bold')
    # Add colorbar for correlations
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Correlation', fontweight='bold')
    # Add correlation values as text
    for i in range(len(ROIs)):
        for j in range(len([1])):
            if not np.isnan(r_matrix[i, j]):
                ax2.text(j, i, f'{r_matrix[i, j]:.2f}', ha='center', va='center', 
                        color='white' if abs(r_matrix[i, j]) > 0.2 else 'black',
                        fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / f'test_sigma_{sigma_val}_{correlation_type}{added_description}_r_heatmap.png', 
                dpi=300, bbox_inches='tight')
