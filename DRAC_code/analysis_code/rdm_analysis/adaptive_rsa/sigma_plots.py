import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

sigma_val = "5"
corr = 'gaussian'
correlation_type = "pearson"
added_description = "" # "" or "_no_sub_8"
data_dir = results_dir / f"{corr}/{sigma_val}"
ROIs = ["V1", "V2", "V3", "V4", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]


try:
    permutation_path = results_dir / f'permutations/sigma_{correlation_type}_{sigma_val}{added_description}_weighted_permutations_r.json'
    with open(permutation_path, 'r') as f:
        permutation_results = json.load(f)
    permutation_data_available = True
except FileNotFoundError:
    permutation_data_available = False
    print(f"Permutation results not available, skipping p-value analysis...")

# Load the correlation results for this sigma range
sigma_corr_results_path = results_dir / data_dir / f'{sigma_val}_{corr}{added_description}_range_results.json'
with open(sigma_corr_results_path, 'r') as f:
    sigma_corr_results = json.load(f)

##################################
# Plots for r values across sigmas
##################################

plt.figure(figsize=(10, 6))

for roi in ROIs: 
    sigmas_with_data = []
    correlations = []
    

    if sigma_val in sigma_corr_results and roi in sigma_corr_results[sigma_val]:
        sigmas_with_data.append(sigma_val)
        correlations.append(sigma_corr_results[sigma_val][roi][f'{correlation_type}_r'])
    
    if sigmas_with_data:
        plt.plot(sigmas_with_data, correlations, '-', label=f'{roi}')

plt.xlabel('Sigma Value')
plt.ylabel(f'{correlation_type} Correlation')
plt.title('Effect of Sigma on Expertise-Correlation Relationship')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / data_dir / f'sigma_{sigma_val}{added_description}_r_sigma_variations.png', dpi=300)

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
    ax1.set_title('P-values', fontweight='bold')
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
    plt.savefig(results_dir / f'permutations/sigma_{sigma_val}{added_description}_pvalue_heatmap_sigmas.png', 
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
    ax2.set_title('Correlation', fontweight='bold')
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
    plt.savefig(results_dir / f'permutations/sigma{sigma_val}{added_description}_correlation_heatmap_sigmas.png', 
                dpi=300, bbox_inches='tight')
