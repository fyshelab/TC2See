import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

lower, upper, step = 5, 10, 1
thresholds_to_permute = [str(num) for num in range(lower, upper + 1, step)]
ROIs = ["V1", "V2", "V3", "V4", "V6", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC"]


try:
    permutation_path = results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_permutation_test_nulls.json'
    with open(permutation_path, 'r') as f:
        permutation_results = json.load(f)
    permutation_data_available = True
except FileNotFoundError:
    permutation_data_available = False
    print(f"Permutation results not available, skipping p-value analysis...")

# Load the correlation results for this threshold range
threshold_results_path = results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_threshold_results.json'
with open(threshold_results_path, 'r') as f:
    threshold_results = json.load(f)


######################################
# Plots for r values across thresholds
######################################

plt.figure(figsize=(10, 6))

for roi in ROIs: 
    thresholds_with_data = []
    correlations = []
    
    for threshold in thresholds_to_permute:
        if threshold in threshold_results and roi in threshold_results[threshold]:
            thresholds_with_data.append(threshold)
            correlations.append(threshold_results[threshold][roi]['pearson_r'])
    
    if thresholds_with_data:
        plt.plot(thresholds_with_data, correlations, '-', label=f'{roi}')

plt.xlabel('Expertise Similarity Threshold (%)')
plt.ylabel('Pearson Correlation')
plt.title('Effect of Subgroup Threshold on Expertise-Correlation Relationship')
plt.xticks(range(lower, upper + 1))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_similarity_threshold_analysis.png', dpi=300)



####################################################
# Plot p-values for different ROIs at each threshold
####################################################


if permutation_data_available:
    observed_correlations = {roi: {thr: None for thr in thresholds_to_permute} for roi in ROIs}
    p_values = {roi: {thr: None for thr in thresholds_to_permute} for roi in ROIs}

    for threshold in thresholds_to_permute:
        for roi in ROIs:
            observed_r = threshold_results[threshold][roi]['pearson_r']
            null_distribution = permutation_results[roi][threshold]

            p_val = (np.sum(np.array(null_distribution) >= observed_r) + 1) / (len(null_distribution) + 1)
            p_values[roi][threshold] = p_val


    roi_avg_pvalues = {}
    for roi in ROIs:
        valid_pvalues = [p_values[roi][threshold] for threshold in thresholds_to_permute 
                        if p_values[roi][threshold] is not None]
        if valid_pvalues:
            roi_avg_pvalues[roi] = np.mean(valid_pvalues)
    top_7_rois = sorted(roi_avg_pvalues.keys(), key=lambda x: roi_avg_pvalues[x])[:7]

    plt.figure(figsize=(12, 8))

    for roi in top_7_rois:
        # Extract valid threshold-pvalue pairs for this ROI
        valid_thresholds = []
        valid_pvalues = []
        
        for threshold in thresholds_to_permute:
            if p_values[roi][threshold] is not None:
                valid_thresholds.append(threshold)
                valid_pvalues.append(p_values[roi][threshold])
        
        if len(valid_thresholds) > 0 and min(valid_pvalues) < 0.1:
            plt.plot(valid_thresholds, valid_pvalues, '-', linewidth=2, markersize=6, label=roi)


    # Significant threshold line
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
            label='p = 0.05')

    plt.xlabel('Expertise Similarity Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('P-value (Permutation Test)', fontsize=12, fontweight='bold')
    plt.title('P-values across Thresholds for Each ROI', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_permutation_pvalues_plot.png', dpi=300, bbox_inches='tight')



##################################
# P-value and correlation heatmaps 
##################################
# Create matrices for heatmap
p_matrix = np.full((len(ROIs), len(thresholds_to_permute)), np.nan)
r_matrix = np.full((len(ROIs), len(thresholds_to_permute)), np.nan)
for i, roi in enumerate(ROIs):
    for j, threshold in enumerate(thresholds_to_permute):
        if p_values[roi][threshold] is not None:
            p_matrix[i, j] = p_values[roi][threshold]
            r_matrix[i, j] = threshold_results[threshold][roi]['pearson_r']

# P-value heatmap
fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
im1 = ax1.imshow(p_matrix, cmap='Blues_r', aspect='auto', vmin=0, vmax=0.05)
ax1.set_xticks(range(len(thresholds_to_permute)))
ax1.set_xticklabels(thresholds_to_permute)
ax1.set_yticks(range(len(ROIs)))
ax1.set_yticklabels(ROIs)
ax1.set_xlabel('Expertise Similarity Threshold', fontweight='bold')
ax1.set_ylabel('ROI', fontweight='bold')
ax1.set_title('P-values', fontweight='bold')
cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
cbar1.set_label('P-value', fontweight='bold')
# Add p-values as text and red overlay for non-significant cells
for i in range(len(ROIs)):
    for j in range(len(thresholds_to_permute)):
        if not np.isnan(p_matrix[i, j]):
            p_val = p_matrix[i, j]
            ax1.text(j, i, f'{p_val:.5f}', ha='center', va='center',
                    color='white' if p_val < 0.025 else 'black', fontsize=9)
            if p_val > 0.05:
                ax1.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            color='red', alpha=0.6))
plt.tight_layout()
plt.savefig(results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_pvalue_heatmap.png', dpi=300, bbox_inches='tight')

# Correlation heatmap 
fig, ax2 = plt.subplots(1, 1, figsize=(8, 8))
im2 = ax2.imshow(r_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
ax2.set_xticks(range(len(thresholds_to_permute)))
ax2.set_xticklabels(thresholds_to_permute)
ax2.set_yticks(range(len(ROIs)))
ax2.set_yticklabels(ROIs)
ax2.set_xlabel('Expertise Similarity Threshold', fontweight='bold')
ax2.set_ylabel('ROI', fontweight='bold')
ax2.set_title('Correlation', fontweight='bold')
# Add colorbar for correlations
cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
cbar2.set_label('Correlation', fontweight='bold')
# Add correlation values as text
for i in range(len(ROIs)):
    for j in range(len(thresholds_to_permute)):
        if not np.isnan(r_matrix[i, j]):
            ax2.text(j, i, f'{r_matrix[i, j]:.2f}', ha='center', va='center', 
                    color='white' if abs(r_matrix[i, j]) > 0.2 else 'black',
                    fontsize=10)
plt.tight_layout()
plt.savefig(results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_correlation_heatmap.png', dpi=300, bbox_inches='tight')