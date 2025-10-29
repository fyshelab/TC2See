import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json
import traceback
from matplotlib.ticker import FormatStrFormatter


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
rdm_dist = "correlation"
version = "" # "" or "no_sub_8_"

sigma = 5  # Gaussian decay parameter 
sigma_dir = results_dir / f'gaussian_rsa/sigma_{sigma}'
sigma_dir.mkdir(parents=True, exist_ok=True)

# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)

with open(sigma_dir / f'data/{version}sigma_{sigma}_expertise_vs_rsa_correlation_similarity.json', 'r') as f:
    expertise_vs_rsa_correlation_similarity = json.load(f)

results_df = pd.read_parquet(sigma_dir / f'data/{version}sigma_{sigma}_results_df.parquet')

with open(sigma_dir / f'data/{version}sigma_{sigma}_adaptive_subgroup_rsa_results.json', 'r') as f:
    adaptive_subgroup_rsa_results = json.load(f)


##########################################################################################
# Plot correlation between expertise scores and similarity-based correlations for each ROI
##########################################################################################
    
for corr_type in ["pearson", "spearman"]:
    rois_sorted_by_p = results_df['ROI'].tolist()

    plt.figure(figsize=(18, 23)) 
    for i, roi in enumerate(rois_sorted_by_p):
        valid_subjects = list(adaptive_subgroup_rsa_results[roi].keys())
        x = [expertise_scores[subject] for subject in valid_subjects]
        y = [adaptive_subgroup_rsa_results[roi][subject] for subject in valid_subjects]
        
        # Plot scatter with regression line
        plt.subplot(5, 3, i + 1)
        plt.scatter(x, y, alpha=0.7)
        
        if corr_type == "pearson":
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            x_line = np.linspace(min(x), max(x), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r-')

        plt.ylim(-0.02, 0.37)
        plt.xlim(35, 90)

        plt.title(f'ROI {roi}')
        plt.text(0.05, 0.95, 
                f'r = {expertise_vs_rsa_correlation_similarity[roi][f"{corr_type}_r"]:.3f}\np = {expertise_vs_rsa_correlation_similarity[roi][f"{corr_type}_p"]:.3f}\np_adj = {results_df.loc[results_df["ROI"] == roi, f"{corr_type}_p_corrected"].values[0]:.3f}',  
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.xlabel('Expertise Score')
        plt.ylabel(f'Proximity Aware Correlation ({corr_type})')
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.subplots_adjust(top=0.92)  
    print(f"Saving plot to {str(sigma_dir)}" + "\\" + f"visuals/scatters/{corr_type}_{version}sigma_{sigma}_scatters.png")
    plt.savefig(sigma_dir / f'visuals/scatters/{corr_type}_{version}sigma_{sigma}_scatters.png', dpi=300)