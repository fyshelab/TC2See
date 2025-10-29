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


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
rdm_dist = "correlation"
version = "pred_embed_" # "" or "no_sub_8_"

sigma = 5  # Gaussian decay parameter 
sigma_dir = results_dir / f'gaussian_rsa/sigma_{sigma}'
sigma_dir.mkdir(parents=True, exist_ok=True)

# Load ROI name mapping
with open(dataset_root / "roi_names.json", 'r') as f:
    roi_names = json.load(f)

ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]
# ROIs = [roi_names[str(i)] for i in range(1, 181)]
# ROIs = ["24dd", "24dv", "p24", "p24pr", "p32", "a32pr", "s32", "d32", "p32pr", "33pr"]
    
if version == "no_sub_8_":
    all_subjects = ['05', '06', '07', '09', '10', '11', '12', '14', '15', '16', '17', 
                    '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']
else:
    all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                    '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)

# Initialize dictionary to store RDMs
RDM_dict = {subject: {} for subject in all_subjects}

# Load RDMs for each subject and ROI
for subject in all_subjects:
    try:
        for ROI in ROIs:
            if version == "pred_embed_":
                rdm_path = dataset_root / f"processed/predicted_embeddings_RDMs/individual/sub_{subject}/experiment-8.4/ViT-B=16-features_large"
                rdm_file_path = rdm_path / f'{ROI}.hdf5'
            else:
                rdm_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}/sub_{subject}" / ROI
                rdm_file_path = rdm_path / f'rdm_for_{ROI}.hdf5'
            
            rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
            RDM_dict[subject][ROI] = rdm
    except Exception as e:
        print(f"Error processing subject {subject}, ROI: {ROI}")
        continue


# Function to calculate Gaussian weight based on expertise difference
def calculate_gaussian_weight(subject1, subject2, sigma):
    if subject1 not in expertise_scores or subject2 not in expertise_scores:
        return 0
        
    score1 = expertise_scores[subject1]
    score2 = expertise_scores[subject2]
    expertise_diff = abs(score1 - score2)
    
    weight = np.exp(-(expertise_diff**2) / (2 * sigma**2))
    return weight



##############################################################################
# For each ROI, calculate weighted correlations based on expertise similarity
##############################################################################

expertise_vs_rsa_correlation_similarity = {}
weighted_participants = {}
adaptive_subgroup_rsa_results = {}

for ROI in ROIs:
    print(f"Analyzing ROI: {ROI} with Gaussian sigma {sigma}")
    try:
        adaptive_subgroup_rsa_results[ROI] = {}
    
        for subject1 in all_subjects:
            if ROI not in RDM_dict[subject1]:
                continue
                
            raw_weights = []
            correlations = []
            
            for subject2 in all_subjects:
                if (subject1 == subject2 or ROI not in RDM_dict[subject2]):
                    continue
                
                # Calculate Gaussian weight based on expertise similarity
                raw_weight = calculate_gaussian_weight(subject1, subject2, sigma)
                corr_value = rsatoolbox.rdm.compare(RDM_dict[subject1][ROI], RDM_dict[subject2][ROI], method='corr')
                
                raw_weights.append(raw_weight)
                correlations.append(corr_value.item())

            # Normalize weights so they sum to 1
            total_weight = sum(raw_weights)
            normalized_weights = [w / total_weight for w in raw_weights]
            
            # Calculate weighted average
            weighted_correlations = [w * c for w, c in zip(normalized_weights, correlations)]
            adaptive_subgroup_rsa_results[ROI][subject1] = sum(weighted_correlations)
            
        
        # Subjects with valid data
        valid_subjects = list(adaptive_subgroup_rsa_results[ROI].keys())
        
        # Get expertise scores and correlations as arrays
        scores = [expertise_scores[subject] for subject in valid_subjects]
        similarity_correlations = [adaptive_subgroup_rsa_results[ROI][subject] for subject in valid_subjects]
        
        # Calculate pearson correlations
        r, p_value_pearson = pearsonr(scores, similarity_correlations)
        rho, p_value = spearmanr(scores, similarity_correlations)   
    
        expertise_vs_rsa_correlation_similarity[ROI] = {
            'pearson_r': r,
            'pearson_p': p_value_pearson,
            'spearman_r': rho,
            'spearman_p': p_value
        }
        
    except Exception as e:
        print(f"\n\nError analyzing ROI {ROI}, subject1: {subject1}, subject2: {subject2} \n")
        traceback.print_exc()
        print("\n\n")
        continue


results_df = pd.DataFrame(expertise_vs_rsa_correlation_similarity).T.reset_index()
results_df = results_df.rename(columns={'index': 'ROI'})

# Benjamini/Hochberg multiple comparison correction
_, corrected_p_values_pearson, _, _ = multipletests(results_df['pearson_p'], method='fdr_bh')
results_df['pearson_p_corrected'] = corrected_p_values_pearson

_, corrected_p_values_spearman, _, _ = multipletests(results_df['spearman_p'], method='fdr_bh')
results_df['spearman_p_corrected'] = corrected_p_values_spearman



########################################
# Save results for use in other analyses
########################################

# Save similar_expertise_correlation_results as a JSON file
with open(sigma_dir / f'data/{version}sigma_{sigma}_expertise_vs_rsa_correlation_similarity.json', 'w') as f:
    json.dump(expertise_vs_rsa_correlation_similarity, f)
    
# Save results_df as a parquet file
results_df.to_parquet(sigma_dir / f'data/{version}sigma_{sigma}_results_df.parquet', index=False)

# Save the subject_similarity_correlations dictionary as a JSON file
with open(sigma_dir / f'data/{version}sigma_{sigma}_adaptive_subgroup_rsa_results.json', 'w') as f:
    json.dump(adaptive_subgroup_rsa_results, f)



##########################################################################################
# Plot correlation between expertise scores and similarity-based correlations for top ROIs
##########################################################################################

for corr_type in ["pearson", "spearman"]:
    rois_sorted_by_p = results_df['ROI'].tolist()

    plt.figure(figsize=(20, 12)) # 13 
    # plt.figure(figsize=(20, 8)) # 10
    for i, roi in enumerate(rois_sorted_by_p):
        valid_subjects = list(adaptive_subgroup_rsa_results[roi].keys())
        x = [expertise_scores[subject] for subject in valid_subjects]
        y = [adaptive_subgroup_rsa_results[roi][subject] for subject in valid_subjects]
        
        # Plot scatter with regression line
        plt.subplot(3, 5, i + 1) # 13
        # plt.subplot(2, 5, i + 1) # 10
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

    plt.suptitle(f'Expertise vs Proximity Aware Correlation\n(Gaussian σ = {sigma})',
             fontsize=16, fontweight='bold', y=.98)
    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.subplots_adjust(top=0.85) 
    print(f"Saving plot to {str(sigma_dir)}" + "\\" + f"visuals/scatters/{corr_type}_{version}sigma_{sigma}_scatters.png")
    plt.savefig(sigma_dir / f'visuals/scatters/{corr_type}_{version}sigma_{sigma}_scatters.png', dpi=300, bbox_inches=None)


# for corr_type in ["pearson", "spearman"]:
#     rois_sorted_by_p = results_df['ROI'].tolist()
#     individual_plot_dir = sigma_dir / f'visuals/scatters/individual_plots/{corr_type}'
#     individual_plot_dir.mkdir(parents=True, exist_ok=True)

#     for roi in rois_sorted_by_p:
#         valid_subjects = list(adaptive_subgroup_rsa_results[roi].keys())
#         x = [expertise_scores[subject] for subject in valid_subjects]
#         y = [adaptive_subgroup_rsa_results[roi][subject] for subject in valid_subjects]

#         plt.figure(figsize=(6, 5))
#         plt.scatter(x, y, alpha=0.7)

#         if corr_type == "pearson":
#             slope, intercept, r_value, p_value, std_err = linregress(x, y)
#             x_line = np.linspace(min(x), max(x), 100)
#             y_line = slope * x_line + intercept
#             plt.plot(x_line, y_line, 'r-')

#         plt.ylim(-0.02, 0.37)
#         plt.xlim(35, 90)
#         plt.title(f'ROI {roi}')
#         plt.text(0.05, 0.95, 
#                  f'r = {expertise_vs_rsa_correlation_similarity[roi][f"{corr_type}_r"]:.3f}\n'
#                  f'p = {expertise_vs_rsa_correlation_similarity[roi][f"{corr_type}_p"]:.3f}\n'
#                  f'p_adj = {results_df.loc[results_df["ROI"] == roi, f"{corr_type}_p_corrected"].values[0]:.3f}',  
#                  transform=plt.gca().transAxes, verticalalignment='top')
#         plt.xlabel('Expertise Score')
#         plt.ylabel(f'Proximity Aware Correlation ({corr_type})')
#         plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

#         filename = f'{roi}_{corr_type}_{version}sigma_{sigma}.png'
#         save_path = individual_plot_dir / filename
#         print(f"Saving individual plot to {save_path}")
#         plt.tight_layout()
#         plt.savefig(save_path, dpi=300)
#         plt.close()