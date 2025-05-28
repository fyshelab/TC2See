import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
import json


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

rdm_dist = "correlation"
similarity_threshold = 10  # Expertise score difference threshold in percentage points
threshold_dir = results_dir / f'adaptive_rsa/{similarity_threshold}_percent'
threshold_dir.mkdir(parents=True, exist_ok=True)

ROIs = ["V1", "V2", "V3", "V4", "V6", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC"]
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
            rdm_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}/sub_{subject}" / ROI
            rdm_file_path = rdm_path / f'rdm_for_{ROI}.hdf5'
            
            rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
            RDM_dict[subject][ROI] = rdm
    except Exception as e:
        print(f"Error processing subject {subject}, ROI: {ROI}")
        continue


# Function to determine if two subjects have similar expertise levels
def has_similar_expertise(subject1, subject2, threshold):
    if subject1 not in expertise_scores or subject2 not in expertise_scores:
        return False
        
    score1 = expertise_scores[subject1]
    score2 = expertise_scores[subject2]
    
    return abs(score1 - score2) <= threshold


##############################################################################
# For each ROI, calculate correlations between subjects with similar expertise
##############################################################################

expertise_vs_rsa_correlation_similarity = {}
matched_participants = {}
adaptive_subgroup_rsa_results = {}

for ROI in ROIs:
    print(f"Analyzing ROI: {ROI} with similarity threshold {similarity_threshold}%")
    try:
        adaptive_subgroup_rsa_results[ROI] = {}
        
        for subject1 in all_subjects:
            if ROI not in RDM_dict[subject1]:
                continue
                
            # Find subjects with similar expertise
            similar_expertise_correlations = []
            similar_expertise_subjects = []
            
            for subject2 in all_subjects:
                if (subject1 == subject2 or ROI not in RDM_dict[subject2]):
                    continue
                
                # Check if subjects have similar expertise
                if has_similar_expertise(subject1, subject2, similarity_threshold):
                    corr_value = rsatoolbox.rdm.compare(RDM_dict[subject1][ROI], RDM_dict[subject2][ROI], method='corr')
                    similar_expertise_correlations.append(corr_value)
                    similar_expertise_subjects.append(subject2)
            
            if similar_expertise_correlations:
                adaptive_subgroup_rsa_results[ROI][subject1] = np.mean(similar_expertise_correlations)
                matched_participants[subject1] = similar_expertise_subjects
        
        # Subjects with valid data
        valid_subjects = list(adaptive_subgroup_rsa_results[ROI].keys())
        
        # Get expertise scores and correlations as arrays
        scores = [expertise_scores[subject] for subject in valid_subjects]
        similarity_correlations = [adaptive_subgroup_rsa_results[ROI][subject] for subject in valid_subjects]
        
        # Calculate Spearman and Pearson correlations
        rho, p_value = spearmanr(scores, similarity_correlations)        
        r, p_value_pearson = pearsonr(scores, similarity_correlations)
    
        expertise_vs_rsa_correlation_similarity[ROI] = {
            'spearman_rho': rho,
            'spearman_p': p_value,
            'pearson_r': r,
            'pearson_p': p_value_pearson
        }
        
    except Exception as e:
        print(f"Error analyzing ROI {ROI}")
        continue


results_df = pd.DataFrame(expertise_vs_rsa_correlation_similarity).T.reset_index()
results_df = results_df.rename(columns={'index': 'ROI'})

# Benjamini/Hochberg multiple comparison correction
_, corrected_p_values, _, _ = multipletests(results_df['spearman_p'], method='fdr_bh')
results_df['spearman_p_corrected'] = corrected_p_values

_, corrected_p_values_pearson, _, _ = multipletests(results_df['pearson_p'], method='fdr_bh')
results_df['pearson_p_corrected'] = corrected_p_values_pearson



########################################
# Save results for use in other analyses
########################################

# Save similar_expertise_correlation_results as a JSON file
with open(results_dir / f'{threshold_dir}/{similarity_threshold}pct_expertise_vs_rsa_correlation_similarity.json', 'w') as f:
    json.dump(expertise_vs_rsa_correlation_similarity, f)
    
# Save results_df as a parquet file
results_df.to_parquet(results_dir / f'{threshold_dir}/{similarity_threshold}pct_results_df.parquet', index=False)

# Save the matched_participants dictionary as a JSON file
with open(results_dir / f'{threshold_dir}/{similarity_threshold}pct_matched_participants.json', 'w') as f:
    json.dump(matched_participants, f)

# Save the subject_similarity_correlations dictionary as a JSON file
with open(results_dir / f'{threshold_dir}/{similarity_threshold}pct_adaptive_subgroup_rsa_results.json', 'w') as f:
    json.dump(adaptive_subgroup_rsa_results, f)



##########################################################################################
# Plot correlation between expertise scores and similarity-based correlations for top ROIs
##########################################################################################

# Sort ROIs (lowest p-values)
rois_sorted_by_p = results_df.sort_values(by='pearson_p')['ROI'].tolist()

plt.figure(figsize=(18, 12)) 
for i, roi in enumerate(rois_sorted_by_p):

    valid_subjects = list(adaptive_subgroup_rsa_results[roi].keys())
    x = [expertise_scores[subject] for subject in valid_subjects]
    y = [adaptive_subgroup_rsa_results[roi][subject] for subject in valid_subjects]
    
    # Plot scatter with regression line
    plt.subplot(3, 4, i+1)
    plt.scatter(x, y, alpha=0.7)
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, 'r-')

    plt.title(f'ROI {roi}')
    plt.text(0.05, 0.95, 
             f'r = {expertise_vs_rsa_correlation_similarity[roi]["pearson_r"]:.3f}\np = {expertise_vs_rsa_correlation_similarity[roi]["pearson_p"]:.3f}\np_adj = {results_df.loc[results_df["ROI"] == roi, "pearson_p_corrected"].values[0]:.3f}',  
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.xlabel('Expertise Score')
    plt.ylabel('Avg Similar-Expertise RDM Correlation')
plt.tight_layout()
plt.savefig(results_dir / f'{threshold_dir}/{similarity_threshold}pct_top_rois.png', dpi=300)