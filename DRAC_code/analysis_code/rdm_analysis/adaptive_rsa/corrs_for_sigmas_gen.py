import numpy as np
import rsatoolbox
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import json
import traceback

dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

rdm_dist = "correlation"
sigma_val = 5
corr = 'gaussian'
added_description = "_no_sub_8" # "_no_sub_8"
data_dir = results_dir / f"{corr}/{sigma_val}"
data_dir.mkdir(parents=True, exist_ok=True)

ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]

if added_description == "":
    all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17',  
                    '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']
else:
    all_subjects = ['05', '06', '07', '09', '10', '11', '12', '14', '15', '16', '17',  
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



def calculate_gaussian_weight(subject1, subject2, sigma):
    if subject1 not in expertise_scores or subject2 not in expertise_scores:
        return 0
        
    score1 = expertise_scores[subject1]
    score2 = expertise_scores[subject2]
    expertise_diff = abs(score1 - score2)
    
    weight = np.exp(-(expertise_diff**2) / (2 * sigma**2))
    return weight



####################################################################################################
# Save the correlation of expertise and similar RDM correlation relationship at different thresholds 
####################################################################################################

sigma_results = {}
sigma_results[sigma_val] = {}

expertise_vs_rsa_correlation_similarity = {}
weighted_participants = {}
adaptive_subgroup_rsa_results = {}

for ROI in ROIs:
    print(f"Analyzing ROI: {ROI} with Gaussian sigma {sigma_val}")
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
                raw_weight = calculate_gaussian_weight(subject1, subject2, sigma_val)
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
    
        sigma_results[sigma_val][ROI] = {
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


# Save threshold_results dict as JSON
sigmas_results_path = results_dir / data_dir / f'{sigma_val}{added_description}_{corr}_range_results.json'

with open(sigmas_results_path, 'w') as f:
    json.dump(sigma_results, f, indent=4)
