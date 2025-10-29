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
data_dir.mkdir(parents=True, exist_ok=True)

rdm_dist = "correlation"
added_description = ""  # ""  or  "_no_sub_8"
n_permutations = 10000
ROIs = ["V1", "V2", "V3", "V4", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]

if added_description == "_no_sub_8_":
    all_subjects = ['05', '06', '07', '09', '10', '11', '12', '14', '15', '16', '17',  
                    '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']
else:
    all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17',  
                    '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                    '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

permutation_results_r = {roi: [] for roi in ROIs}
permutation_results_p = {roi: [] for roi in ROIs}

# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)

subjects = list(expertise_scores.keys())
expertise_scores_values = list(expertise_scores.values())

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


def calculate_gaussian_weight(subject1, subject2, sigma, scores):
    if subject1 not in scores or subject2 not in scores:
        return 0
        
    score1 = scores[subject1]
    score2 = scores[subject2]
    expertise_diff = abs(score1 - score2)
    
    weight = np.exp(-(expertise_diff**2) / (2 * sigma**2))
    return weight


# adaptive_subgroup_rsa_results = {} # PARSA values for all subjects and ROIs

# for ROI in ROIs:
#     print(f"Analyzing ROI: {ROI} with Gaussian sigma {sigma_val}")
#     adaptive_subgroup_rsa_results[ROI] = {}

#     for subject1 in all_subjects:
#         if ROI not in RDM_dict[subject1]:
#             continue
            
#         raw_weights = []
#         correlations = []
        
#         for subject2 in all_subjects:
#             if (subject1 == subject2 or ROI not in RDM_dict[subject2]):
#                 continue
            
#             # Calculate Gaussian weight based on expertise similarity
#             raw_weight = calculate_gaussian_weight(subject1, subject2, sigma_val)
#             corr_value = rsatoolbox.rdm.compare(RDM_dict[subject1][ROI], RDM_dict[subject2][ROI], method='corr')
            
#             raw_weights.append(raw_weight)
#             correlations.append(corr_value.item())

#         # Normalize weights so they sum to 1
#         total_weight = sum(raw_weights)
#         normalized_weights = [w / total_weight for w in raw_weights]
        
#         # Calculate weighted average
#         weighted_correlations = [w * c for w, c in zip(normalized_weights, correlations)]
#         adaptive_subgroup_rsa_results[ROI][subject1] = sum(weighted_correlations)


for correlation_type in ["pearson", "spearman"]:
    for i in range(n_permutations):
        print(f"Getting permutation {i} values...")
        shuffled_scores = dict(zip( subjects, sample(expertise_scores_values, len(expertise_scores_values)) ))

        for ROI in ROIs:
            # Get expertise scores and correlations as arrays
            scores = [shuffled_scores[subject] for subject in all_subjects]
            similarity_correlations = []

            for subject1 in all_subjects:
                if ROI not in RDM_dict[subject1]:
                    continue
                    
                raw_weights = []
                correlations = []
                
                for subject2 in all_subjects:
                    if (subject1 == subject2 or ROI not in RDM_dict[subject2]):
                        continue
                    
                    # Calculate Gaussian weight based on expertise similarity
                    raw_weight = calculate_gaussian_weight(subject1, subject2, sigma_val, shuffled_scores)
                    corr_value = rsatoolbox.rdm.compare(RDM_dict[subject1][ROI], RDM_dict[subject2][ROI], method='corr')
                    
                    raw_weights.append(raw_weight)
                    correlations.append(corr_value.item())

                # Normalize weights so they sum to 1
                total_weight = sum(raw_weights)
                normalized_weights = [w / total_weight for w in raw_weights]
                
                # Calculate weighted average
                weighted_correlations = [w * c for w, c in zip(normalized_weights, correlations)]
                similarity_correlations.append( sum(weighted_correlations) )

            # Calculate pearson correlations
            if correlation_type == "pearson":
                r, p_value = pearsonr(scores, similarity_correlations)
            if correlation_type == "spearman":
                r, p_value = spearmanr(scores, similarity_correlations)   

            permutation_results_r[ROI].append(r)
            permutation_results_p[ROI].append(p_value)

    # Save results
    permutation_out_path_r = data_dir / f'test_sigma_{sigma_val}_{correlation_type}{added_description}_weighted_permutations_r.json'
    with open(permutation_out_path_r, 'w') as f:
        json.dump(permutation_results_r, f)

    permutation_out_path_p = data_dir / f'test_sigma_{sigma_val}_{correlation_type}{added_description}_weighted_permutations_p.json'
    with open(permutation_out_path_p, 'w') as f:
        json.dump(permutation_results_p, f)
