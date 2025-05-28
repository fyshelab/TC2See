import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
from scipy.stats import pearsonr
import json
import random


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
lower, upper, step = 1, 20, 1
rdm_dist = "correlation"

ROIs = ["V1", "V2", "V3", "V4", "V6", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC"]
all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']


# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)


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


n_permutations = 1000
thresholds_to_permute = [5, 6, 7, 8, 9, 10] 
permutation_results = {roi: {thr: [] for thr in thresholds_to_permute} for roi in ROIs}

print("\nStarting permutation generation...")

for i in range(n_permutations):
    if (i + 1) % 100 == 0:
        print(f"Permutation {i + 1}/{n_permutations}")

    shuffled_scores = dict(zip(expertise_scores.keys(), random.sample(list(expertise_scores.values()), len(expertise_scores))))

    for threshold in thresholds_to_permute:
        for roi in ROIs:
            try:
                subgroup_rdm_means = {}
                for subject1 in all_subjects:
                    # if roi not in RDM_dict[subject1] or subject1 not in shuffled_scores:
                    #     continue

                    similar_correlations = []
                    for subject2 in all_subjects:
                        # if (subject1 == subject2 or
                        #     roi not in RDM_dict[subject2] or
                        #     subject2 not in shuffled_scores):
                        #     continue

                        if has_similar_expertise(subject1, subject2, threshold):
                            corr_val = rsatoolbox.rdm.compare(
                                RDM_dict[subject1][roi],
                                RDM_dict[subject2][roi],
                                method='corr'
                            )
                            similar_correlations.append(corr_val)

                    if similar_correlations:
                        subgroup_rdm_means[subject1] = np.mean(similar_correlations)

                valid_subjects = list(subgroup_rdm_means.keys())

                shuffled_x = [shuffled_scores[sub] for sub in valid_subjects]
                shuffled_y = [subgroup_rdm_means[sub] for sub in valid_subjects]

                r, _ = pearsonr(shuffled_x, shuffled_y)
                permutation_results[roi][threshold].append(r)

            except Exception as e:
                continue

permutation_out_path = results_dir / 'adaptive_rsa/permutation_test_nulls.json'
with open(permutation_out_path, 'w') as f:
    json.dump(permutation_results, f)