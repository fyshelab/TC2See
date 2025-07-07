import numpy as np
import rsatoolbox
from pathlib import Path
from scipy.stats import pearsonr
import json
from random import sample
from tqdm import tqdm


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
lower, upper, step = 1, 20, 1
thresholds_to_permute = list(range(lower, upper + 1, step))
rdm_dist = "correlation"
n_permutations = 1000

ROIs = ["V1", "V2", "V3", "V4", "V6", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC"]
all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

permutation_results_r = {roi: {thr: [] for thr in thresholds_to_permute} for roi in ROIs}
permutation_results_p = {roi: {thr: [] for thr in thresholds_to_permute} for roi in ROIs}

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


# Function to determine if two subjects have similar expertise levels
def has_similar_expertise(subject1, subject2, threshold, shuffled_scores):
    if subject1 not in shuffled_scores or subject2 not in shuffled_scores:
        return False
        
    score1 = shuffled_scores[subject1]
    score2 = shuffled_scores[subject2]
    
    return abs(score1 - score2) <= threshold


print("\nStarting permutation generation...")

for i in tqdm(range(n_permutations)):

    shuffled_scores = dict(zip( subjects, sample(expertise_scores_values, len(expertise_scores_values)) ))

    for threshold in thresholds_to_permute:
        for roi in ROIs:
            try:
                subgroup_rdm_means = {}
                for subject1 in all_subjects:
                    if roi not in RDM_dict[subject1]:
                        continue

                    similar_correlations = []
                    for subject2 in all_subjects:
                        if (subject1 == subject2 or ROI not in RDM_dict[subject2]):
                            continue

                        if has_similar_expertise(subject1, subject2, threshold, shuffled_scores):
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
                
           
                r, p = pearsonr(shuffled_x, shuffled_y)
                permutation_results_r[roi][threshold].append(r)
                permutation_results_p[roi][threshold].append(p)

            except Exception as e:
                continue

permutation_out_path_r = results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_permutations_r.json'
with open(permutation_out_path_r, 'w') as f:
    json.dump(permutation_results_r, f)


permutation_out_path_p = results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_permutations_p.json'
with open(permutation_out_path_p, 'w') as f:
    json.dump(permutation_results_p, f)