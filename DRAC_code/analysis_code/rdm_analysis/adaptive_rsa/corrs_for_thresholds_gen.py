import numpy as np
import rsatoolbox
from pathlib import Path
from scipy.stats import pearsonr
import json


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

rdm_dist = "correlation"
lower, upper, step = 1, 10, 1
thresholds = list(range(lower, upper + 1, step))

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



####################################################################################################
# Save the correlation of expertise and similar RDM correlation relationship at different thresholds 
####################################################################################################

threshold_results = {}

for threshold in thresholds:
    print(f"Analyzing correlations using threshold {threshold}%")
    threshold_results[threshold] = {}
    matched_participants = {}
    
    for roi in ROIs:
        try:
            diff_threshold_adaptive_subgroup_rsa_results = {}
            
            for subject1 in all_subjects:
                if roi not in RDM_dict[subject1]:
                    continue
                    
                # Find subjects with similar expertise
                similar_expertise_correlations = []
                similar_expertise_subjects = []
                
                for subject2 in all_subjects:
                    
                    if (subject1 == subject2 or ROI not in RDM_dict[subject2]):
                        continue

                    # Check if subjects have similar expertise using current threshold
                    if has_similar_expertise(subject1, subject2, threshold):
                        corr_value = rsatoolbox.rdm.compare(RDM_dict[subject1][roi], RDM_dict[subject2][roi], method='corr')
                        similar_expertise_correlations.append(corr_value)
                        similar_expertise_subjects.append(subject2)
                
                # Only include subjects that had at least one match
                if similar_expertise_correlations:
                    diff_threshold_adaptive_subgroup_rsa_results[subject1] = np.mean(similar_expertise_correlations)
                    matched_participants[subject1] = similar_expertise_subjects
            
            valid_subjects = list(diff_threshold_adaptive_subgroup_rsa_results.keys())
            
            # Get expertise scores and correlations as arrays
            scores = [expertise_scores[subject] for subject in valid_subjects]
            similarity_correlations = [diff_threshold_adaptive_subgroup_rsa_results[subject] for subject in valid_subjects]
            
            r, p_value = pearsonr(scores, similarity_correlations)
            
            threshold_results[threshold][roi] = {
                'pearson_r': r,
                'pearson_p': p_value,
                'valid_subjects': len(valid_subjects)
            }
            
        except Exception as e:
            print(f"    Error analyzing ROI {roi} at threshold {threshold}")
            continue


# Save threshold_results dict as JSON
threshold_results_path = results_dir / f'adaptive_rsa/{lower}_{upper}_{step}_threshold_results.json'
with open(threshold_results_path, 'w') as f:
    json.dump(threshold_results, f, indent=4)