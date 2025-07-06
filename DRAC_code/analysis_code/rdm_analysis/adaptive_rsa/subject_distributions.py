import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from statsmodels.stats.multitest import multipletests
import json
import traceback
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

dataset_root = Path("D:/Documents/DRAC/TC2See/data")
results_dir = Path("D:/Documents/DRAC/TC2See/results")
rdm_dist = "correlation"
sigma = 5  # Gaussian decay parameter (expertise score units)
sigma_dir = results_dir / f'gaussian/gaussian_sigma_{sigma}'
sigma_dir.mkdir(parents=True, exist_ok=True)

# ROIs = ["V1", "V2", "V3", "V4", "V6", "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC"]
ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "A1", "FFC", "VVC", "Pir"]
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

participant_weights = {}


for ROI in ROIs:
        for subject1 in all_subjects:
            if ROI not in RDM_dict[subject1]:
                continue
                
            raw_weights = []
            correlations = []
            participant_weights[subject1] = {}
            
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
            weighted_subjects = all_subjects.copy()
            weighted_subjects.remove(subject1)

            for i, subject2 in enumerate(weighted_subjects):
                participant_weights[subject1][subject2] = normalized_weights[i]
    

plots_dir = sigma_dir / 'participant_weights'
plots_dir.mkdir(parents=True, exist_ok=True)

# Plot bar charts for each subject's weights
for subject1 in participant_weights:
    if not participant_weights[subject1]:  # Skip if no weights for this subject
        continue

    # Get other subjects and their weights
    other_subjects = list(participant_weights[subject1].keys())
    weights = list(participant_weights[subject1].values())
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    bars = plt.bar(other_subjects, weights, alpha=0.7, color='steelblue')
    
    # Customize the plot
    plt.title(f'Participant Weights for Subject {subject1}\n(Gaussian σ = {sigma})', fontsize=14, fontweight='bold')
    plt.xlabel('Other Subjects', fontsize=12)
    plt.ylabel('Weight Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    max_weight = max(weights)
    plt.ylim(top=max_weight + 0.15 * max_weight)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    
    # Add value labels on top of bars
    for bar, weight in zip(bars, weights):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f'{weight:.3f}',
            ha='center', va='bottom', fontsize=8,
            rotation=65
        )
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) 
    
    # Save the plot
    plot_filename = f'subject_{subject1}_weights.png'
    plt.savefig(plots_dir / plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot for subject {subject1}")


########################################################################
# Plot Subject Expertise Scores as a Bar Plot Ordered From Least to Most
########################################################################

sorted_items = sorted(expertise_scores.items(), key=lambda item: item[1], reverse=True)
keys_sorted, values_sorted = zip(*sorted_items)

plt.figure(figsize=(16, 6))
bars = plt.bar(keys_sorted, values_sorted)
plt.xlabel("Subject")
plt.ylabel("Score")
plt.title("Expertise Scores for Each Subject")
plt.xticks(rotation=45, ha='right')

plt.gca().yaxis.set_major_locator(MultipleLocator(10))
plt.grid(axis='y', alpha=0.3)

# Add angled value labels on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2, yval + 1,
        f'{yval:.2f}',
        ha='center', va='bottom',
        rotation=45
    )

max_score = max(values_sorted)
plt.ylim(top=max_score + 0.15 * max_score)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.tight_layout()
plt.savefig(results_dir / "subject_expertise_distribution", dpi=300)