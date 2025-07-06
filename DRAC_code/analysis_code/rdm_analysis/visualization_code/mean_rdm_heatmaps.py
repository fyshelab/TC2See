import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import h5py
from scipy.stats import pearsonr
import seaborn as sns
import json
import rsatoolbox


dataset_root = Path("D:/Documents/DRAC/TC2See/data")

stimulus_images = h5py.File(dataset_root / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {i: name for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}
images_dir = dataset_root / Path("cropped")


dataset_root = Path("D:/Documents/DRAC/TC2See/data")
results_dir = Path("D:/Documents/DRAC/TC2See/results")

rdm_dist = "correlation"
rdm_plots_dir = results_dir / 'rdm_plots'
rdm_plots_dir.mkdir(parents=True, exist_ok=True)

expertise_groups   = ['low', 'high']

ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]

all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

subject_strs_dict = {
    'low':      ['low', ['06', '15', '20', '21', '22', '23', '28', '29', '32', '33', '39']],
    'moderate': ['moderate', []],
    'high':     ['high', ['08', '09', '11', '12', '16', '24', '26', '35', '38']]
}

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

# Define groups
group_dict = {
    'low': subject_strs_dict['low'][1],
    'high': subject_strs_dict['high'][1]
}

# Function to compute average RDM for a specific ROI and group
def average_rdms_for_roi(subjects, RDM_dict, roi):
    roi_rdms = []
    for subject in subjects:
        if roi in RDM_dict[subject]:
            rdm = RDM_dict[subject][roi].get_matrices()[0]
            roi_rdms.append(rdm)
    if roi_rdms:
        return np.nanmean(np.stack(roi_rdms), axis=0)
    else:
        return None

# Plot settings
vmin = 0.75
vmax = 1.0

# Generate and save RDM plots for each ROI and group
for roi in ROIs:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, group in enumerate(['low', 'high']):
        avg_rdm = average_rdms_for_roi(group_dict[group], RDM_dict, roi)
        
        if avg_rdm is not None:
            sns.heatmap(avg_rdm, annot=False, cmap="rocket", ax=axes[i], vmin=vmin, vmax=vmax)
            axes[i].set_title(f'{group.capitalize()} Expertise')
        else:
            axes[i].set_visible(False)
            print(f"No valid RDMs for ROI: {roi}, group: {group}")
    
    fig.suptitle(f'Average RDMs - {roi}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(rdm_plots_dir / f'mean/{roi}_low_high_expertise_side_by_side.png')
    plt.close()

