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
import rsatoolbox
from tqdm import tqdm


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

expertise_groups   = ['low', 'high']
rdm_dist = "correlation"

subject_strs_dict = {
    'low':      ['low', ['06', '15', '20', '21', '22', '23', '28', '29', '32', '33', '39']],
    'moderate': ['moderate', []],
    'high':     ['high', ['08', '09', '11', '12', '16', '24', '26', '35', '38']]
}

ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]

group_RDMs_corr = {}
group_RDMs_std = {}
lowest_vmin = 10
highest_vmax = -10

for group in expertise_groups:
    print(f"Plotting RDMs for {group} scoring group")
    group_RDMs_corr[group] = {}
    group_RDMs_std[group] = {}
    expertise_level_str = subject_strs_dict[group][0]
    subject_strs        = subject_strs_dict[group][1]

    for ROI in tqdm(ROIs):
        all_sub_corr_mats = []

        for subject_str in subject_strs:

            # rdm_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}/sub_{subject_str}" / ROI
            rdm_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}_ordered/sub_{subject_str}" / ROI
            rdm_file_path = rdm_path / f'rdm_for_{ROI}.hdf5'
            
            rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
            sub_corr_matrix = rdm.get_matrices()[0]

            # Convert dissimilarities to correlation values
            sub_corr_matrix = 1 - sub_corr_matrix

            # Clip correlation values to avoid atanh(±1) infinities
            sub_corr_matrix = np.clip(sub_corr_matrix, -0.999999, 0.999999)

            # Apply Fisher z-transform
            z_matrix = np.arctanh(sub_corr_matrix)

            # Set diagonal to 0 (z = atanh(0) = 0)
            np.fill_diagonal(z_matrix, 0)

            all_sub_corr_mats.append(z_matrix)

            # Plot individual RDM
            fig, axes = plt.subplots(1, 1, figsize=(10, 8))

            sns.heatmap(sub_corr_matrix, annot=False, cmap='seismic', ax=axes, vmin=-0.7, vmax=0.75)
            axes.set_title(f'Correlation Matrix Heatmap for {ROI} ({group} scoring participant)')
            fig.savefig(results_dir / f"ordered_rdm_plots/all_subjects/sub_{subject_str}_{ROI}_{group}.png")
            plt.close()



        # Average RDM for this group and ROI
        avg_z_matrix = np.mean(all_sub_corr_mats, axis=0)
        avg_corr_matrix = np.tanh(avg_z_matrix)

        avg_std_matrix = np.std(all_sub_corr_mats, axis=0)
        
        np.fill_diagonal(avg_corr_matrix, 0)

        group_RDMs_corr[group][ROI] = avg_corr_matrix
        group_RDMs_std[group][ROI] = avg_std_matrix


# Plot side-by-side heatmaps for low and high group averages
for ROI in ROIs:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, group in enumerate(expertise_groups):
        matrix = group_RDMs_corr[group][ROI]
        sns.heatmap(matrix, annot=False, cmap='seismic', ax=axes[i], vmin=-0.25, vmax=0.20)
        axes[i].set_title(f'{group.capitalize()} Scoring Group - {ROI}')

    fig.suptitle(f'Average of RDMs for ROI: {ROI}', fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(results_dir / f"ordered_rdm_plots/group_avg/{ROI}_group_comparison.png")
    plt.close()


# Plot side-by-side heatmaps for low and high group STDs
for ROI in ROIs:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for i, group in enumerate(expertise_groups):
        matrix = group_RDMs_std[group][ROI]
        sns.heatmap(matrix, annot=False, cmap='seismic', ax=axes[i], vmin=0.0, vmax=0.3)
        axes[i].set_title(f'{group.capitalize()} Scoring Group - {ROI}')

    fig.suptitle(f'STD of RDMs for ROI: {ROI}', fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(results_dir / f"ordered_rdm_plots/group_std/{ROI}_group_comparison.png")
    plt.close()