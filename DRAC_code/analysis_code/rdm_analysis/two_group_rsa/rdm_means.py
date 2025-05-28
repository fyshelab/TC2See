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


dataset_root = Path('../data')
derivatives_path = dataset_root / 'processed/fmriprep_surfs'

stimulus_images = h5py.File(dataset_root / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {i: name for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}
images_dir = dataset_root / Path("cropped")

expertise_groups   = ['low', 'high']
avg_duplicates = True

subject_strs_dict = {
    'low':      ['low', ['06', '15', '20', '21', '23', '28', '29', '32', '39']],
    'moderate': ['moderate', ['05', '07', '10', '14', '17', '18', '19', '22', '25', '30', '31', '33', '34', '37']],
    'high':     ['high', ['08', '09', '11', '12', '16', '24', '26', '27', '35', '36', '38']]
}


num_runs = 6
ROIs = ["V1","V2","V3","V3A","V3B","V3CD","V4","V6","V7","V8","VMV1","VMV2","VMV3","LO1","LO2","PIT","FFC","VVC","PH","PEF","a9-46v","p9-46v","IFSa","9a"]


group_RDMs_corr = {}
mean_of_group_cors = {}

for group in expertise_groups:
    mean_of_group_cors[group] = {}
    group_RDMs_corr[group] = {}
    expertise_level_str = subject_strs_dict[group][0]
    subject_strs        = subject_strs_dict[group][1]

    for ROI in ROIs:
        all_sub_corr_mats = []
        mean_of_group_cors[group][ROI] = {}

        for subject_str in subject_strs:
            roi_path = dataset_root / f"img_bold_arrays/sub_{subject_str}/corr_matrices_avg_dups/ROIs/{ROI}"

            correlation_file_name = roi_path / "correlation_matrix.npy"
            sub_corr_matrix = np.load(correlation_file_name)
            all_sub_corr_mats.append(sub_corr_matrix)


        def flatten_upper_tri(matrix):
            upper_triangle = np.triu(matrix, k=1)
            flattened_up_triang = upper_triangle.flatten()
            return flattened_up_triang

        flattened_matrices = [flatten_upper_tri(matrix) for matrix in all_sub_corr_mats]

        correlation_matrix = np.corrcoef(flattened_matrices)

        # Calculate the mean of the values in the lower triangle of the correlation matrix
        indices = np.tril_indices_from(correlation_matrix, k=-1)
        lower_triangle_values = correlation_matrix[indices]
        average_of_lower_triangle_values = np.mean(lower_triangle_values)
        standard_deviation_lower_triangle = np.std(lower_triangle_values)
        mean_of_group_cors[group][ROI]["mean"] = average_of_lower_triangle_values
        mean_of_group_cors[group][ROI]["std"]  = standard_deviation_lower_triangle

        np.fill_diagonal(correlation_matrix, 0)

        group_RDMs_corr[group][ROI] = correlation_matrix

        # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # fig, axes = plt.subplots(1, 1, figsize=(24, 24))
        # vmin = -.5
        # vmax = .5

        # # Plot for group 1
        # sns.heatmap(correlation_matrix, annot=False, cmap='seismic', ax=axes, vmin=vmin,vmax=vmax)
        # axes.set_title(f'Correlation Matrix Heatmap for {ROI} ({group} scoring group)')

        # fig.savefig(f"../results/rsa_heatmaps/between_subjects/cor_bt_subs_{ROI}_{group}.png")

        # plt.close()

for ROI in ROIs:
    print(f"\n{ROI}: ")
    for group in expertise_groups:
        print(f"Mean for {group} scoring group: {round(mean_of_group_cors[group][ROI]['mean'],4)}, std: {round(mean_of_group_cors[group][ROI]['std'],4)}")