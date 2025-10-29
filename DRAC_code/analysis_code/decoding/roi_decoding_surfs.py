from pathlib import Path
import os
import json
import numpy as np
import torch
from tqdm import tqdm
# from scipy.ndimage import zoom, binary_dilation
import h5py
import nibabel as nib
from fracridge import FracRidgeRegressorCV
from sklearn.linear_model import Ridge

from tc2see import load_data
from metrics import (
    cosine_distance, two_versus_two,
)

from noise_ceiling import (
    compute_ncsnr,
    compute_nc,
)

tc2see_version = 3
output_file_suffix = "40"

ROIs = {
    "FFC": [18],"V1": [1],"V2": [4],"V3": [5],"V3A": [13],"V3B": [19],"V3CD": [158],"V4": [6],"V6": [3],"V7": [16],
    "V8": [7], "VMV1": [153],"VMV2": [160],"VMV3": [154],"LO1": [20],"LO2": [21],"PIT": [22],"VVC": [163], 
    "LO1": [20],"LO2": [21], "LO3": [159], "PHA1": [126], "PHA2": [155], "PHA3": [127], "IPS1": [17], "MT": [23]
}


ROI_combos = {
    "all_rois": [ roi for roi in range(1,181)],

    # High level Visual
    "FFC": [18], 
    "VVC": [163], 
    "LO1": [20], "LO2": [21], "LO3": [159], "PHA1": [126], 
    "PHA2": [155], "PHA3": [127], "IPS1": [17], "MT": [23],
    "LOC1_to_LOC3": [20, 21, 159],
    "PHA1_to_PHA3": [126, 155, 127],

    # Low level Visual
    "V1": [1],
    "V2": [4],
    "V3_V3A_V3B_V3CD": [5, 13, 19, 158],
    "V4": [6],
    "V6": [3],
    "V7": [16],
    "V8": [7]
}


glasser_L = nib.freesurfer.io.read_annot("../../data/lh.HCPMMP1.annot")
glasser_R = nib.freesurfer.io.read_annot("../../data/rh.HCPMMP1.annot")

ROI_masks = {}

for key, vals in ROI_combos.items():

    # mask glasser atlas to mark current loop ROI as 1s
    L_mask = np.isin(glasser_L[0], vals) # vals is a list of ROIs to set as 1
    R_mask = np.isin(glasser_R[0], vals)
    print(f"{key}: {sum(L_mask) + sum(R_mask)}")
    
    # concatenate left and right hemispheres 
    L_R_concat_mask = np.concatenate([L_mask, R_mask], axis=0)
    ROI_masks[key] = L_R_concat_mask

model_name = 'ViT-B=32'
embedding_name = 'embedding' 
num_runs = 6
tr = 2 # 1.97

with h5py.File(f'../../data/{model_name}-features.hdf5', 'r') as f:
    stimulus = f[embedding_name][:]

load_data_params = dict(
    path = f'../../data/processed/hdf5s/tc2see-v{tc2see_version}-fsaverage-surfs.hdf5', 
    tr_offset = num_runs / tr,
    run_normalize='linear_trend',
    interpolation=False,
)

accuracies = {}
# subjs = [str(sub) if sub >= 10 else '0'+str(sub) for sub in range(5,40)] 
subjs = [40]
top_n_nc_vals = 256

for subj in tqdm(subjs):
    print(f"====== Subject {subj} ======")
    accuracies[subj] = {}
    try:
        subject = f'sub-{subj}'

        # stimulus_ids are needed for the two_versus_two accuracy scores below
        _, stimulus_ids = load_data(
            f'../../data/processed/hdf5s/tc2see-v{tc2see_version}-fsaverage-surfs.hdf5', 
            subject,
            tr_offset= num_runs / tr,
            run_normalize='linear_trend',
            interpolation=False,
        )

        accuracies_list = []
        std_list = []

        for ROI, ROI_mask in ROI_masks.items():
            print(f"- Decoding {ROI}...")
            accuracies[subj][ROI] = {}
            ROI_accuracies_list = []
            
            # Cross validation. Use every run as test data once.
            for test_run_id in range(num_runs):
                training_run_ids = list(range(num_runs))
                training_run_ids.remove(test_run_id) # Remove the test data id 

                # load the training and test data
                bold_train, stimulus_ids_train = load_data(
                    **load_data_params,
                    subject = subject,
                    run_ids = training_run_ids
                )  

                bold_test, stimulus_ids_test = load_data(
                    **load_data_params,
                    subject = subject,
                    run_ids = [test_run_id]
                )

                ncsnr = compute_ncsnr(bold_train, stimulus_ids_train) # Compute noise ceiling noise ratio
                nc = compute_nc(ncsnr, num_averages=1)

                nc[~ROI_mask] = 0
                nc_list = list(nc)  # Convert nc to a Python list
                indices = list(range(len(nc_list)))
                argsort_ids = sorted( indices, key=lambda i: -nc_list[i] )

                if np.count_nonzero(nc) < top_n_nc_vals:
                    argsort_ids = argsort_ids[:np.count_nonzero(nc)] 
                else:
                    argsort_ids = argsort_ids[:top_n_nc_vals] 

                X_train = bold_train[:, argsort_ids] 
                X_nan_train = np.isnan(X_train) # Checks if any not a number values in x and sets those to zero
                X_train[X_nan_train] = 0.

                X_test = bold_test[:, argsort_ids]
                X_nan_test = np.isnan(X_test) # Checks if any not a number values in x and sets those to zero
                X_test[X_nan_test] = 0.

                Y_train = stimulus[stimulus_ids_train] 
                Y_test = stimulus[stimulus_ids_test]

                model = FracRidgeRegressorCV()
                model.fit(X_train, Y_train)
                Y_pred = model.predict(X_test) 

                y_test_array = torch.from_numpy(Y_test[None]).double()       
                y_pred_array = torch.from_numpy(Y_pred[:, None]).double()

                distances = cosine_distance(
                    y_test_array,      
                    y_pred_array
                ) 

                accuracy = round(two_versus_two(distances, stimulus_ids=stimulus_ids).item() * 100, 2) 
                ROI_accuracies_list.append(accuracy)
            
            # get average of accuracies for this iteration of runs
            ROI_accuracy_avg = np.mean(ROI_accuracies_list)
            accuracies[subj][ROI] = round(ROI_accuracy_avg, 3)
            print(f"   Mean Accuracy: {round(ROI_accuracy_avg, 3)}")

            # get standard deviation of accuracies for this iteration of runs
            ROI_accuracy_std = np.std(ROI_accuracies_list)
            print(f"   Std: {round(ROI_accuracy_std, 3)}\n")

            # save accuracies checkpoint
            with open(f'ROI_accuracies_surfs{output_file_suffix}.json', 'w') as json_file:
                json.dump(accuracies, json_file, indent=4)

    except Exception as e:
        print(f"There was an error for subject {subj}: ", e)

print(accuracies)