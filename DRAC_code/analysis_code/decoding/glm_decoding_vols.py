from pathlib import Path
import json
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
import h5py
from fracridge import FracRidgeRegressorCV

from tc2see import load_data
from metrics import (
    cosine_distance, two_versus_two,
)
from noise_ceiling import (
    compute_ncsnr,
    compute_nc,
)

# dir2 = os.path.abspath('../..')
# dir1 = os.path.dirname(dir2)
# if not dir1 in sys.path: 
#     sys.path.append(dir1)

dataset_root = os.path.expanduser('~/projects/def-afyshe-ab/TC2See')
james_root = Path(__file__).parent.parent.parent.parent.parent
tc2see_version = 3

accuracies = {}

subjs = ['05']

for subj in tqdm(subjs):
    tr = 2 # 1.97
    subject_no = subj 
    subject = f'sub-{subject_no}'
    

    # bold, stimulus_ids, mask, affine = load_data(
    #     f'{dataset_root}/tc2see-betas_GLMsingle_T1w.hdf5', ################### just put this in ###############
    #     # f'../data/processed/hdf5s/tc2see-v{tc2see_version}-bold.hdf5', 
    #     subject,
    #     tr_offset=6 / tr,
    #     run_normalize='linear_trend',
    #     interpolation=False,
    # )


    model_name = 'ViT-B=32'
    embedding_name = 'embedding' 


    file_path = f'{dataset_root}/tc2see-betas_GLMsingle_T1w.hdf5' 
    h5_file = h5py.File(file_path, 'r')

    # Extract each key as a variable
    betas = h5_file[f'{subject}/betas'][()]
    betas_mean = h5_file[f'{subject}/betas_mean'][()]
    betas_std = h5_file[f'{subject}/betas_std'][()]
    noise_ceiling = h5_file[f'{subject}/noise_ceiling'][()]
    stimulus_ids = h5_file[f'{subject}/stimulus_ids'][()]
    betas_normalized = (betas - betas_mean[:, None]) / betas_std[:, None]
    fmri_mask = h5_file[f'{subject}/fmri_mask'][()]
    # print the number of 1s in the mask
    print(f"Number of 1s in the mask: {np.sum(fmri_mask)}")
    h5_file.close()

    print(f"\nbetas.shape: {betas.shape} \nbetas_mean.shape: {betas_mean.shape} \nbetas_std.shape: {betas_std.shape} \
            \nnoise_ceiling.shape: {noise_ceiling.shape} \nstimulus_ids.shape: {stimulus_ids.shape} \
            \nbetas_normalized.shape: {betas_normalized.shape}\n fmri_mask.shape: {fmri_mask.shape}")

    # load the clip embeddings
    with h5py.File(f'../../data/{model_name}-features.hdf5', 'r') as f:
        stimulus = f[embedding_name][:]
    Y = stimulus[stimulus_ids] # get the stimulus representations to decode


    subject = f'sub-{subject_no}'
    # 6 Runs - 1 run as the test each time (a run is each time the person gets into the scanner and looks into the scanner for a certain amount of time ~ approx 6 mins)
    results = dict
    permutation_test = False
    num_runs = 6

    accuracies_list = []
        
    # Cross validation. Use every id as test data once.
    for test_run_id in tqdm(range(num_runs)):
        training_run_ids = list(range(num_runs))
        training_run_ids.remove(test_run_id) # Remove the test data id 

        bold_train = betas[training_run_ids, :, :]                 # Shape: (5, 75, 200258)
        bold_train = bold_train.reshape(-1, bold_train.shape[-1])  # Shape: (375, 200258)

        stimulus_ids_train = stimulus_ids[training_run_ids, :] # Shape: (5, 75)
        stimulus_ids_train = stimulus_ids_train.reshape(-1)    # Shape: (375,)

        bold_test = betas[test_run_id:test_run_id+1, :, :]  # Shape: (1, 75, 200258)
        bold_test = bold_test.reshape(bold_test.shape[1], bold_test.shape[2]) # Shape: (75, 200258)
        stimulus_ids_test = stimulus_ids[test_run_id:test_run_id+1, :].reshape(-1)   # Shape: (1, 75)
        
        ncsnr = compute_ncsnr(bold_train, stimulus_ids_train) # Compute noise ceiling noise ratio
        nc = compute_nc(ncsnr, num_averages=1)

        nc_vc = nc.copy() 
        # nc_vc[~mask] = 0 # Set values not in mask to zero 
        argsort_ids = np.argsort(-nc_vc) # Default ascending, make descending 
        argsort_ids = argsort_ids[:256] 

        X_train = bold_train[:, argsort_ids]    
        X_nan_train = np.isnan(X_train) # Checks if any not a number values in x and sets those to zero
        X_train[X_nan_train] = 0.

        X_test = bold_test[:, argsort_ids]
        X_nan_test = np.isnan(X_test) # Checks if any not a number values in x and sets those to zero
        X_test[X_nan_test] = 0.

        with h5py.File(f'../../data/{model_name}-features.hdf5', 'r') as f:
            stimulus = f[embedding_name][:]
        Y_train = stimulus[stimulus_ids_train] 
        Y_test = stimulus[stimulus_ids_test]

        if permutation_test:
            ids = np.arange(Y_train.shape[0]) 
            np.random.shuffle(ids)
            Y_train = Y_train[ids]

        model = FracRidgeRegressorCV()
        model.fit(X_train, Y_train) 
        Y_test_pred = model.predict(X_test) # Y_test and Y_test_pred are n x 512 matrics (n is the number of birds).

        distances = cosine_distance( 
            torch.from_numpy(Y_test[None]).float(), 
            torch.from_numpy(Y_test_pred[:, None]).float()
        ) 

        print(f"Distances shape: {distances.shape}")
        stimulus_ids_flat = stimulus_ids.reshape(-1)
        accuracy = round(two_versus_two(distances, stimulus_ids=stimulus_ids_flat).item() * 100, 2) 
        accuracies_list.append(accuracy)

    accuracy_avg = np.mean(accuracies_list)
    accuracies[subj] = accuracy_avg
    print(f"Subject {subj} accuracy: ", accuracy_avg)

print(accuracies)

with open('GLM_accuracies_vols.json', 'w') as json_file:
    json.dump(accuracies, json_file, indent=4)