from pathlib import Path
import os
import numpy as np
import h5py
import nibabel as nib
import pandas as pd


dataset_root = os.path.expanduser('~/projects/def-afyshe-ab/TC2See')
james_data_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
stimulus_images = h5py.File(james_data_dir / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {i: name for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}

subjects = [ sub for sub in range(5,41) if sub not in [13]]
subject_strs = [ '0'+str(sub) if sub < 10 else str(sub) for sub in subjects]


ROIs = {
    "V1": [1], "V2": [4], "V3": [5], "V4": [6],"V6": [3],"V7": [16], "V8": [7],
    "LO1": [20],"LO2": [21], "LO3": [159], "PIT": [22], "FFC": [18], "VVC": [163] 
}


for subject_str in subject_strs:
    try:
        subject = f'sub-{subject_str}'
        print(f"Saving data for subject {subject_str}...")

        # Load the GLM data
        betas_file_path = f'{dataset_root}/tc2see-betas_GLMsingle_T1w.hdf5'
        h5_file = h5py.File(betas_file_path, 'r')

        # Extract each key as a variable
        betas = h5_file[f'{subject}/betas'][()] # (6, 75, 185048) 
        betas_mean = h5_file[f'{subject}/betas_mean'][()] # (6, 185048)
        betas_std = h5_file[f'{subject}/betas_std'][()] # (6, 185048)
        stimulus_ids = h5_file[f'{subject}/stimulus_ids'][()] # (6, 75)
        betas_normalized = (betas - betas_mean[:, None]) / betas_std[:, None] # (6, 75, 185048)
        fmri_mask = h5_file[f'{subject}/fmri_mask'][()] # (67, 80, 68)
        h5_file.close()

        # Load the ROI mask
        roi_mask_file = f'{dataset_root}/fmri_prep_vols_v2/derivatives/{subject}/{subject}_HCP_MMP1_dilated2.nii.gz'
        roi_mask = nib.load(roi_mask_file)
        roi_mask = roi_mask.get_fdata()

        # Filter the ROI mask to only include the voxels from the fMRI mask
        roi_mask = roi_mask[fmri_mask] # (185048,) Still includes zero values for ROIs

        # Only keep values in roi_mask that are not zero
        rois_mask = roi_mask[roi_mask != 0] 
        ROI_masks = {key: np.isin(rois_mask, vals) for key, vals in ROIs.items()}
        roi_mask_boolean = roi_mask != 0

        # Filter normalized betas to only include ROIs that are not 0
        betas_normalized = betas_normalized[:, :, roi_mask_boolean] # (6, 75, 185048) --> (6, 75, 96640)
        betas_normalized_stacked = betas_normalized.reshape(-1, betas_normalized.shape[2])

        glm_df = pd.DataFrame(betas_normalized_stacked)
        glm_df.columns = glm_df.columns.astype(str) 

        stimulus_ids = stimulus_ids.reshape(-1)
        glm_df['stimulus_id'] = stimulus_ids
        glm_df['stimulus_category'] = glm_df['stimulus_id'].apply(lambda x: "Sparrow" if "Sparrow" in stimulus_id_map[x] else "Warbler")

        numeric_glm_df = glm_df.drop(columns=['stimulus_id', 'stimulus_category'])
        stim_ids = glm_df['stimulus_id'].values
        stim_cats = glm_df['stimulus_category'].values
                
        # Save ROI Representations
        for ROI, ROI_mask in ROI_masks.items():
            roi_path =  james_data_dir / f"processed/glm_roi_representations/sub_{subject_str}" / ROI

            if not roi_path.exists():
                roi_path.mkdir(parents=True, exist_ok=True)
                glm_file_name = roi_path / f"reps_for_{ROI}.parquet"
                
                glm_df_tmp = numeric_glm_df.loc[:, ROI_mask].copy()
                glm_df_tmp['stimulus_id'] = stim_ids
                glm_df_tmp['stimulus_category'] = stim_cats

                glm_df_tmp.to_parquet(glm_file_name, index=False)

    except Exception as e:
        print(f"Error processing subject {subject_str}: {e}")
        continue