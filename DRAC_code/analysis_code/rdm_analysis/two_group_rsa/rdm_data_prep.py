import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
import nibabel as nib

dir2 = os.path.abspath('..')
dir1 = os.path.dirname(dir2)
tc2see_dir = Path('/home/jamesmck/projects/def-afyshe-ab/jamesmck/bird_data_analysis/analysis_code/decoding/')
if not dir1 in sys.path: 
    sys.path.append(dir1)
    sys.path.append(str(tc2see_dir))

from tc2see import load_data

subjects = [ sub for sub in range(5,41) if sub not in [13]]
subject_strs = [ '0'+str(sub) if sub < 10 else str(sub) for sub in subjects]

ROIs = {
    "FFC": [18], "VVC": [163], "LO1": [20],"LO2": [21], "LO3": [159], "PHA1": [126], "PHA2": [155], "PHA3": [127], "IPS1": [17], "MT": [23]
}

dataset_root = Path('../../data')
derivatives_path = dataset_root / 'processed/fmriprep_surfs'

stimulus_images = h5py.File(dataset_root / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {i: name for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}
images_dir = dataset_root / Path("cropped")

tc2see_version = 3
tr = 2
num_runs = 6

load_data_params = dict(
    path = dataset_root / f'processed/hdf5s/tc2see-v{tc2see_version}-fsaverage-surfs.hdf5', 
    tr_offset = num_runs / tr,
    run_normalize='linear_trend',
    interpolation=False,
)

glasser_L = nib.freesurfer.io.read_annot("../../data/lh.HCPMMP1.annot")
glasser_R = nib.freesurfer.io.read_annot("../../data/rh.HCPMMP1.annot")

ROI_masks = {}

for key, vals in ROIs.items():

    # mask glasser atlas to mark current loop ROI as 1s
    L_mask = np.isin(glasser_L[0], vals) # vals is a list of ROIs to set as 1
    R_mask = np.isin(glasser_R[0], vals)
    
    # concatenate left and right hemispheres 
    L_R_concat_mask = np.concatenate([L_mask, R_mask], axis=0)
    ROI_masks[key] = L_R_concat_mask


for subject_str in subject_strs:
    print(f"Saving data for subject {subject_str}...")

    bold_run, stimulus_ids = load_data(
        **load_data_params,
        subject = f'sub-{subject_str}',
        run_ids = list(range(num_runs))
    )
    bold_run = pd.DataFrame(bold_run)
    bold_run.columns = bold_run.columns.astype(str)


    bold_run['stimulus_ids'] = stimulus_ids
    bold_run['stimulus_category'] = bold_run['stimulus_ids'].apply(lambda x: "Sparrow" if "Sparrow" in stimulus_id_map[x] else "Warbler")

    numeric_bold_run = bold_run.drop(columns=['stimulus_ids', 'stimulus_category'])
    stim_ids = bold_run['stimulus_ids'].values
    stim_cats = bold_run['stimulus_category'].values
            
    # Save ROI Representations
    for ROI, ROI_mask in ROI_masks.items():
        roi_path =  dataset_root / f"processed/roi_representations/sub_{subject_str}" / ROI

        if not roi_path.exists():
            roi_path.mkdir(parents=True, exist_ok=True)
            bold_file_name = roi_path / f"reps_for_{ROI}.parquet"
            
            bold_run_tmp = numeric_bold_run.loc[:, ROI_mask].copy()
            bold_run_tmp['stimulus_ids'] = stim_ids
            bold_run_tmp['stimulus_category'] = stim_cats

            bold_run_tmp.to_parquet(bold_file_name, index=False)

