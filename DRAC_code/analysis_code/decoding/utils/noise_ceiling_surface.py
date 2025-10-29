import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import traceback

import nibabel as nib
from nilearn import surface
import bids
from bids import BIDSLayout

from tc2see import load_data

shared_data_dir = Path(os.path.expanduser('~/projects/def-afyshe-ab/TC2See'))
data_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")

tc2see_version = 3 
derivatives_path = shared_data_dir / 'fmri_prep_surfs'
num_runs = 6 if tc2see_version in (1, 3) else 8

# Initialize BIDSLayouts for querying files.
dataset_layout = BIDSLayout(shared_data_dir / 'bids_data/TC2See')
derivatives_layout = BIDSLayout(derivatives_path, derivatives=True, validate = False)

task = "bird"
space = 'fsaverage' 
subjects = ["10"] 
# subjects = ["05", "06", "07", "08", "09", "10", "11", "12", "14", "15", "16", "17", "18", "19",
#             "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", 
#             "34", "35", "36", "37", "38", "39", "40"] 

tr = 2. # 1.97  # TR duration (in seconds)
num_stimuli = 75 # 112  # Total number of different stimuli

# Load stimulus images and create a mapping of stimulus names to unique identifiers
stimulus_images = h5py.File(data_root / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {name: i for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}
 
new_or_append = 'a' # Use 'a' for append/overwrite, 'w' for new hdf5 file
           
# Create or append to an HDF5 file to store preprocessed fMRI data
with h5py.File(shared_data_dir / f'hdf5s/tc2see-fsaverage-surfs.hdf5', new_or_append) as f:
    for subject in tqdm(subjects):
        if f'sub-{subject}' not in list(f.keys()):
            try:
                print(f"Processing subject {subject}...")
                group = f.require_group(f'sub-{subject}')

                fsaverage_surf_list = []
                for hemi in ('L', 'R'):
                    
                    leftOrRight = 0 if hemi == 'L' else 1
                
                    fsaverage_surf_hemi = derivatives_layout.get(
                            subject=subject,
                            run=1,
                            task=task,
                            space=space, 
                            extension='func.gii'
                    )[leftOrRight]

                    fsaverage_surf_hemi = surface.load_surf_data(fsaverage_surf_hemi).astype(np.float64)
                    fsaverage_surf_list.append(fsaverage_surf_hemi)

                fsaverage_surf = np.concatenate(fsaverage_surf_list, axis=0)

                num_voxels = fsaverage_surf.shape[0]
                num_trs = fsaverage_surf.shape[1]

                group.require_dataset('bold', shape=(num_runs, num_trs, num_voxels), dtype='f4')
                group.require_dataset('bold_mean', shape=(num_runs, num_voxels), dtype='f4')
                group.require_dataset('bold_std', shape=(num_runs, num_voxels), dtype='f4')
                group.require_dataset('bold_trend', shape=(num_runs, 2, num_voxels), dtype='f4')
                group.require_dataset('bold_trend_std', shape=(num_runs, num_voxels), dtype='f4')
                group.require_dataset('stimulus_trs', shape=(num_runs, num_stimuli), dtype='f4')
                group.require_dataset('stimulus_ids', shape=(num_runs, num_stimuli), dtype='i4')
                
                for run_id in tqdm(range(num_runs)):
                    
                    fsaverage_surf_list = []
                    for hemi in ('L', 'R'):
                        
                        leftOrRight = 0 if hemi == 'L' else 1
                    
                        fsaverage_surf_hemi = derivatives_layout.get(
                                subject=subject,
                                run=run_id + 1,
                                task=task,
                                space=space, 
                                extension='func.gii'
                        )[leftOrRight]

                        fsaverage_surf_hemi = surface.load_surf_data(fsaverage_surf_hemi).astype(np.float64)
                        fsaverage_surf_list.append(fsaverage_surf_hemi)

                    fsaverage_surf = np.concatenate(fsaverage_surf_list, axis=0) # (327684, 231)
                    fsaverage_surf = np.transpose(fsaverage_surf) # (231, 327684)                    
                    num_trs_run = fsaverage_surf.shape[0]

                    trend_coeffs = np.stack([np.arange(num_trs_run), np.ones(shape=num_trs_run)], axis=1) # (231, 2)
                    
                    # Perform linear detrending on the bold data
                    bold_trend = np.linalg.lstsq(trend_coeffs, fsaverage_surf, rcond=None)[0] # (2, 327684)
                    bold_predicted = trend_coeffs @ bold_trend # (231, 327684)                    
                    bold_detrend = fsaverage_surf - bold_predicted # (231, 327684)

                    # Load events data for the current subject and run
                    events_file = dataset_layout.get(
                        subject=subject,
                        run=run_id + 1,
                        task=task,
                        extension='tsv'
                    )[0]
                    
                    events_df = pd.read_csv(events_file.path, sep='\t')
                    events_df = events_df[events_df['stimulus'] != '+']
                    stimulus_names = [Path(stimulus_path).stem for stimulus_path in events_df['stimulus']]
                    stimulus_names = [
                        name[:name.find('hash')-1] if "hash" in name else name
                        for name in stimulus_names
                    ]
                    stimulus_ids = [stimulus_id_map[name] for name in stimulus_names]
                    
                    stimulus_trs = np.array(events_df['tr']).astype(np.float32)
                    
                    # Store various datasets in the HDF5 file
                    group['bold'][run_id, :num_trs_run] = fsaverage_surf
                    group['bold_mean'][run_id] = fsaverage_surf.mean(axis=0)
                    group['bold_std'][run_id] = fsaverage_surf.std(axis=0)
                    group['bold_trend'][run_id] = bold_trend
                    group['bold_trend_std'][run_id] = bold_detrend.std(axis=0)
                    group['stimulus_trs'][run_id] = stimulus_trs
                    group['stimulus_ids'][run_id] = stimulus_ids
                
            except Exception as e:
                print(f"Error processing {subject}: \n")
                traceback.print_exc()
                del f[f'sub-{subject}']
                continue
        else:
            print(f"Subject {subject} already exists")
            print(f[f'sub-{subject}']['bold'].shape)