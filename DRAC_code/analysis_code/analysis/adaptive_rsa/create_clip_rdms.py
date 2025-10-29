import os
import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
from bids import BIDSLayout
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
import json
import traceback
from sklearn.linear_model import HuberRegressor
from matplotlib.ticker import FormatStrFormatter
import h5py


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
shared_dataset_root = os.path.expanduser('~/projects/def-afyshe-ab/TC2See')
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
dataset_layout = BIDSLayout(shared_dataset_root + '/bids_data/TC2See')

stimulus_images = h5py.File(dataset_root / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {name: i for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}
stimulus_name_map = {i: name for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}

image_order = []
all_stimulus_names = []

for run_id in range(6):
    events_file = dataset_layout.get(
        subject="05", # Stimuli are the same for every participant
        run=run_id + 1,
        task='bird',
        extension='tsv'
    )[0]

    events_df = pd.read_csv(events_file.path, sep='\t')
    events_df = events_df[events_df['stimulus'] != '+']
    stimulus_names = [Path(stimulus_path).stem for stimulus_path in events_df['stimulus']]
    stimulus_ids = [stimulus_id_map[name] for name in stimulus_names]

    image_order.extend(stimulus_ids)
    all_stimulus_names.extend(stimulus_names)

image_order = set(image_order)

# mod_im_combos = [("ViT-B=32", "bird"), ("ViT-B=32", "fg_mask"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "max_dist")]
mod_im_combos = [("ViT-B=32", "bird")]

for model_name, image_type in mod_im_combos:
    # CLIP Embeddings
    with h5py.File(dataset_root / f'{model_name}-features_{image_type}.hdf5', 'r') as f:
        clip_embeddings = f['embedding'][:]

    stimuli_idx_list = sorted(image_order)
    filtered_clip_embeddings = np.array([embedding for e, embedding in enumerate(clip_embeddings) if e in stimuli_idx_list])

    RDM_path = dataset_root / f"processed/glm_RDMs/CLIP_RDMs"
    rdm_file_path = RDM_path / f'rdms_{model_name}_{image_type}.hdf5'

    data = rsatoolbox.data.Dataset(
        filtered_clip_embeddings,
        obs_descriptors={'stim_ids': stimuli_idx_list}
    )
    
    rdm = rsatoolbox.rdm.calc_rdm(data, method='correlation')
    rdm.save(rdm_file_path, file_type='hdf5', overwrite=True)

