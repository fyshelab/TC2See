import os
import numpy as np
import pandas as pd
import rsatoolbox
from bids import BIDSLayout
from pathlib import Path
import json
import traceback
import h5py
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import KMeans

dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
shared_data_dir = Path(os.path.expanduser('~/projects/def-afyshe-ab/TC2See'))
dataset_layout = BIDSLayout(shared_data_dir / 'bids_data/TC2See')
output_RDM_dir = dataset_root / "processed/predicted_embeddings_RDMs/"

stimulus_images = h5py.File(dataset_root / 'stimulus-images.hdf5', 'r')
stimulus_id_map = {name: i for i, name in enumerate(stimulus_images.attrs['stimulus_names'])}

rdm_dist = "correlation"

embed_types = {
    # "experiment-8.3": { 
    #     "ViT-B=16-features_large.hdf5": ["embedding"],
    #     "ViT-B=32-features_large.hdf5": ["embedding"],
    # },
    "experiment-8.4": { 
        "ViT-B=16-features_large.hdf5": ['A1', 'FFC', 'LO1', 'LO2', 'PIT', 'Pir', 'V1', 'V2', 'V3', 'V4', 'V7', 'V8', 'VVC'],
        "ViT-B=32-features_large.hdf5": ['A1', 'FFC', 'LO1', 'LO2', 'PIT', 'Pir', 'V1', 'V2', 'V3', 'V4', 'V7', 'V8', 'VVC'],
    }
}

all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']


def get_image_order(subject):
    all_stimulus_ids = []
    all_stimulus_names = []

    for run_id in range(6):
        events_file = dataset_layout.get(
            subject=subject, 
            run=run_id + 1,
            task='bird',
            extension='tsv'
        )[0]

        events_df = pd.read_csv(events_file.path, sep='\t')
        events_df = events_df[events_df['stimulus'] != '+']

        stimulus_names = [Path(stimulus_path).stem for stimulus_path in events_df['stimulus']]
        stimulus_ids = [stimulus_id_map[name] for name in stimulus_names]

        all_stimulus_ids.extend(stimulus_ids)
        all_stimulus_names.extend(stimulus_names)

    return all_stimulus_ids, all_stimulus_names



for subject in all_subjects:
    stimulus_ids, stimulus_names = get_image_order(subject)
    for experiment, model_versions in embed_types.items():
        for model_version_file, embed_types_list in model_versions.items():
            model_version = model_version_file.split(".")[0]
            for embed_type in embed_types_list:
                RDM_file_dir = output_RDM_dir / f"individual/sub_{subject}/{experiment}/{model_version}"
                RDM_file_dir.mkdir(parents=True, exist_ok=True)
                output_RDM_file = RDM_file_dir / f"{embed_type}.hdf5"

                print(f"Creating RDM for subject: {subject}, embedding: {experiment}/{model_version_file}/{embed_type}")

                with h5py.File(shared_data_dir / f'decoding/{experiment}/{model_version_file}', 'r') as f:
                    embeddings_pred = f['sub-' + subject][embed_type]['Y_pred'][()] # (450, 512)
                
                embeddings_pred_df = pd.DataFrame(embeddings_pred)
                embeddings_pred_df["stimulus_names"] = stimulus_names
                embed_preds_averaged = embeddings_pred_df.groupby("stimulus_names").mean() # (150, 512)

                unique_images = embed_preds_averaged.index.to_numpy()
                averaged_embeddings = embed_preds_averaged.to_numpy()
                
                data = rsatoolbox.data.Dataset(
                    averaged_embeddings,
                    obs_descriptors={'stimulus_names': unique_images}
                )
                
                rdm = rsatoolbox.rdm.calc_rdm(data, method=rdm_dist)
                rdm.save(output_RDM_file, file_type='hdf5', overwrite=True)