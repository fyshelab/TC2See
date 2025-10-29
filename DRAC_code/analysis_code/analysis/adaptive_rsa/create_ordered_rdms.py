import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import json
import traceback
import h5py
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import KMeans

dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

avg_duplicates = True
rdm_dist = "correlation"
mod_im_combos = [("ViT-B=32", "bird"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "fg_mask"), ("ViT-B=32", "max_dist")]
model_name, image_type = mod_im_combos[3]  

# Load ROI name mapping
with open(dataset_root / "roi_names.json", 'r') as f:
    roi_names = json.load(f)

with h5py.File(dataset_root / f'{model_name}-features_{image_type}.hdf5', 'r') as f:
    clip_embeddings = f['embedding'][:]


# Cluster the embeddings to find similar groups 

# kmeans = KMeans(n_clusters=9, random_state=42)
# agglomerative = AgglomerativeClustering(n_clusters=5, random_state=42)
# cluster_labels = agglomerative.fit_predict(clip_embeddings).tolist()

linkage_matrix = linkage(clip_embeddings, method='ward')
dendro_order = leaves_list(linkage_matrix)
stimulus_id_order = [str(i) for i in dendro_order]


# ROIs = [roi_names[str(i)] for i in range(1, 181)]
ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]
all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']


for subject in all_subjects:

    ordered_RDM_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}_ordered/sub_{subject}"

    for ROI in ROIs:
        print(f"Creating RDM for subject: {subject}, ROI: {ROI}")
        roi_representations_path = dataset_root / f"processed/glm_roi_representations/sub_{subject}" / ROI
        rdm_roi_path = ordered_RDM_path / ROI
        rdm_file_path = rdm_roi_path / f'rdm_for_{ROI}.hdf5'
        
        rdm_roi_path.mkdir(parents=True, exist_ok=True)
        sub_roi_data = pd.read_parquet(roi_representations_path / f'reps_for_{ROI}.parquet')
        
        if avg_duplicates:
            sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: 1 if x == "Sparrow" else 2)
            sub_roi_data = sub_roi_data.groupby('stimulus_id').mean().reset_index()
            sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: "Sparrow" if x == 1 else "Warbler")
    
        # sub_roi_data['cluster'] = [cluster_labels[int(stimulus_id)] for stimulus_id in sub_roi_data['stimulus_id']]
        stimulus_order_map = {stim_id: i for i, stim_id in enumerate(stimulus_id_order)}
        sub_roi_data['sort_idx'] = sub_roi_data['stimulus_id'].astype(int).map(lambda x: dendro_order.tolist().index(x))
        sub_roi_data = sub_roi_data.sort_values(by='sort_idx')

        stim_category = list(sub_roi_data['stimulus_category'])
        stim_ids = list(sub_roi_data['stimulus_id'])

        sub_roi_data = sub_roi_data.drop(columns=['stimulus_id', 'stimulus_category', 'sort_idx'])
        
        data = rsatoolbox.data.Dataset(
            sub_roi_data.to_numpy(),
            obs_descriptors={'stim_category': stim_category, 'stim_ids': stim_ids}
        )
        rdm = rsatoolbox.rdm.calc_rdm(data, method='correlation')
        rdm_matrix = rdm.get_matrices()[0] 

        rdm_df = pd.DataFrame(rdm_matrix, index=data.obs_descriptors["stim_ids"], columns=data.obs_descriptors["stim_ids"])
        
        rdm.save(rdm_file_path, file_type='hdf5', overwrite=True)