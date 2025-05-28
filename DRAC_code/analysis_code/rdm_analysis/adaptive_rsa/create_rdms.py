import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import json

dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

avg_duplicates = True
rdm_dist = "correlation"


ROIs = [roi_names[str(i)] for i in range(1, 181)]
all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']


# Load ROI name mapping
with open(dataset_root / "roi_names.json", 'r') as f:
    roi_names = json.load(f)


for subject in all_subjects:
    for ROI in ROIs:
        try:
            roi_path = dataset_root / f"processed/glm_roi_representations/sub_{subject}" / ROI
            rdm_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}/sub_{subject}" / ROI
            rdm_file_path = rdm_path / f'rdm_for_{ROI}.hdf5'
            
            if not rdm_path.exists():
                rdm_path.mkdir(parents=True, exist_ok=True)
                sub_roi_data = pd.read_parquet(roi_path / f'reps_for_{ROI}.parquet')
                
                # Stimulus IDs are sorted at this point, so we can use them in rdm correlations
                if avg_duplicates:
                    sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: 1 if x == "Sparrow" else 2)
                    sub_roi_data = sub_roi_data.groupby('stimulus_id').mean().reset_index()
                    sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: "Sparrow" if x == 1 else "Warbler")
                
                stim_category = list(sub_roi_data['stimulus_category'])
                stim_ids = list(sub_roi_data['stimulus_id'])
                sub_roi_data = sub_roi_data.drop(columns=['stimulus_id', 'stimulus_category'])
                
                data = rsatoolbox.data.Dataset(
                    sub_roi_data.to_numpy(),
                    obs_descriptors={'stim_category': stim_category, 'stim_ids': stim_ids}
                )
                rdm = rsatoolbox.rdm.calc_rdm(data, method='correlation')
                rdm_matrix = rdm.get_matrices()[0] 
                rdm_df = pd.DataFrame(rdm_matrix, index=data.obs_descriptors["stim_ids"], columns=data.obs_descriptors["stim_ids"])
                
                print(f"Saving RDM for subject {subject}, ROI: {ROI}")
                rdm.save(rdm_file_path, file_type='hdf5', overwrite=True)

        except Exception as e:
            print(f"Error processing subject {subject}, ROI: {ROI} - {e}")
            continue