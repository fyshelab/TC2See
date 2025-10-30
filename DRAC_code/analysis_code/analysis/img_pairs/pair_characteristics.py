import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
import json
import traceback
import h5py
from sklearn.linear_model import HuberRegressor
from matplotlib.ticker import FormatStrFormatter
from PIL import Image


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

bird_img_annotations = pd.read_csv(dataset_root / "Bird_Image_Annotations.csv")
rdm_dist = "correlation"
version = "top_5_new" # top_5 or "top_5_avg_tie" or "low_high" or "top_5_new"

ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]
all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

if version == "low_high":
    expertise_groups = {
        'low':  ['06', '15', '20', '21', '22', '23', '28', '29', '32', '33', '39'],
        'high': ['08', '09', '11', '12', '16', '24', '26', '35', '38'],
    }
elif version == "top_5":
    expertise_groups = {
        'low':  ["29", "32", "28", "15", "20"], 
        'high': ["08", "26", "12", "24", "09"], 
    }
elif version == "top_5_avg_tie":
    expertise_groups = {
        'low':  ["32", "29", "28", "15", "23", "21", "20"], 
        'high': ["08", "26", "12", "24", "09"],
    }
elif version == "top_5_new":
    expertise_groups = {
        'low':  ["29", "21", "15", "28", "23"], 
        'high': ["08", "38", "26", "09", "35"],
    }

quiz_scores_file = "participant_quiz_scores.json" if version != "top_5_new" else "participant_new_scores.json"
with open(dataset_root / quiz_scores_file, 'r') as f:
    expertise_scores = json.load(f)


with h5py.File(dataset_root / f'stimulus-images.hdf5', 'r') as f:
    stimulus_names = list(f.keys())

# Initialize dictionary to store RDMs
RDM_dict = {subject: {} for subject in all_subjects}

# Load RDMs for each subject and ROI
for subject in all_subjects:
    try:
        for ROI in ROIs:
            if version == "pred_embed_":
                rdm_path = dataset_root / f"processed/predicted_embeddings_RDMs/individual/sub_{subject}/experiment-8.4/ViT-B=16-features_large"
                rdm_file_path = rdm_path / f'{ROI}.hdf5'
            else:
                rdm_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}/sub_{subject}" / ROI
                rdm_file_path = rdm_path / f'rdm_for_{ROI}.hdf5'
            
            rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
            RDM_dict[subject][ROI] = rdm
    except Exception as e:
        print(f"Error processing subject {subject}, ROI: {ROI}")
        continue


roi_path = dataset_root / f"processed/glm_roi_representations/sub_05" / "V1"
sub_roi_data = pd.read_parquet(roi_path / f'reps_for_V1.parquet')
sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: 1 if x == "Sparrow" else 2)
sub_roi_data = sub_roi_data.groupby('stimulus_id').mean().reset_index()
stim_ids = list(sub_roi_data['stimulus_id'])
bird_images = [stimulus_names[i] + ".png" for i in stim_ids]


def get_rdm_upper_triangles(subject_list, roi, rdm_dict):
    num_subjects = len(subject_list)
    rdm_arrays = []
    for subject in subject_list:
        if subject in rdm_dict and roi in rdm_dict[subject]:
            rdm = rdm_dict[subject][roi]
            rdm_arrays.append(rdm.dissimilarities)

    rdm_dissimilarities = [np.array(rdm).flatten() for rdm in rdm_arrays]
    n_items = int(np.sqrt(2 * len(rdm_dissimilarities[0]) + 0.25) + 0.5)

    dissim_matrices = np.zeros((num_subjects, n_items, n_items))
    
    for sub in range(num_subjects):
        idx = 0
        for i in range(n_items):
            for j in range(i + 1, n_items):
                dissim_matrices[sub, i, j] = float(rdm_dissimilarities[sub][idx])
                dissim_matrices[sub, j, i] = float(rdm_dissimilarities[sub][idx])
                idx += 1

    upper_tri_per_sub = {}
    for sub in range(num_subjects):
        upper_tri_per_sub[sub] = []
        for i in range(n_items):
            for j in range(i + 1, n_items):
                upper_tri_per_sub[sub].append((dissim_matrices[sub, i, j], i, j))
        upper_tri_per_sub[sub].sort(key=lambda x: x[0])

    return upper_tri_per_sub

def get_image_characteristics(subject_list, upper_tri_dict, bird_img_annotations, bird_images):

    df_rows = []

    for sub in upper_tri_dict:
        for (dissimilarity, img1_idx, img2_idx) in upper_tri_dict[sub]:
            temp_dict = {}
            temp_dict['subject'] = subject_list[sub]
            temp_dict['dissimilarity'] = dissimilarity
            temp_dict['image1'] = bird_images[img1_idx]
            temp_dict['image2'] = bird_images[img2_idx]
  
            img1_annotations = bird_img_annotations[bird_img_annotations['File Name'] == bird_images[img1_idx]].iloc[0]
            img2_annotations = bird_img_annotations[bird_img_annotations['File Name'] == bird_images[img2_idx]].iloc[0]

            for col in bird_img_annotations.columns:
                if col != 'File Name':
                    col_str = col.replace(" ", "_")
                    temp_dict[f'img1_{col_str}'] = img1_annotations[col]
                    temp_dict[f'img2_{col_str}'] = img2_annotations[col]

            df_rows.append(temp_dict)
  

    df = pd.DataFrame(df_rows)
    # sort by accesnding dissimilarity
    df = df.sort_values(by='dissimilarity', ascending=True).reset_index(drop=True)
    return df


for ROI in ROIs:
    print(f"\n------ ROI: {ROI} ------")

    low_expertise_subjects = expertise_groups['low']
    high_expertise_subjects = expertise_groups['high']
    
    # Get upper triangle of subject RDM {subject: [ (dissimilarity, img1_idx, img2_idx) ]}
    low_expertise_u_tris= get_rdm_upper_triangles(low_expertise_subjects, ROI, RDM_dict)
    if version == "top_5_avg_tie":
        # Average the last 3 subjects' dissimilarities
        n_subjects = len(low_expertise_u_tris)
        n_comparisons = len(low_expertise_u_tris[0])
        
        # Create a new dictionary to store the result
        new_low_expertise_u_tris = {}
        
        # Keep the first subjects as-is
        for sub in range(n_subjects - 3):
            new_low_expertise_u_tris[sub] = low_expertise_u_tris[sub]
        
        # Average the last 3 subjects
        averaged_dissimilarities = []
        for comp_idx in range(n_comparisons):
            # Get the dissimilarity values from the last 3 subjects for this comparison
            dissim_sum = 0
            i_idx = low_expertise_u_tris[n_subjects - 3][comp_idx][1]
            j_idx = low_expertise_u_tris[n_subjects - 3][comp_idx][2]
            
            for sub in range(n_subjects - 3, n_subjects):
                dissim_sum += low_expertise_u_tris[sub][comp_idx][0]
            
            avg_dissim = dissim_sum / 3
            averaged_dissimilarities.append((avg_dissim, i_idx, j_idx))
        
        # Sort by dissimilarity (to maintain the sorted order)
        averaged_dissimilarities.sort(key=lambda x: x[0])
        
        # Add the averaged dissimilarities as the 5th subject
        new_low_expertise_u_tris[n_subjects - 3] = averaged_dissimilarities
        
        # Update the low_expertise_u_tris
        low_expertise_u_tris = new_low_expertise_u_tris
        
        # Update the subject list to reflect only 5 subjects
        low_expertise_subjects = low_expertise_subjects[:4] + ['tied_avg']
    
    high_expertise_u_tris = get_rdm_upper_triangles(high_expertise_subjects, ROI, RDM_dict)
    
    low_exp_DF = get_image_characteristics(low_expertise_subjects, low_expertise_u_tris, bird_img_annotations, bird_images)
    high_exp_DF = get_image_characteristics(high_expertise_subjects, high_expertise_u_tris, bird_img_annotations, bird_images)

    roi_results_dir = results_dir / f"img_pair_DFs/{ROI}/{version}"
    roi_results_dir.mkdir(parents=True, exist_ok=True)
    low_exp_DF.to_parquet(roi_results_dir / f'low_exp_DF.parquet', index=False)
    high_exp_DF.to_parquet(roi_results_dir / f'high_exp_DF.parquet', index=False)