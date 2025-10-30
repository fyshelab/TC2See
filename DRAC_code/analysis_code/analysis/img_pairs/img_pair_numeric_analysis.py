import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
import json
import traceback
import h5py
from sklearn.linear_model import HuberRegressor
from matplotlib.ticker import FormatStrFormatter
from PIL import Image
from matplotlib.ticker import PercentFormatter


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

rdm_dist = "correlation"
bird_img_annotations = pd.read_csv(dataset_root / "Bird_Image_Annotations.csv")
filenames = bird_img_annotations['File Name'].astype(str).values
n_images = len(filenames)


all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)

ROI_group = "all" # v1-v4_Cntrl or all

if ROI_group == "v1-v4_Cntrl":
    ROIs = ["V1", "V2", "V3", "V4", "A1", "Pir"] 
elif ROI_group == "all":
    ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]
else:
    ROIs = []

img_variables = [('Head_Direction', 'head direction'), ('Species', 'species'), ('Sub_Species', 'sub species'), 
                 ('Branches', 'branches'), ('Leaves', 'leaves'), ('Grass', 'grass'), ('Bg_Focused', 'bg focused'), 
                 ('Beak_Open', 'beak open')]


# Load RDMs for each subject and ROI
RDM_dict = {subject: {} for subject in all_subjects}

# Load RDMs for each subject and ROI
for subject in all_subjects:
    try:
        for ROI in ROIs:
            rdm_path = dataset_root / f"processed/glm_RDMs/{rdm_dist}/sub_{subject}" / ROI
            rdm_file_path = rdm_path / f'rdm_for_{ROI}.hdf5'
            
            rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
            RDM_dict[subject][ROI] = rdm
    except Exception as e:
        print(f"Error processing subject {subject}, ROI: {ROI}")
        continue


# Get expertise scores for subjects in the dataset
subject_expertise = [(subject, expertise_scores[subject]) for subject in all_subjects 
                     if subject in expertise_scores]

# Sort by expertise score
subject_expertise.sort(key=lambda x: x[1])

# Get top 5 and bottom 5 subjects 
low_expertise_subjects = [subj for subj, _ in subject_expertise[:5]]
high_expertise_subjects = [subj for subj, _ in subject_expertise[-5:][::-1]] # reversed high expertise to show highest first


# Function to average RDMs
def average_rdms(subject_list, roi, rdm_dict):
    """Average RDMs across subjects for a given ROI"""
    rdm_arrays = []
    for subject in subject_list:
        if subject in rdm_dict and roi in rdm_dict[subject]:
            rdm = rdm_dict[subject][roi]
            rdm_arrays.append(rdm.dissimilarities)
    
    if len(rdm_arrays) == 0:
        return None
    
    # Average across subjects
    avg_dissimilarity = np.mean(rdm_arrays, axis=0)
    return avg_dissimilarity

def plot_feature_sensitivity_heatmap(results, version, ROIs, img_variables, outpath=None):
    """
    Create a heatmap of delta_rho (high - low) for each ROI and attribute.
    
    results: dict containing correlation results
    version: which version key to use ("top_5" or "low_high")
    ROIs: list of ROI names
    img_variables: list of (column_name, pretty_name) tuples
    outpath: optional file path to save figure
    """
    # build a DataFrame: rows=ROIs, columns=attributes
    data = {}
    for im_var, im_var_name in img_variables:
        col_vals = []
        for ROI in ROIs:
            delta_key = f"delta_rho"
            delta_rho = results[im_var][ROI].get(delta_key, 0.0)  # default 0 if missing
            col_vals.append(delta_rho)
        data[im_var_name] = col_vals

    heatmap_df = pd.DataFrame(data, index=ROIs)

    # plot
    plt.figure(figsize=(len(img_variables)*1.2, len(ROIs)*0.8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="bwr", center=0, cbar_kws={'label': 'Δrho (high - low)'})
    plt.title(f"Feature Sensitivity Map Across ROIs ({version})")
    plt.ylabel("ROIs")
    plt.xlabel("Attributes")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if outpath:
        plt.savefig(outpath, dpi=300)
    
    return heatmap_df


results = {}

for im_var, im_var_name in img_variables:
    print(f"\n\n======== variable: {im_var} ========")
    results[im_var] = {}
    im_var_column = im_var.replace("_", " ")

    # 0 for same attribute, 1 for different
    attribute = bird_img_annotations[im_var_column].astype(str).values
    equal_attributes = (attribute[:, None] == attribute[None, :])
    attribute_rdm = (~equal_attributes).astype(float)  # shape (n_images, n_images)
    mask = np.triu(np.ones((n_images, n_images), dtype=bool), k=1)
    attribute_rdm_upper = attribute_rdm[mask]

    percent_series_low = {}
    percent_series_high = {}

    for ROI in ROIs:
        results[im_var][ROI] = {}

        flat_avg_low_exp_upper_rdm = average_rdms(low_expertise_subjects, ROI, RDM_dict)
        flat_avg_high_exp_upper_rdm = average_rdms(high_expertise_subjects, ROI, RDM_dict)

        flat_avg_low_exp_upper_rdm = np.asarray(flat_avg_low_exp_upper_rdm).ravel()
        flat_avg_high_exp_upper_rdm = np.asarray(flat_avg_high_exp_upper_rdm).ravel()

        rho_low, p_low = spearmanr(attribute_rdm_upper, flat_avg_low_exp_upper_rdm)
        rho_high, p_high = spearmanr(attribute_rdm_upper, flat_avg_high_exp_upper_rdm)

        results[im_var][ROI]["attr_vs_low_rho"] = rho_low
        results[im_var][ROI]["attr_vs_low_p"] = p_low
        results[im_var][ROI]["attr_vs_high_rho"] = rho_high
        results[im_var][ROI]["attr_vs_high_p"] = p_high
        results[im_var][ROI]["delta_rho"] = rho_high - rho_low

        print(f"{ROI} {im_var_name}: rho_low= {rho_low:.4f}, rho_high= {rho_high:.4f}, delta= {rho_high - rho_low:.4f}")
            

# Example usage after all correlations are computed for a version
heatmap_df = plot_feature_sensitivity_heatmap(results, version="top_5", ROIs=ROIs, img_variables=img_variables,
                                              outpath=results_dir / "img_pair_plots/heatmaps/var_x_roi.png")