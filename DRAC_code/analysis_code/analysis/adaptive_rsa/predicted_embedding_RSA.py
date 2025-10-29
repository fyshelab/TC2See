import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import json
import traceback
from sklearn.linear_model import HuberRegressor
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
RDM_dir = dataset_root / "processed/predicted_embeddings_RDMs/"

all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

subject_strs_dict = {
    'low':['06', '15', '20', '21', '22', '23', '28', '29', '32', '33', '39'],
    'mid': ['05', '07', '10', '14', '17', '18', '19', '25', '27', '30', '31', '34', '36', '37', '40'],
    'high': ['08', '09', '11', '12', '16', '24', '26', '35', '38']
}

embed_types = {
    # "experiment-8.3": { 
    #     "ViT-B=16-features_large.hdf5": ["embedding"],
    #     "ViT-B=32-features_large.hdf5": ["embedding"],
    # },
    "experiment-8.4": { 
        "ViT-B=16-features_large.hdf5": ['A1', 'FFC', 'LO1', 'LO2', 'PIT', 'Pir', 'V1', 'V2', 'V3', 'V4', 'V7', 'V8', 'VVC'],
        # "ViT-B=32-features_large.hdf5": ['A1', 'FFC', 'LO1', 'LO2', 'PIT', 'Pir', 'V1', 'V2', 'V3', 'V4', 'V7', 'V8', 'VVC'],
    }
}

true_CLIP_RDM = {}
avg_RDMs = {}
categories = [
    # ("experiment-8.3", "ViT-B=16-features_large", "embedding"),
    # ("experiment-8.3", "ViT-B=32-features_large", "embedding"),
    ("experiment-8.4", "ViT-B=16-features_large", "A1"),
    ("experiment-8.4", "ViT-B=16-features_large", "FFC"),
    ("experiment-8.4", "ViT-B=16-features_large", "LO1"),
    ("experiment-8.4", "ViT-B=16-features_large", "LO2"),
    ("experiment-8.4", "ViT-B=16-features_large", "PIT"),
    ("experiment-8.4", "ViT-B=16-features_large", "Pir"),
    ("experiment-8.4", "ViT-B=16-features_large", "V1"),
    ("experiment-8.4", "ViT-B=16-features_large", "V2"),
    ("experiment-8.4", "ViT-B=16-features_large", "V3"),
    ("experiment-8.4", "ViT-B=16-features_large", "V4"),
    ("experiment-8.4", "ViT-B=16-features_large", "V7"),
    ("experiment-8.4", "ViT-B=16-features_large", "V8"),
    ("experiment-8.4", "ViT-B=16-features_large", "VVC"),
]

groups = subject_strs_dict.keys()
for group_name in groups:
    for experiment, model, _ in categories:
        if group_name not in avg_RDMs:
            avg_RDMs[group_name] = {}
        if experiment not in avg_RDMs[group_name]:
            avg_RDMs[group_name][experiment] = {}
        if model not in avg_RDMs[group_name][experiment]:
            avg_RDMs[group_name][experiment][model] = {}


model_versions = ["ViT-B=16"]

# Load RDMs for each true clip embedding 
for model_name in model_versions:
    RDM_path = dataset_root / f"processed/glm_RDMs/CLIP_RDMs"
    rdm_file_path = RDM_path / f'rdms_{model_name}_bird.hdf5'
        
    rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
    true_CLIP_RDM[f"{model_name}-features_large"] = rdm

output_rsa_dict = {}
per_sub_rsa_dict = {}

for group_name, group_subs in subject_strs_dict.items():
    output_rsa_dict[group_name] = {}

    for experiment, embedding_type in embed_types.items():
        output_rsa_dict[group_name][experiment] = {}
        if experiment not in per_sub_rsa_dict:
            per_sub_rsa_dict[experiment] = {}

        for hdf5_path, embedding_names in embedding_type.items():
            model_version = hdf5_path.split(".")[0]
            output_rsa_dict[group_name][experiment][model_version] = {}
            if model_version not in per_sub_rsa_dict[experiment]:
                per_sub_rsa_dict[experiment][model_version] = {}

            for embedding_name in embedding_names:
                rdm_list = []
                if embedding_name not in per_sub_rsa_dict[experiment][model_version]:
                    per_sub_rsa_dict[experiment][model_version][embedding_name] = {}

                for sub in group_subs:
                    pred_embed_rdm_dir = RDM_dir / f"individual/sub_{sub}/{experiment}/{model_version}"
                    pred_embed_rdm_file = pred_embed_rdm_dir / f"{embedding_name}.hdf5"
                    rdm = rsatoolbox.rdm.rdms.load_rdm(pred_embed_rdm_file, file_type='hdf5')
                    pred_v_true_CLIP_rsa = rsatoolbox.rdm.compare(rdm, true_CLIP_RDM[model_version], method='corr')
                    per_sub_rsa_dict[experiment][model_version][embedding_name][sub] = pred_v_true_CLIP_rsa[0].item()
                    rdm_list.append(rdm)

                model_num = model_version.split("-")[1]
                RDM_output_file = RDM_dir / f"groups/{group_name}/avg_rdm_{model_num}_{embedding_name}.hdf5"
                
                rdm_list = rsatoolbox.rdm.concat(rdm_list)
                averaged_rdm = rdm_list.mean(weights=None)
                averaged_rdm.save(RDM_output_file, file_type='hdf5', overwrite=True)
                avg_RDMs[group_name][experiment][model_version][embedding_name] = averaged_rdm

                RSA_corr = rsatoolbox.rdm.compare(averaged_rdm, true_CLIP_RDM[model_version], method='corr')
                output_rsa_dict[group_name][experiment][model_version][embedding_name] = RSA_corr[0].item()


out_file = RDM_dir / "grouped_avg_pred_v_CLIP_rsa_results.json"
with open(out_file, "w") as f:
    json.dump(output_rsa_dict, f, indent=4)

out_file = RDM_dir / "per_sub_pred_v_CLIP_rsa_results.json"
with open(out_file, "w") as f:
    json.dump(per_sub_rsa_dict, f, indent=4)


######################################################################
### Compare Group Averaged Predicted Embedding RDMs Accross Groups ###
######################################################################

between_group_rsa = {}

group_pairs = [
    ("low", "mid"),
    ("low", "high"),
    ("mid", "high"),
]

for experiment in embed_types.keys():
    if experiment not in between_group_rsa:
        between_group_rsa[experiment] = {}
    for model_version in embed_types[experiment].keys():
        model_name = model_version.split(".")[0]
        if model_name not in between_group_rsa[experiment]:
            between_group_rsa[experiment][model_name] = {}
        for embedding_name in embed_types[experiment][model_version]:
            if embedding_name not in between_group_rsa[experiment][model_name]:
                between_group_rsa[experiment][model_name][embedding_name] = {}

            for g1, g2 in group_pairs:
                rdm1 = avg_RDMs[g1][experiment][model_name][embedding_name]
                rdm2 = avg_RDMs[g2][experiment][model_name][embedding_name]

                rsa_val = rsatoolbox.rdm.compare(rdm1, rdm2, method="corr")
                between_group_rsa[experiment][model_name][embedding_name][f"{g1}_vs_{g2}"] = rsa_val[0].item()

# Save to json
out_file = RDM_dir / "between_group_avg_rsa_results.json"
with open(out_file, "w") as f:
    json.dump(between_group_rsa, f, indent=4)