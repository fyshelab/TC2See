import rsatoolbox
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set paths
dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
RDM_dir = dataset_root / "processed/predicted_embeddings_RDMs/"
out_dir = RDM_dir / "group_RDM_plots"
out_dir.mkdir(parents=True, exist_ok=True)

# Define experiments, models, and embeddings to visualize
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



def load_avg_rdm(group, model_num, embedding_name):
    """Load averaged RDM for a given group/model/embedding."""
    file_path = RDM_dir / f"groups/{group}/avg_rdm_{model_num}_{embedding_name}.hdf5"
    return rsatoolbox.rdm.rdms.load_rdm(file_path, file_type="hdf5")

def get_global_scales(categories):
    """Compute global min/max for each embedding type across all models/experiments."""
    scales = {}
    diff_scales = {}

    for experiment, model, embedding in categories:
        model_num = model.split("-")[1]

        # Load averaged RDMs
        low_rdm = load_avg_rdm("low", model_num, embedding)
        high_rdm = load_avg_rdm("high", model_num, embedding)

        low_mat = low_rdm.get_matrices()[0]
        high_mat = high_rdm.get_matrices()[0]
        diff_mat = high_mat - low_mat

        # Update scale for low/high
        cur_min = min(low_mat.min(), high_mat.min())
        cur_max = max(low_mat.max(), high_mat.max())
        if embedding not in scales:
            scales[embedding] = [cur_min, cur_max]
        else:
            scales[embedding][0] = min(scales[embedding][0], cur_min)
            scales[embedding][1] = max(scales[embedding][1], cur_max)

        # Update scale for differences (symmetric around 0)
        cur_abs = max(abs(diff_mat.min()), abs(diff_mat.max()))
        if embedding not in diff_scales:
            diff_scales[embedding] = cur_abs
        else:
            diff_scales[embedding] = max(diff_scales[embedding], cur_abs)

    return scales, diff_scales


def plot_low_high_diff(low_rdm, high_rdm, title, save_path, rdm_min, rdm_max, diff_abs):
    """Plot heatmaps of low, high, and difference (high - low) with consistent embedding-wise scales."""
    low_mat = low_rdm.get_matrices()[0]
    high_mat = high_rdm.get_matrices()[0]
    diff_mat = high_mat - low_mat

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Low expertise
    sns.heatmap(low_mat, cmap="viridis", cbar=True, square=True,
                vmin=rdm_min, vmax=rdm_max, ax=axes[0])
    axes[0].set_title("Low Expertise")

    # High expertise
    sns.heatmap(high_mat, cmap="viridis", cbar=True, square=True,
                vmin=rdm_min, vmax=rdm_max, ax=axes[1])
    axes[1].set_title("High Expertise")

    # Difference
    sns.heatmap(diff_mat, cmap="coolwarm", center=0, cbar=True, square=True,
                vmin=-diff_abs, vmax=diff_abs, ax=axes[2])
    axes[2].set_title("High - Low")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# --- Main execution ---
# First pass: compute global scales per embedding type
scales, diff_scales = get_global_scales(categories)

# Second pass: make plots with consistent scales
for experiment, model, embedding in categories:
    model_num = model.split("-")[1]

    low_rdm = load_avg_rdm("low", model_num, embedding)
    high_rdm = load_avg_rdm("high", model_num, embedding)

    plot_title = f"{model} / {embedding} ({experiment})"
    save_file = out_dir / f"heatmap_{experiment}_{model}_{embedding}.png"

    rdm_min, rdm_max = scales[embedding]
    diff_abs = diff_scales[embedding]

    plot_low_high_diff(low_rdm, high_rdm, plot_title, save_file, rdm_min, rdm_max, diff_abs)