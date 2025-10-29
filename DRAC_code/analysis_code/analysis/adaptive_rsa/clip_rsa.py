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
import matplotlib as mpl


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
rdm_dist = "correlation"

sigma = 5  # Gaussian decay parameter 
sigma_dir = results_dir / f'gaussian_rsa/sigma_{sigma}'
sigma_dir.mkdir(parents=True, exist_ok=True)

ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]

all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

subject_strs_dict = {
    'all_subjects': all_subjects,
    'low':['06', '15', '20', '21', '22', '23', '28', '29', '32', '33', '39'],
    'mid': ['05', '07', '10', '14', '17', '18', '19', '25', '27', '30', '31', '34', '36', '37', '40'],
    'high': ['08', '09', '11', '12', '16', '24', '26', '35', '38']
}

# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)

# Initialize dictionary to store RDMs
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

all_clip_combos = [("ViT-B=32", "bird"), ("ViT-B=32", "fg_mask"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "max_dist")]

CLIP_RDM_dict = {}

# Load RDMs for each clip embedding 
for model_name, image_type in all_clip_combos:
    RDM_path = dataset_root / f"processed/glm_RDMs/CLIP_RDMs"
    rdm_file_path = RDM_path / f'rdms_{model_name}_{image_type}.hdf5'
        
    rdm = rsatoolbox.rdm.rdms.load_rdm(rdm_file_path, file_type='hdf5')
    CLIP_RDM_dict[f"{model_name}_{image_type}"] = rdm


# Function to calculate Gaussian weight based on expertise difference
def calculate_gaussian_weight(subject1, subject2, sigma):
    if subject1 not in expertise_scores or subject2 not in expertise_scores:
        return 0
        
    score1 = expertise_scores[subject1]
    score2 = expertise_scores[subject2]
    expertise_diff = abs(score1 - score2)
    
    weight = np.exp(-(expertise_diff**2) / (2 * sigma**2))
    return weight



##############################################################################
# For each ROI, calculate weighted correlations based on expertise similarity
##############################################################################

expertise_vs_rsa_correlation_similarity = {}
expertise_vs_rsa_grouped = {}
clip_rsa_results = {}
clip_rsa_results_grouped = {}

for ROI in ROIs:
    print(f"Analyzing ROI: {ROI} with Gaussian sigma {sigma}")
    try:
        clip_rsa_results[ROI] = {}
        clip_rsa_results_grouped[ROI] = {}
        expertise_vs_rsa_correlation_similarity[ROI] = {}
        expertise_vs_rsa_grouped[ROI] = {}
    
        for subject1 in all_subjects:
            if ROI not in RDM_dict[subject1]:
                continue

            clip_rsa_results[ROI][subject1] = {}
            clip_rsa_results_grouped[ROI][subject1] = {}
            
            for model_name, image_type in all_clip_combos:
                clip_rsa_results_grouped[ROI][subject1][f"{model_name}_{image_type}"] = {}
                expertise_vs_rsa_grouped[ROI][f"{model_name}_{image_type}"] = {}

                corr_value = rsatoolbox.rdm.compare(RDM_dict[subject1][ROI], CLIP_RDM_dict[f"{model_name}_{image_type}"], method='corr')
                clip_rsa_results[ROI][subject1][f"{model_name}_{image_type}"] = corr_value.item()

                if subject1 in subject_strs_dict["low"]:
                    clip_rsa_results_grouped[ROI][subject1][f"{model_name}_{image_type}"]["low"] = corr_value.item()
                elif subject1 in subject_strs_dict["high"]:
                    clip_rsa_results_grouped[ROI][subject1][f"{model_name}_{image_type}"]["high"] = corr_value.item()
            
        
        for model_name, image_type in all_clip_combos:
            # Get expertise scores and correlations as arrays
            scores = [expertise_scores[subject] for subject in all_subjects]
            similarity_correlations = [clip_rsa_results[ROI][subject][f"{model_name}_{image_type}"] for subject in all_subjects]
            
            # Calculate pearson correlations
            r, p_value_pearson = pearsonr(scores, similarity_correlations)
            rho, p_value = spearmanr(scores, similarity_correlations)   
        
            expertise_vs_rsa_correlation_similarity[ROI][f"{model_name}_{image_type}"] = {
                'pearson_r': r,
                'pearson_p': p_value_pearson,
                'scores': scores,
                'similarity_correlations': similarity_correlations
                # 'spearman_r': rho,
                # 'spearman_p': p_value
            }

            for group in ["low", "high"]:
                scores = [expertise_scores[subject] for subject in subject_strs_dict[group]]
                similarity_correlations = [clip_rsa_results_grouped[ROI][subject][f"{model_name}_{image_type}"][group]
                                            for subject in subject_strs_dict[group]]
                
                # Calculate pearson correlations
                r, p_value_pearson = pearsonr(scores, similarity_correlations)
                rho, p_value = spearmanr(scores, similarity_correlations)

                expertise_vs_rsa_grouped[ROI][f"{model_name}_{image_type}"][group] = {
                    'pearson_r': r,
                    'pearson_p': p_value_pearson,
                    'scores': scores,
                    'similarity_correlations': similarity_correlations
                    # 'spearman_r': rho,
                    # 'spearman_p': p_value
                }
        
    except Exception as e:
        print(f"\n\nError analyzing ROI {ROI}, subject1: {subject1}, combo: {model_name}_{image_type} \n")
        traceback.print_exc()
        print("\n\n")
        continue


def run_comparison_tests(
    expertise_vs_rsa_correlation_similarity,
    ROIs,
    all_clip_combos,
    results_dir,
    correlation_method='pearson'  # or 'spearman'
):
    output_dir = results_dir / "clip_rsa/scatter_plots_by_embedding"
    output_dir.mkdir(parents=True, exist_ok=True)

    nrows, ncols = 3, 5

    for model_name, image_type in all_clip_combos:
        key = f"{model_name}_{image_type}"
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 12))
        axs = axs.flatten()

        for idx, ROI in enumerate(ROIs):
            ax = axs[idx]
            correlations = expertise_vs_rsa_correlation_similarity[ROI][key]["similarity_correlations"]
            scores = expertise_vs_rsa_correlation_similarity[ROI][key]["scores"]

            # Compute correlation and p-value
            if correlation_method == 'pearson':
                r_val, p_val = pearsonr(scores, correlations)
            elif correlation_method == 'spearman':
                r_val, p_val = spearmanr(scores, correlations)
            else:
                raise ValueError("correlation_method must be 'pearson' or 'spearman'")

            # Scatter plot
            ax.scatter(scores, correlations, color='steelblue', alpha=0.7, edgecolor='k')

            # Add regression line for Pearson
            if correlation_method == 'pearson':
                slope, intercept = np.polyfit(scores, correlations, 1)
                x_vals = np.array(ax.get_xlim())
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color='red', linestyle='--', linewidth=1.5)

            stat_label = "r" if correlation_method == 'pearson' else "ρ"
            ax.set_title(f'{ROI}\n{stat_label}={r_val:.2f}, p={p_val:.3f}', fontsize=10)
            ax.set_xlabel('Expertise Score')
            ax.set_ylabel('RSA Correlation')
            ax.set_ylim(-0.022, 0.09)

        # Turn off any unused subplots (e.g. last two in the 3×5 grid)
        for j in range(len(ROIs), len(axs)):
            axs[j].axis('off')

        fig.suptitle(
            f'CLIP embedding: {key} – RSA Correlation vs Expertise Score ({correlation_method})',
            fontsize=18
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"{key}_scatter_grid_{correlation_method}.png")
        plt.close()


run_comparison_tests(
    expertise_vs_rsa_correlation_similarity=expertise_vs_rsa_correlation_similarity,
    ROIs=ROIs,
    all_clip_combos=all_clip_combos,
    results_dir=results_dir,
    correlation_method="pearson"
)


def plot_rsa_differences_by_expertise(
    clip_rsa_results,
    ROIs,
    all_subjects,
    expertise_scores,
    results_dir,
    base_variant=("ViT-B=32", "bird"),
    compare_variants=[("ViT-B=32", "fg_mask"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "max_dist")],
    correlation_method="pearson"  # or "spearman"
):
    output_dir = results_dir / "clip_rsa/rsa_differences_by_expertise_scatter"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_key = f"{base_variant[0]}_{base_variant[1]}"

    nrows, ncols = 3, 5

    for variant in compare_variants:
        compare_key = f"{variant[0]}_{variant[1]}"
        fig, axs = plt.subplots(nrows, ncols, figsize=(20, 12))
        axs = axs.flatten()

        for idx, ROI in enumerate(ROIs):
            ax = axs[idx]

            diffs = []
            scores = []

            for subject in all_subjects:
                if ROI not in clip_rsa_results or subject not in clip_rsa_results[ROI]:
                    continue
                if base_key not in clip_rsa_results[ROI][subject] or compare_key not in clip_rsa_results[ROI][subject]:
                    continue

                diff = clip_rsa_results[ROI][subject][base_key] - clip_rsa_results[ROI][subject][compare_key]
                score = expertise_scores.get(subject, 0)

                diffs.append(diff)
                scores.append(score)

            if not diffs:
                ax.set_axis_off()
                continue

            # Compute correlation
            if correlation_method == "pearson":
                r_val, p_val = pearsonr(scores, diffs)
            elif correlation_method == "spearman":
                r_val, p_val = spearmanr(scores, diffs)
            else:
                raise ValueError("correlation_method must be 'pearson' or 'spearman'")

            # Scatter plot (single color)
            ax.scatter(scores, diffs, color="steelblue", edgecolor="k", alpha=0.7)

            # Add regression line for Pearson
            if correlation_method == "pearson":
                slope, intercept = np.polyfit(scores, diffs, 1)
                x_vals = np.array(ax.get_xlim())
                y_vals = intercept + slope * x_vals
                ax.plot(x_vals, y_vals, color="red", linestyle="--", linewidth=1.5)

            stat_label = "r" if correlation_method == "pearson" else "ρ"
            ax.set_title(f"{ROI}\n{stat_label}={r_val:.2f}, p={p_val:.3f}", fontsize=10)

            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Expertise Score")
            ax.set_ylabel("Δ RSA")
            ax.set_ylim(min(diffs) - 0.01, max(diffs) + 0.01)

        for j in range(len(ROIs), len(axs)):
            axs[j].axis("off")

        fig.suptitle(
            f"RSA difference (scatter): {base_key} − {compare_key} vs expertise score ({correlation_method})",
            fontsize=18
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"difference_{base_key}_minus_{compare_key}_scatter_{correlation_method}.png")
        plt.close()


plot_rsa_differences_by_expertise(
    clip_rsa_results,
    ROIs,
    all_subjects,
    expertise_scores,
    results_dir,
    base_variant=("ViT-B=32", "bird"),
    compare_variants=[("ViT-B=32", "fg_mask"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "max_dist")],
    correlation_method="pearson"
)



def plot_grouped_rsa_differences(
    clip_rsa_results,
    ROIs,
    subject_strs_dict,
    results_dir,
    base_variant=("ViT-B=32", "bird"),
    compare_variants=[("ViT-B=32", "fg_mask"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "max_dist")],
    colormap="viridis"
):
    import matplotlib as mpl

    output_dir = results_dir / "clip_rsa/grouped_rsa_differences"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_key = f"{base_variant[0]}_{base_variant[1]}"

    nrows, ncols = 3, 5  # 13 ROIs
    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 12))
    axs = axs.flatten()

    # Expertise groups in order
    groups = ["low", "mid", "high"]

    # Colormap for expertise groups
    cmap = plt.cm.get_cmap(colormap)
    norm = mpl.colors.Normalize(vmin=0, vmax=2)  # 0->low, 1->mid, 2->high
    group_colors = [cmap(norm(i)) for i in range(len(groups))]

    for idx, ROI in enumerate(ROIs):
        ax = axs[idx]

        # Compute average ΔRSA for each group and each comparison variant
        group_means = []
        for compare_variant in compare_variants:
            compare_key = f"{compare_variant[0]}_{compare_variant[1]}"
            means_per_group = []
            for group in groups:
                subjects = subject_strs_dict[group]
                diffs = []
                for subject in subjects:
                    if ROI not in clip_rsa_results or subject not in clip_rsa_results[ROI]:
                        continue
                    if base_key not in clip_rsa_results[ROI][subject] or compare_key not in clip_rsa_results[ROI][subject]:
                        continue
                    diff = clip_rsa_results[ROI][subject][base_key] - clip_rsa_results[ROI][subject][compare_key]
                    diffs.append(diff)
                mean_diff = np.mean(diffs) if diffs else 0
                means_per_group.append(mean_diff)
            group_means.append(means_per_group)

        # Plot grouped bars
        group_means = np.array(group_means)  # shape = (num_comparisons, num_groups)
        x = np.arange(len(compare_variants))
        width = 0.2
        for i, color in enumerate(group_colors):
            ax.bar(x + i*width - width, group_means[:, i], width=width, color=color, edgecolor='k', label=groups[i] if idx==0 else "")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{base_variant[1]}-{c[1]}" for c in compare_variants])
        ax.set_ylabel("Δ RSA")
        ax.set_title(ROI, fontsize=10)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Add legend and colorbar
    handles = [mpl.patches.Patch(color=group_colors[i], label=groups[i]) for i in range(len(groups))]
    fig.legend(handles=handles, title="Expertise group", loc="upper right")

    # Turn off unused axes
    for j in range(len(ROIs), len(axs)):
        axs[j].axis("off")

    fig.suptitle("Grouped ΔRSA by ROI and Expertise", fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / "grouped_expertise_rsa_differences.png")
    plt.close()


plot_grouped_rsa_differences(
    clip_rsa_results,
    ROIs,
    subject_strs_dict,
    results_dir,
    base_variant=("ViT-B=32", "bird"),
    compare_variants=[("ViT-B=32", "fg_mask"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "max_dist")],
    colormap="viridis"
)




def plot_r_and_p(expertise_vs_rsa_correlation_similarity, ROI):
    data = expertise_vs_rsa_correlation_similarity[ROI]

    labels = list(data.keys())
    r_values = [data[k]['pearson_r'] for k in labels]
    p_values = [data[k]['pearson_p'] for k in labels]

    x = range(len(labels))

    # Ensure directories exist
    r_plot_dir = results_dir / "clip_rsa/r_plots"
    p_plot_dir = results_dir / "clip_rsa/p_plots"
    r_plot_dir.mkdir(parents=True, exist_ok=True)
    p_plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot r values
    plt.figure(figsize=(10, 4))
    plt.bar(x, r_values, color='skyblue')
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Pearson r')
    plt.title(f'Pearson r values for ROI: {ROI}')
    plt.tight_layout()
    r_plot_path = r_plot_dir / f"{ROI}_r_plot.png"
    plt.savefig(r_plot_path)
    plt.close()

    # Plot p values
    plt.figure(figsize=(10, 4))
    plt.bar(x, p_values, color='lightcoral')
    plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Pearson p')
    plt.title(f'Pearson p values for ROI: {ROI}')
    plt.tight_layout()
    p_plot_path = p_plot_dir / f"{ROI}_p_plot.png"
    plt.savefig(p_plot_path)
    plt.close()

def plot_grouped_r_and_p(expertise_vs_rsa_grouped, ROI):
    data = expertise_vs_rsa_grouped[ROI]

    model_keys = list(data.keys())  # model_type keys like 'ViT-B=16_bird'
    groups = ["low", "high"]

    # Collect r and p values per group
    r_values_low = [data[model]["low"]["pearson_r"] for model in model_keys]
    r_values_high = [data[model]["high"]["pearson_r"] for model in model_keys]
    p_values_low = [data[model]["low"]["pearson_p"] for model in model_keys]
    p_values_high = [data[model]["high"]["pearson_p"] for model in model_keys]

    x = np.arange(len(model_keys))
    bar_width = 0.35

    # Create output directories
    r_plot_dir = results_dir / "clip_rsa/r_plots_grouped"
    p_plot_dir = results_dir / "clip_rsa/p_plots_grouped"
    r_plot_dir.mkdir(parents=True, exist_ok=True)
    p_plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot r values
    plt.figure(figsize=(12, 5))
    plt.bar(x - bar_width/2, r_values_low, bar_width, label='low', color='skyblue')
    plt.bar(x + bar_width/2, r_values_high, bar_width, label='high', color='navy')
    plt.xticks(x, model_keys, rotation=45, ha='right')
    plt.ylabel('Pearson r')
    plt.title(f'Pearson r values for ROI: {ROI}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r_plot_dir / f"{ROI}_r_plot.png")
    plt.close()

    # Plot p values
    plt.figure(figsize=(12, 5))
    plt.bar(x - bar_width/2, p_values_low, bar_width, label='low', color='lightcoral')
    plt.bar(x + bar_width/2, p_values_high, bar_width, label='high', color='darkred')
    plt.axhline(y=0.05, color='red', linestyle='--', linewidth=1)
    plt.xticks(x, model_keys, rotation=45, ha='right')
    plt.ylabel('Pearson p')
    plt.title(f'Pearson p values for ROI: {ROI}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(p_plot_dir / f"{ROI}_p_plot.png")
    plt.close()

def run_group_comparison_tests(expertise_vs_rsa_grouped, ROIs, all_clip_combos, subject_strs_dict, results_dir):
    output_dir = results_dir / "clip_rsa/mannwhitney_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_summary = []

    for ROI in ROIs:
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()
        subplot_idx = 0

        for model_name, image_type in all_clip_combos:
            key = f"{model_name}_{image_type}"

            try:
                low_corrs = expertise_vs_rsa_grouped[ROI][key]['low']['similarity_correlations']
                high_corrs = expertise_vs_rsa_grouped[ROI][key]['high']['similarity_correlations']
            except KeyError:
                continue  # Skip missing data

            # Mann–Whitney U test
            stat, p_val = mannwhitneyu(low_corrs, high_corrs, alternative='two-sided')

            # Save summary
            stats_summary.append({
                "ROI": ROI,
                "model": key,
                "U_statistic": stat,
                "p_value": p_val,
                "low_mean": sum(low_corrs) / len(low_corrs),
                "high_mean": sum(high_corrs) / len(high_corrs),
            })

            # Plot in current subplot
            df = pd.DataFrame({
                'correlation': low_corrs + high_corrs,
                'group': ['low'] * len(low_corrs) + ['high'] * len(high_corrs)
            })

            ax = axs[subplot_idx]
            sns.boxplot(data=df, x='group', y='correlation', palette='pastel', ax=ax)
            sns.stripplot(data=df, x='group', y='correlation', color='black', jitter=0, size=5, ax=ax)

            ax.set_title(f'{key}\nU p={p_val:.3f}')
            ax.set_xlabel('')
            ax.set_ylabel(f'Correlation ( Brain RDM vs CLIP RDM )')

            subplot_idx += 1

        # If there are unused subplots, hide them
        for j in range(subplot_idx, len(axs)):
            axs[j].axis('off')

        fig.suptitle(f'ROI: {ROI} – Mann–Whitney U tests', fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_dir / f"{ROI}_mannwhitney_grid.png")
        plt.close()

    # Save summary CSV
    summary_df = pd.DataFrame(stats_summary)
    summary_df.to_csv(output_dir / "mannwhitney_test_results.csv", index=False)

# for ROI in expertise_vs_rsa_correlation_similarity.keys():
#     plot_r_and_p(expertise_vs_rsa_correlation_similarity, ROI)
#     plot_grouped_r_and_p(expertise_vs_rsa_grouped, ROI)