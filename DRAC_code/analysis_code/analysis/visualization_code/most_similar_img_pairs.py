import numpy as np
import pandas as pd
import rsatoolbox
from pathlib import Path
import matplotlib.pyplot as plt
import json
import h5py
from PIL import Image


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
rdm_dist = "correlation"

ROIs = ["V1", "V2", "V3", "V4",  "V7", "V8", "LO1", "LO2", "PIT", "FFC", "VVC", "A1", "Pir"]
all_subjects = ['05', '06', '07', '08', '09', '10', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

# Load expertise scores
with open(dataset_root / "participant_quiz_scores.json", 'r') as f:
    expertise_scores = json.load(f)

with h5py.File(dataset_root / f'stimulus-images.hdf5', 'r') as f:
    stimulus_names = list(f.keys())

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


# Get the images used in the experiment 
roi_path = dataset_root / f"processed/glm_roi_representations/sub_05" / "V1"
sub_roi_data = pd.read_parquet(roi_path / f'reps_for_V1.parquet')
sub_roi_data['stimulus_category'] = sub_roi_data['stimulus_category'].apply(lambda x: 1 if x == "Sparrow" else 2)
sub_roi_data = sub_roi_data.groupby('stimulus_id').mean().reset_index()
stim_ids = list(sub_roi_data['stimulus_id'])
bird_images = [stimulus_names[i] + ".png" for i in stim_ids]
print("Bird images length:", len(bird_images))


# Get expertise scores for subjects in the dataset
subject_expertise = [(subject, expertise_scores[subject]) for subject in all_subjects 
                     if subject in expertise_scores]

# Sort by expertise score
subject_expertise.sort(key=lambda x: x[1])

# Get top 5 and bottom 5 subjects 
low_expertise_subjects = [subj for subj, _ in subject_expertise[:5]]
high_expertise_subjects = [subj for subj, _ in subject_expertise[-5:][::-1]] # reversed high expertise to show highest first

print("Low expertise subjects:", low_expertise_subjects)
print("High expertise subjects:", high_expertise_subjects)

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



def get_ranked_upper_tri(rdm_dissimilarities):
    """
    Extract upper triangular values, rank them, and keep track of indices
    Returns: list of tuples (dissimilarity_value, img1_idx, img2_idx)
    """
    rdm_dissimilarities = np.array(rdm_dissimilarities).flatten()
    # Number of images
    n_images = int(np.sqrt(2 * len(rdm_dissimilarities) + 0.25) + 0.5)
    
    # Convert dissimilarity vector to matrix
    dissim_matrix = np.zeros((n_images, n_images))
    idx = 0
    for i in range(n_images):
        for j in range(i + 1, n_images):
            dissim_matrix[i, j] = float(rdm_dissimilarities[idx])
            dissim_matrix[j, i] = float(rdm_dissimilarities[idx])
            idx += 1
    
    # Extract upper triangular with indices
    upper_tri_values = []
    for i in range(n_images):
        for j in range(i + 1, n_images):
            upper_tri_values.append((dissim_matrix[i, j], i, j))
    
    # Sort by dissimilarity value (ascending - most similar first)
    upper_tri_values.sort(key=lambda x: x[0])
    
    return upper_tri_values



def plot_image_pairs(ranked_list, bird_images, dataset_root, title, save_path):
    """
    Plot top 10 most similar image pairs
    """
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.995)
    
    for idx, (dissim, row, col) in enumerate(ranked_list[:10]):
        col_idx = idx // 5 
        row_idx = idx % 5   
        
        # Calculate subplot positions (each pair takes 2 columns)
        img1_ax = axes[row_idx, col_idx * 2]
        img2_ax = axes[row_idx, col_idx * 2 + 1]
        
        # Load images
        img1_path = dataset_root / f"bird_images/{bird_images[row]}"
        img2_path = dataset_root / f"bird_images/{bird_images[col]}"

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Plot first image
        img1_ax.imshow(img1)
        img1_ax.axis('off')
        
        # Plot second image
        img2_ax.imshow(img2)
        img2_ax.axis('off')
        
        img1_ax.set_title(f"Pair #{idx+1} | Dissim: {dissim:.4f}", 
                        fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    # Separator between columns
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.05, 0.96], 
                               transform=fig.transFigure,
                               color='red', linestyle='--', linewidth=2))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


for ROI in ROIs:
    print(f"\n------ ROI: {ROI} ------")
    
    # Get average of RDMs for top and bottom 5 subjects
    low_expertise_rdm = average_rdms(low_expertise_subjects, ROI, RDM_dict)
    high_expertise_rdm = average_rdms(high_expertise_subjects, ROI, RDM_dict)
    
    # Get ranked dissimilarity values with image indices
    low_ranked = get_ranked_upper_tri(low_expertise_rdm)
    high_ranked = get_ranked_upper_tri(high_expertise_rdm)
    
    # Plot of image pairs for top 10 most similar brain representations for low expertise subjects
    plot_image_pairs(
        low_ranked, 
        bird_images, 
        dataset_root, 
        f"{ROI} - Low Expertise: Top 10 Most Similar Pairs",
        f"{ROI}_low_expertise_pairs.png"
    )
    
    # Plot of image pairs for top 10 most similar brain representations for high expertise subjects
    plot_image_pairs(
        high_ranked, 
        bird_images, 
        dataset_root, 
        f"{ROI} - High Expertise: Top 10 Most Similar Pairs",
        f"{ROI}_high_expertise_pairs.png"
    )