import os
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt  
import pandas as pd
from tqdm import tqdm
import json

patch_size = 67

# Bird images
bird_images_dir = Path("/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/bird_images")
# Original bird image embeddings
original_embeds_dir = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/original_embeddings")
# Ocluded bird image embeddings
occluded_embedding_root = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/occluded_embeddings/size_{patch_size}")

output_plot_dir = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/occlusion_plots/size_{patch_size}")
output_plot_dir.mkdir(exist_ok=True)
 
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

original_embeddings = {}
original_embeds_paths = sorted(original_embeds_dir.glob("*.npy"))

for embedding_path in tqdm(original_embeds_paths):
    image_name = embedding_path.stem
    original_embedding = np.load(embedding_path)
    original_embeddings[image_name] = original_embedding


def save_max_dist_embed_and_img(image_name, max_cosine_distance_id):
    bird_image_path = bird_images_dir / f"{image_name}.png"
    bird_image_array = np.array(Image.open(bird_image_path).convert("RGB"))

    H, W, C = bird_image_array.shape
    occluded_image = None
    occluded_img_idx = 0
    
    for y in range(0, H - mask_size + 1, stride):
        for x in range(0, W - mask_size + 1, stride):
            if occluded_img_idx = max_cosine_distance_id
                bird_image_array[y:y+mask_size, x:x+mask_size] = (0, 0, 0)
                occluded_images = bird_image_array
            occluded_img_idx += 1
    
    pil_img = Image.fromarray(occluded_images)
    processed_img = preprocess(pil_img).unsqueeze(0).to(device) 

    with torch.no_grad():
        image_features = model.encode_image(processed_img).to(device) 

    embedding = image_features.cpu().numpy().squeeze()  
    np.save(output_plot_dir / f"{image_name}_max_cos_occ.npy", embedding)


def save_violin_plot(df, image_name, mean_cosine_distance):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Distance Type", y="Distance", data=df, inner="point", scale="width")
    plt.title(f"Occlusion Distance Distribution: {image_name}  ({patch_size}x{patch_size} occlusions)")
    plt.ylabel(f"Distance (mean distance: {mean_cosine_distance:.2f})")
    plt.xlabel("Distance Metric")

    output_plot_path = output_plot_dir / f"{image_name}_violin_plot.png"
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Violin plot saved to {output_plot_path}")


image_metrics_list = []

for image_folder in tqdm(occluded_embedding_root.iterdir(), desc="Processing Occluded Embeddings"):
    if not image_folder.is_dir():
        continue
    
    image_name = image_folder.name
    original_embedding = original_embeddings.get(image_name)
    
    if original_embedding is None:
        print(f"Skipping {image_name} (no original embedding found)")
        continue
    
    distance_data = []
    all_occlusion_files = image_folder.glob("occlusion_*.npy")
    
    # Load occluded embeddings
    for occlusion_file in all_occlusion_files:
        occluded_embedding = np.load(occlusion_file)

        # Compute distances
        cosine_distance = 1 - np.dot(original_embedding, occluded_embedding) / (
            np.linalg.norm(original_embedding) * np.linalg.norm(occluded_embedding)
        )

        distance_data.append({
            "Distance Type": "Cosine", 
            "occlusion_id": int(occlusion_file.stem.split("_")[1]),  
            "Distance": cosine_distance
        })
    
    df = pd.DataFrame(distance_data)

    mean_cosine_distance = float(df["Distance"].mean())
    std_cosine_distance = float(df["Distance"].std())  
    max_cosine_distance = df.loc[df["Distance"].idxmax()]
    max_cosine_distance_id = float(max_cosine_distance["occlusion_id"])

    image_metrics_list.append({
        "image_name": image_name, 
        "mean_cosine_distance": mean_cosine_distance, 
        "std_cosine_distance": std_cosine_distance,
        "max_cosine_distance_id": max_cosine_distance_id
    })

    # Save max distance embedding and image
    save_max_dist_embed_and_img(image_name, max_cosine_distance_id)

    # Save violin plot
    save_violin_plot(df, image_name, mean_cosine_distance)

# Save metrics list to json
with open(output_plot_dir / f"occ_{patch_size}x_{patch_size}_image_embeddings_stats.json", "w") as f:
    json.dump(image_metrics_list, f, indent=4)
