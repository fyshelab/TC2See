import os
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
import h5py
from tqdm import tqdm
import json
import pandas as pd


patch_size = 67
stride = 33

plot_dir = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/occlusion_plots/size_{patch_size}")
embeddings_stats_file = plot_dir / f"occ_{patch_size}x_{patch_size}_image_embeddings_stats.json"
image_folder = Path("/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/bird_images")
image_paths = sorted(image_folder.glob("*.png"))[0:10]

# load the embeddings stats
with open(embeddings_stats_file, "r") as f:
    embeddings_stats = json.load(f)
embedding_stats_DF = pd.DataFrame(embeddings_stats)


def apply_occlusion(image):
    H, W, C = image.shape
    occluded_images = []
    
    for y in range(0, H - mask_size + 1, stride):
        for x in range(0, W - mask_size + 1, stride):
            occluded_img = image.copy()
            occluded_img[y:y+mask_size, x:x+mask_size] = (0, 0, 0)
            occluded_images.append(occluded_img)
    
    return occluded_images


for im in image_paths:
    img = np.array(Image.open(im).convert("RGB"))
    occluded_images = apply_occlusion(img)
    highest_cosine_dist_id = int(embedding_stats_DF.loc[embedding_stats_DF["image_name"] == str(im.stem)]["max_cosine_distance_id"].values[0])

    for i, occluded_img in enumerate(occluded_images):
        if i == highest_cosine_dist_id:
            occluded_img = Image.fromarray(occluded_img)
            occluded_img.save(plot_dir / f"{im.stem}_max_cos_occ.png")