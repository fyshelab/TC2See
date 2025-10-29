import os
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
import h5py
from tqdm import tqdm
import json


mask_size = 67
stride = 33

input_folder = Path("/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/bird_images")
original_embeds_dir = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/original_embeddings")
occluded_embeds_dir = Path(f"/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/occluded_embeddings/size_{mask_size}")
max_distance_dir = occluded_embeds_dir / "max_cosine_distance"

image_paths = sorted(input_folder.glob("*.png"))

device = "cuda" if torch.cuda.is_available() else "cpu"  
model, preprocess = clip.load("ViT-B/32", device=device)



def apply_occlusion(image):
    H, W, C = image.shape
    occluded_images = []
    
    for y in range(0, H - mask_size + 1, stride):
        for x in range(0, W - mask_size + 1, stride):
            occluded_img = image.copy()
            occluded_img[y:y+mask_size, x:x+mask_size] = (0, 0, 0)
            occluded_images.append(occluded_img)
    
    return occluded_images



def save_max_distance_embed_and_img(original_img_embed, occluded_embeddings, occluded_images, image_name):
    max_cosine_distance = -1
    max_cosine_distance_idx = None

    for i, occluded_embed in enumerate(occluded_embeddings):
        cosine_distance = 1 - np.dot(original_img_embed, occluded_embed) / (
            np.linalg.norm(original_img_embed) * np.linalg.norm(occluded_embed)
        )

        if cosine_distance > max_cosine_distance:
            max_cosine_distance = cosine_distance
            max_cosine_distance_idx = i

    max_dist_embed_path = max_distance_dir / f"{image_name}"
    max_dist_embed_path.mkdir(exist_ok=True)

    # Save max distance image embedding
    max_dist_embedding = occluded_embeddings[max_cosine_distance_idx]
    np.save(max_dist_embed_path / f"max_distance_embedding.npy", max_dist_embedding)

    # Save max distance occluded image
    occluded_img = Image.fromarray( occluded_images[max_cosine_distance_idx] )
    occluded_img.save(max_dist_embed_path / f"max_distance_image.png")


    max_dist_embed_info = {
        "occlussion_idx": max_cosine_distance_idx,
        "distance_to_original_embedding": max_cosine_distance
    }
    with open(max_dist_embed_path / f"info_max_distance_embedding.json", "w") as f:
        json.dump(max_dist_embed_info, f, indent=4)

    



for im, image_path in enumerate(image_paths):
    image_name = image_path.stem
    img = np.array(Image.open(image_path).convert("RGB"))
    occluded_images = apply_occlusion(img)
    occluded_images.append(img)

    image_output_dir = occluded_embeds_dir / image_name
    image_output_dir.mkdir(parents=True, exist_ok=True)

    occluded_embeddings = []
    original_img_embed = None

    for i, occluded_img in enumerate(occluded_images):
        print(f"Img: {image_name} ({im + 1}/{len(image_paths)}) Occlusion: {i}")

        occl_embed_file_path = image_output_dir / f"occlusion_{i}.npy"

        pil_img = Image.fromarray(occluded_img)
        processed_img = preprocess(pil_img).unsqueeze(0).to(device) 

        with torch.no_grad():
            image_features = model.encode_image(processed_img).to(device) 

        embedding = image_features.cpu().numpy().squeeze()  

        if i != len(occluded_images) - 1: 
            np.save(image_output_dir / f"occlusion_{i}.npy", embedding)
            occluded_embeddings.append(embedding)
        else: # Original image embedding is last item in occluded_images
            np.save(original_embeds_dir / f"{image_name}.npy", embedding)
            original_img_embed = embedding

    print("    Saving max distance embeding and image...")
    save_max_distance_embed_and_img(original_img_embed, occluded_embeddings, occluded_images, image_name)
