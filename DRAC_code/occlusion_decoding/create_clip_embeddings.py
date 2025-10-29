import re
import numpy as np
import torch
import clip
from PIL import Image
import glob
from pathlib import Path
import re
import h5py

device = "cpu"
# mod_im_combos = [("ViT-B=32", "bird"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "fg_mask"), ("ViT-B=32", "max_dist")]
mod_im_combos = [("ViT-B=32", "fg_mask")]

for model_name, image_type in mod_im_combos:
    clip_model = "ViT-B/32" if model_name == 'ViT-B=32' else "ViT-B/16"

    model, preprocess = clip.load(clip_model, device=device) 

    embeddings = []
    for filename in sorted(glob.glob(f"../data/{image_type}_images/*.png")):

        if "hash" in filename:
            continue

        name = Path(filename).stem
        print(f"Creating {clip_model} embedding for {name}...")

        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

        image_features = image_features.cpu().numpy()
        embeddings.append(image_features[0]) 

    embeddings_matrix = np.stack(embeddings, axis=0)

    hdf5_file_path = f'../data/{model_name}-features_{image_type}.hdf5'

    with h5py.File(hdf5_file_path, 'w') as f:
        f.create_dataset('embedding', data=embeddings_matrix)

    print(f"Embeddings saved to {hdf5_file_path}")