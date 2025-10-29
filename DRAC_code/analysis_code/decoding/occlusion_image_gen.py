import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import os

weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1  
model = torch.hub.load('pytorch/vision:v0.13.0', 'deeplabv3_resnet101', weights=weights)
model.eval()
mask_type = "fg"


def segment_bird(image_path, output_path):
    input_image = Image.open(image_path).convert('RGB')
    original_size = input_image.size 

    preprocess = T.Compose([
        T.Resize(520),  
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)  # Get class index for each pixel

    if mask_type == "bg":
        bird_mask = output_predictions == 3 # We need the bird class, class 3
    elif mask_type == "fg":
        bird_mask = output_predictions != 3

    # Resize the mask back to the original image size
    bird_mask_pil = T.Resize(original_size)(Image.fromarray(bird_mask.byte().cpu().numpy()))
    bird_mask_np = np.array(bird_mask_pil)

    # Apply mask to the original image
    input_np = np.array(input_image)
    masked_image = input_np * bird_mask_np[:, :, None] 

    masked_image_pil = Image.fromarray(masked_image.astype('uint8'))
    masked_image_pil.save(output_path)


def process_directory(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        print(f"Processing {filename}...")
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        segment_bird(input_path, output_path)


input_directory = '/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/bird_images/'
output_directory = f'/home/jamesmck/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code/data/{mask_type}_mask_images/'
process_directory(input_directory, output_directory)
