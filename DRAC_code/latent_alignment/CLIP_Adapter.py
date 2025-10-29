import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModelWithProjection
import numpy as np
from typing import Tuple
import random
from tc2see import load_data
from evaluation import evaluate_alignment_quality
import torchvision.transforms as T
from PIL import Image
import nibabel as nib
import glob
from pathlib import Path
import os
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
shared_dataset_root = Path(os.path.expanduser('~/projects/def-afyshe-ab/TC2See'))
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")
print(f"Using device: {device}")

def compute_ncsnr(
        betas: np.ndarray,
        stimulus_ids: np.ndarray,
):
    unique_ids = np.unique(stimulus_ids)
    betas_var = []
    for i in unique_ids:
        stimulus_betas = betas[stimulus_ids == i]
        betas_var.append(stimulus_betas.var(axis=0, ddof=1))
    betas_var_mean = np.nanmean(np.stack(betas_var), axis=0)
    std_noise = np.sqrt(betas_var_mean)
    std_signal = 1. - betas_var_mean
    std_signal[std_signal < 0.] = 0.
    std_signal = np.sqrt(std_signal)
    ncsnr = std_signal / std_noise
    return ncsnr

def compute_nc(ncsnr: np.ndarray, num_averages: int = 1):
    ncsnr_squared = ncsnr ** 2
    nc = 100. * ncsnr_squared / (ncsnr_squared + (1. / num_averages))
    return nc

def get_top_nc_voxel_indices(fmri_data, stimulus_ids, ROI_mask, top_percentage=10.0):
    """
    Compute noise ceiling and return indices of top percentage of voxels.
    
    Args:
        fmri_data: Training fMRI data (samples x voxels) - ALREADY MASKED by ROI
        stimulus_ids: Stimulus IDs corresponding to fMRI data
        ROI_mask: Boolean mask for ROI voxels (only used for mapping back to original space)
        top_percentage: Percentage of top voxels to select (e.g., 10.0 for top 10%)
    
    Returns:
        selected_indices: Indices of selected voxels in the ORIGINAL (unmasked) space
    """
    print(f"fMRI data shape (already ROI-masked): {fmri_data.shape}")
    print(f"ROI mask shape: {ROI_mask.shape}")
    print(f"ROI mask sum (should match fMRI voxel count): {np.sum(ROI_mask)}")
    
    # The fMRI data is already masked, so we work directly with it
    # No need to apply ROI_mask again
    fmri_roi = fmri_data
    
    # Compute noise ceiling on the already-masked data
    ncsnr = compute_ncsnr(fmri_roi, stimulus_ids)
    nc = compute_nc(ncsnr, num_averages=1)
    
    # Handle NaN values by setting them to 0
    nc[np.isnan(nc)] = 0
    
    # Calculate number of voxels to select
    num_voxels_to_select = int(np.ceil((top_percentage / 100.0) * len(nc)))
    num_voxels_to_select = min(num_voxels_to_select, np.count_nonzero(nc))
    
    print(f"Selecting {num_voxels_to_select} out of {len(nc)} ROI voxels")
    
    # Get indices of top voxels within the ROI-masked data
    roi_argsort_ids = np.argsort(-nc)[:num_voxels_to_select]
    
    # Map back to original voxel space
    # roi_indices gives us the positions in the original full brain where ROI voxels are located
    roi_indices = np.where(ROI_mask.astype(bool))[0]
    selected_indices = roi_indices[roi_argsort_ids]
    
    print(f"Selected voxel indices range: {np.min(selected_indices)} - {np.max(selected_indices)}")
    print(f"Noise ceiling range: {np.min(nc[nc > 0]):.3f} - {np.max(nc):.3f}")
    
    return selected_indices, nc

class BrainImageDataset(Dataset):
    def __init__(self, images, fmri_vectors):
        self.images = torch.tensor(images, dtype=torch.float32) if isinstance(images, np.ndarray) else images
        self.fmri_vectors = torch.tensor(fmri_vectors, dtype=torch.float32) if isinstance(fmri_vectors, np.ndarray) else fmri_vectors
        
        assert len(self.images) == len(self.fmri_vectors), "Images and fMRI vectors must have the same length"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.fmri_vectors[idx]

class CLIPVisionEncoderWrapper(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super(CLIPVisionEncoderWrapper, self).__init__()
        
        # Load pre-trained CLIP vision model with projection head
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        
        # Freeze all parameters - no gradients during training
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        self.embedding_dim = self.vision_model.config.projection_dim
        
        # Set to evaluation mode
        self.vision_model.eval()
        
        print(f"CLIP Vision Encoder initialized with projection head")
        print(f"Hidden size: {self.vision_model.config.hidden_size}")
        print(f"Projection dimension: {self.embedding_dim}")
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():  # Ensure no gradients are computed
            outputs = self.vision_model(pixel_values=image)
            # Use image_embeds which are the projected embeddings 
            # These are the embeddings CLIP uses for contrastive learning with text
            image_embedding = outputs.image_embeds
            
        return image_embedding

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, fmri_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        # L2 normalize embeddings for cosine similarity
        fmri_embeddings = F.normalize(fmri_embeddings, p=2, dim=-1)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        
        # Compute similarity matrices
        logits_per_fmri = fmri_embeddings @ image_embeddings.T / self.temperature
        logits_per_image = image_embeddings @ fmri_embeddings.T / self.temperature
        
        # Create labels (diagonal entries are positive pairs)
        batch_size = fmri_embeddings.shape[0]
        labels = torch.arange(batch_size).to(fmri_embeddings.device)
        
        # Compute cross-entropy loss in both directions
        loss_fmri = F.cross_entropy(logits_per_fmri, labels)
        loss_image = F.cross_entropy(logits_per_image, labels)
        
        # Return average of bidirectional losses
        return (loss_fmri + loss_image) / 2

class FMRIImageAdapter(nn.Module):
    def __init__(self, num_voxels: int, embedding_dim: int):
        super(FMRIImageAdapter, self).__init__()
        
        self.num_voxels = num_voxels
        self.embedding_dim = embedding_dim 
        
        self.adapter = nn.Linear(num_voxels, embedding_dim)
        
    def forward(self, fmri_data: torch.Tensor) -> torch.Tensor:
        return self.adapter(fmri_data)

class FMRIClipAlignmentModel(nn.Module):
    def __init__(self, 
                 num_voxels: int, 
                 embedding_dim: int, 
                 temperature: float = 0.07,
                 clip_model_name: str = "openai/clip-vit-base-patch32"):
        super(FMRIClipAlignmentModel, self).__init__()
        
        self.embedding_dim       = embedding_dim
        self.clip_vision_encoder = CLIPVisionEncoderWrapper(clip_model_name)
        self.fmri_image_adapter  = FMRIImageAdapter(num_voxels, embedding_dim)
        self.contrastive_loss    = InfoNCELoss(temperature)
        
    def forward(self, fmri_data: torch.Tensor, image_data: torch.Tensor) -> torch.Tensor:
        # Get CLIP image embedding (frozen)
        clip_image_embedding = self.clip_vision_encoder(image_data)
        
        # Adapt fMRI data directly to CLIP embedding space
        adapted_fmri_embedding = self.fmri_image_adapter(fmri_data)
        
        # Compute contrastive loss
        loss = self.contrastive_loss(adapted_fmri_embedding, clip_image_embedding)
        return loss
    
    def get_embeddings(self, fmri_data: torch.Tensor, image_data: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            adapted_fmri_embedding = self.fmri_image_adapter(fmri_data)
            
            if image_data is not None:
                clip_image_embedding = self.clip_vision_encoder(image_data)
                return adapted_fmri_embedding, clip_image_embedding
            else:
                return adapted_fmri_embedding, None

def train_model(model, train_loader, val_loader, config):
    # Extract parameters from config
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config.get('weight_decay', 0.01)
    gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
    warmup_epochs = config.get('warmup_epochs', 5)
    early_stopping_patience = config.get('early_stopping_patience', 15)
    use_mixed_precision = config.get('use_mixed_precision', False)
    eval_every_n_epochs = config.get('eval_every_n_epochs', 5)
    accumulation_steps = config.get('accumulation_steps', 4) 
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Schedulers
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    
    # Mixed precision - Updated for new PyTorch version
    scaler = torch.amp.GradScaler('cuda', enabled=use_mixed_precision)
    
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    print(f"Starting training for {num_epochs} epochs (effective batch size: {config['batch_size']*accumulation_steps})")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        # Track total batches 
        total_batches_processed = 0
        
        for batch_idx, (images, fmri_vectors) in enumerate(train_loader, 1):
            # Move data and add noise
            images = images.to(device)
            fmri_vectors = fmri_vectors.to(device)
            if config.get('fmri_noise_std', 0) > 0:
                noise = torch.randn_like(fmri_vectors) * config['fmri_noise_std']
                fmri_vectors = fmri_vectors + noise
            
            with torch.autograd.set_detect_anomaly(True):
                # Forward pass - Updated for new PyTorch version
                with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                    loss = model(fmri_vectors, images) / accumulation_steps  # Scale loss
                
                # Backward pass
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            total_train_loss += loss.item() * accumulation_steps  # Unscale for logging
            total_batches_processed += 1
            
            # Step only after accumulation_steps
            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader):
                if gradient_clip_norm > 0:
                    if use_mixed_precision:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                
                if use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Log every 10 batches 
            if batch_idx % (10 * accumulation_steps) == 0:
                avg_loss = total_train_loss / total_batches_processed
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
        
        # --- VALIDATION PHASE ---
        if epoch % eval_every_n_epochs == 0 or epoch == num_epochs - 1:
            model.eval()
            total_val_loss = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for images, fmri_vectors in val_loader:
                    images = images.to(device)
                    fmri_vectors = fmri_vectors.to(device)
                    
                    if use_mixed_precision:
                        with torch.amp.autocast('cuda'):
                            loss = model(fmri_vectors, images)
                    else:
                        loss = model(fmri_vectors, images)
                    
                    total_val_loss += loss.item()
                    num_val_batches += 1
            
            avg_val_loss = total_val_loss / num_val_batches
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"  *** New best validation loss: {best_val_loss:.4f} ***")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter} evaluation periods")
                
                if patience_counter >= early_stopping_patience // eval_every_n_epochs:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Update learning rate
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        # Log epoch results
        avg_train_loss = total_train_loss / len(train_loader)  # Normalize by total batches
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{num_epochs} Complete:")
        print(f"  Average Train Loss: {avg_train_loss:.4f}")
        if epoch % eval_every_n_epochs == 0:
            print(f"  Average Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print("-" * 50)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def create_dataset(expertise_group, config, ROI_mask, selected_voxel_indices=None):
    stimulus_images_paths = sorted(glob.glob("../data/bird_images/*.png"))
    
    # CLIP-specific preprocessing for ViT-B/32
    transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(), 
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])
    
    all_images = []
    for image_path in stimulus_images_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)  
        all_images.append(image)
    
    images = []
    fmri_vectors = []
    all_stimulus_ids = []
    
    for subj in expertise_group:
        subject = f'sub-{subj}'
        bold, stimulus_ids = load_data(
            **config["data_params"],
            subject = subject,
        )  
        
        # Apply ROI mask first
        roi_mask_bool = ROI_mask.astype(bool)
        bold = bold[:, roi_mask_bool]
        
        # If specific voxel indices are provided, select only those
        if selected_voxel_indices is not None:
            # Map selected indices to ROI space
            roi_indices = np.where(roi_mask_bool)[0]
            roi_selected_mask = np.isin(roi_indices, selected_voxel_indices)
            bold = bold[:, roi_selected_mask]
        
        # Handle NaN values
        bold_nan = np.isnan(bold) 
        bold[bold_nan] = 0.
        
        fmri_vectors.extend(bold) 
        all_stimulus_ids.extend(stimulus_ids)
        
        images_ids = stimulus_ids
        images_to_use = [all_images[i] for i in images_ids.flatten()]
        images.extend(images_to_use)
    
    fmri_vectors = np.array(fmri_vectors)
    images = np.array(images)
    all_stimulus_ids = np.array(all_stimulus_ids)
    
    if selected_voxel_indices is None:
        return images, fmri_vectors, all_stimulus_ids
    else:
        return images, fmri_vectors

def create_ROI_masks(ROI_combos):
    glasser_L = nib.freesurfer.io.read_annot(data_root / "lh.HCPMMP1.annot")
    glasser_R = nib.freesurfer.io.read_annot(data_root / "rh.HCPMMP1.annot")
    ROI_masks = {}
    for key, vals in ROI_combos.items():
        # mask glasser atlas to mark current loop ROI as 1s
        L_mask = np.isin(glasser_L[0], vals) # vals is a list of ROIs to set as 1
        R_mask = np.isin(glasser_R[0], vals)
        print(f"{key}: {sum(L_mask) + sum(R_mask)}")
        
        # concatenate left and right hemispheres 
        L_R_concat_mask = np.concatenate([L_mask, R_mask], axis=0)
        ROI_masks[key] = L_R_concat_mask
    
    return ROI_masks

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    num_runs = 6
    tr = 2
    data_params = dict(
        path = shared_dataset_root / "hdf5s/tc2see-fsaverage-surfs.hdf5", 
        tr_offset = num_runs / tr,
        run_normalize='linear_trend',
        interpolation=False,
    )
    
    # Configuration
    config = {
        'experiment_version': '100_epochs_all_subs',
        'clip_model_name': 'openai/clip-vit-base-patch32', 
        'eval_every_n_epochs': 1,         
        'save_embeddings': True,  
        'noise_ceiliing_percentage': 40,

        'num_epochs': 100,                
        'learning_rate': 5e-5,            
        'weight_decay': 0.1,              
        'gradient_clip_norm': 0.5,       
        'accumulation_steps': 8,          
        'warmup_epochs': 10,              
        'lr_scheduler': 'cosine_with_warmup',  
        'use_mixed_precision': True,      
        'early_stopping_patience': 10,    
        'temperature': 0.1,     

        'train_split': 0.7,              
        'val_split': 0.15,               
        'test_split': 0.15,              
        'batch_size': 32,                
        'data_params': data_params,      
        'fmri_noise_std': 0.1,           
        'embedding_dim': 512,                   
    }
    
    expertise_groups = {
        'low': ['06', '15', '20', '21', '22', '23', '28', '29', '32', '33', '39'],
        'high': ['08', '09', '11', '12', '16', '24', '26', '35', '38'],
        # 'all_subs': ['05', '06', '07', '08', '09', '11', '12', '14', '15', '16', '17', 
        #         '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
        #         '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'] # no sub 10, figure out why
    }
    
    ROI_combos = {
        # "all_rois": [roi for roi in range(1, 181)],
        # "V1": [1],
        "V2": [4],
        # "V3": [5],
        "V4": [6],
        # "V7": [16],
        # "V8": [7],
        "LO1": [20],
        "LO2": [21],
        "PIT": [22],
        # "FFC": [18],
        # "VVC": [163],
        "A1": [24]
        # "Pir": [110],
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ROI_masks = create_ROI_masks(ROI_combos)
    
    # Train a model for each ROI combination
    for ROI_combo, ROI_mask in ROI_masks.items():
        print(f"\n====== {ROI_combo} ======\n")
        for group, participant_list in expertise_groups.items():
            print(f"\n{'='*30} {group} {'='*30}")
            print(f"Training model for {group} scoring participant group, ROI: {ROI_combo}")
            print(f"Number of voxels: {sum(ROI_mask)}")
            print(f"{'='*60}")
            
            print("\nLoading data for noise ceiling computation...")
            images_all, fmri_vectors_all, stimulus_ids_all = create_dataset(
                participant_list, config, ROI_mask, selected_voxel_indices=None
            )

            print(f"Total dataset: {len(images_all)} samples")
            print(f"fMRI shape before voxel selection: {fmri_vectors_all[0].shape}")

            # Split data indices
            num_train = int(config['train_split'] * len(images_all))
            num_val = int(config['val_split'] * len(images_all))

            # Get training data for noise ceiling computation
            train_fmri_for_nc = fmri_vectors_all[:num_train]
            train_stimulus_ids = stimulus_ids_all[:num_train]

            # Compute noise ceiling and get top voxel indices 
            selected_voxel_indices, nc_values = get_top_nc_voxel_indices(
                train_fmri_for_nc, train_stimulus_ids, ROI_mask, config['noise_ceiliing_percentage']
            )

            images_all, fmri_vectors_all, stimulus_ids_all = create_dataset(
                participant_list, config, ROI_mask, selected_voxel_indices=None
            )

            print(f"fMRI shape after ROI masking: {fmri_vectors_all[0].shape}")

            # Split data indices
            num_train = int(config['train_split'] * len(images_all))
            num_val = int(config['val_split'] * len(images_all))

            # Get training data for noise ceiling computation
            train_fmri_for_nc = fmri_vectors_all[:num_train]
            train_stimulus_ids = stimulus_ids_all[:num_train]

            # Compute noise ceiling and get top voxel indices
            print(f"Computing noise ceiling and selecting top {config['noise_ceiliing_percentage']}% of voxels...")
            selected_voxel_indices, nc_values = get_top_nc_voxel_indices(
                train_fmri_for_nc, train_stimulus_ids, ROI_mask, config['noise_ceiliing_percentage']
            )

            print(f"Selected {len(selected_voxel_indices)} voxels out of {np.sum(ROI_mask)} ROI voxels")
            print(f"Noise ceiling range: {np.min(nc_values[nc_values > 0]):.3f} - {np.max(nc_values):.3f}")

            # Create dataset with selected voxels
            images, fmri_vectors = create_dataset(
                participant_list, config, ROI_mask, selected_voxel_indices=selected_voxel_indices
            )

            print(f"fMRI shape after voxel selection: {fmri_vectors[0].shape}")
            
            train_images = images[:num_train]
            train_fmri = fmri_vectors[:num_train]
            
            val_images = images[num_train:num_train+num_val]
            val_fmri = fmri_vectors[num_train:num_train+num_val]
            
            test_images = images[num_train+num_val:]
            test_fmri = fmri_vectors[num_train+num_val:]
            
            print(f"Data splits - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
            
            # Create datasets
            train_dataset = BrainImageDataset(train_images, train_fmri)
            val_dataset = BrainImageDataset(val_images, val_fmri)  
            test_dataset = BrainImageDataset(test_images, test_fmri)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                    num_workers=8, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                                    num_workers=8, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, 
                                    num_workers=8, pin_memory=True)
            
            # Initialize model with correct number of voxels for this ROI
            if selected_voxel_indices is not None:
                num_voxels = len(selected_voxel_indices)
            else:
                num_voxels = sum(ROI_mask)

            print(f"\nInitializing model for {num_voxels} selected voxels...")
            model = FMRIClipAlignmentModel(
                num_voxels=num_voxels,  # This should now match your actual data dimensions
                embedding_dim=config['embedding_dim'],
                temperature=config['temperature'],
                clip_model_name=config['clip_model_name']
            )
            
            print(f"Model initialized with embedding dimension: {model.embedding_dim}")
            print(f"CLIP Vision Encoder frozen: {not any(p.requires_grad for p in model.clip_vision_encoder.parameters())}")
            print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            # Train the model
            print(f"\nStarting training for ROI: {ROI_combo}")
            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config 
            )
            
            print(f"Training completed for ROI: {ROI_combo}")
            
            save_dir = results_dir / f"brain_adapter/{config['experiment_version']}"
            os.makedirs(save_dir, exist_ok=True)
            model_save_path = save_dir / f"fmri_clip_model_{ROI_combo}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'roi_combo': ROI_combo,
                'group': group,
                'num_voxels': num_voxels,
                'embedding_dim': model.embedding_dim
            }, model_save_path)
            print(f"Model saved to: {model_save_path}")
            print(f"Model saved to: {model_save_path}")
            print("Loading best model for final evaluation...")
            
            # Free memory used during training
            del model
            torch.cuda.empty_cache()
            model = FMRIClipAlignmentModel(
                num_voxels=num_voxels,
                embedding_dim=config['embedding_dim'],
                temperature=config['temperature'],
                clip_model_name=config['clip_model_name']
            )
            model.to(device)
            saved_model = torch.load(model_save_path, map_location=device, weights_only=False)
            model.load_state_dict(saved_model['model_state_dict'])
            
            print("\nEvaluating model on test set...")
            metrics = evaluate_alignment_quality(
                model, 
                test_loader, 
                device, 
                save_dir,
                config["experiment_version"],
                k_values=[1, 5, 10, 20], 
                save_prefix=f"{group}_{ROI_combo}"
            )
            
            # Print summary of key metrics
            print("\nEvaluation Summary:")
            print(f"Best fMRI→Image R@1: {metrics['fMRI_to_Image_R@1']:.4f}")
            print(f"Best Image→fMRI R@1: {metrics['Image_to_fMRI_R@1']:.4f}")
            print(f"Separation Score: {metrics['separation_score']:.4f}")

if __name__ == "__main__":
    main()