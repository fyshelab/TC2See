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


class BrainImageDataset(Dataset):
    """
    Dataset class for paired fMRI and image data.
    """
    def __init__(self, images, fmri_vectors):
        """
        Args:
            images: numpy array or tensor of images with shape (N, C, H, W)
            fmri_vectors: numpy array or tensor of fMRI vectors with shape (N, num_voxels)
        """
        self.images = torch.tensor(images, dtype=torch.float32) if isinstance(images, np.ndarray) else images
        self.fmri_vectors = torch.tensor(fmri_vectors, dtype=torch.float32) if isinstance(fmri_vectors, np.ndarray) else fmri_vectors
        
        assert len(self.images) == len(self.fmri_vectors), "Images and fMRI vectors must have the same length"
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.fmri_vectors[idx]


class FMRIEncoder(nn.Module):
    """
    Multi-Layer Perceptron to encode high-dimensional fMRI vectors into embeddings.
    Maps from (batch_size, num_voxels) to (batch_size, embedding_dim).
    """
    def __init__(self, num_voxels: int, embedding_dim: int):
        """
        Args:
            num_voxels: Input dimension of the fMRI vector 
            embedding_dim: Output dimension matching CLIP embeddings (e.g., 512)
        """
        super(FMRIEncoder, self).__init__()
        
        self.num_voxels = num_voxels
        self.embedding_dim = embedding_dim
        
        # MLP architecture: input -> hidden -> output
        self.encoder = nn.Sequential(
            nn.Linear(num_voxels, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, fmri_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fMRI encoder.
        
        Args:
            fmri_data: Input fMRI tensor of shape (batch_size, num_voxels)
            
        Returns:
            fmri_embedding: Output tensor of shape (batch_size, embedding_dim)
        """
        return self.encoder(fmri_data)


class CLIPVisionEncoderWrapper(nn.Module):
    """
    Wrapper for pre-trained CLIP Vision Encoder with projection head and frozen parameters.
    Extracts semantically-rich visual embeddings from bird images using the projection head
    that was trained for contrastive alignment with text embeddings.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            model_name: Name of the pre-trained CLIP model to load
        """
        super(CLIPVisionEncoderWrapper, self).__init__()
        
        # Load pre-trained CLIP vision model WITH projection head
        # This gives us the semantically-rich embeddings used for image-text alignment
        self.vision_model = CLIPVisionModelWithProjection.from_pretrained(model_name)
        
        # Freeze all parameters - no gradients during training
        for param in self.vision_model.parameters():
            param.requires_grad = False
            
        # Store embedding dimension for compatibility checks
        # Use projection_dim which is the output of the projection head (typically 512)
        self.embedding_dim = self.vision_model.config.projection_dim
        
        # Set to evaluation mode
        self.vision_model.eval()
        
        print(f"CLIP Vision Encoder initialized with projection head")
        print(f"Hidden size: {self.vision_model.config.hidden_size}")
        print(f"Projection dimension: {self.embedding_dim}")
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract projected visual embeddings from images using frozen CLIP encoder.
        Uses the projection head that was trained for contrastive image-text alignment.
        
        Args:
            image: Input image tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            image_embedding: Projected visual embedding of shape (batch_size, projection_dim)
        """
        with torch.no_grad():  # Ensure no gradients are computed
            outputs = self.vision_model(pixel_values=image)
            # Use image_embeds which are the projected embeddings (not pooler_output)
            # These are the embeddings CLIP uses for contrastive learning with text
            image_embedding = outputs.image_embeds
            
        return image_embedding


class FMRIImageAdapter(nn.Module):
    """
    Advanced adapter for aligning fMRI embeddings with CLIP visual embeddings.
    Uses projection, cross-modal attention, residual connections, and learnable bias.
    """
    def __init__(self, embedding_dim: int, num_attention_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embedding_dim: Dimension of both fMRI and image embeddings
            num_attention_heads: Number of heads for Multi-Head Attention
            dropout: Dropout rate for attention layers
        """
        super(FMRIImageAdapter, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # fMRI projection layer for semantic alignment
        self.fmri_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Learnable bias to correct for domain shifts
        self.learnable_bias = nn.Parameter(torch.zeros(1, embedding_dim))
        
        # Cross-modal attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout for residual connections
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network after attention
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, fmri_embedding: torch.Tensor, clip_image_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align fMRI embeddings with CLIP visual embeddings through cross-modal attention.
        
        Args:
            fmri_embedding: fMRI tensor of shape (batch_size, embedding_dim)
            clip_image_embedding: CLIP image tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Tuple of (aligned_fmri_embedding, aligned_image_embedding)
        """
        # Apply projection and learnable bias to fMRI features
        fmri_features_proj = self.fmri_projection(fmri_embedding) + self.learnable_bias
        
        # Prepare inputs for attention 
        # MultiheadAttention expects (batch_size, sequence_len, embed_dim) 
        fmri_query = fmri_features_proj.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        image_kv = clip_image_embedding.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        
        # Cross-modal attention: fMRI attends to image features
        attn_output, _ = self.cross_attention(
            query=fmri_query,
            key=image_kv,
            value=image_kv
        )
        
        # Remove sequence dimension and apply residual connection with layer norm
        attn_output = attn_output.squeeze(1)  # (batch_size, embedding_dim)
        adapted_fmri_embedding = self.norm1(fmri_features_proj + self.dropout1(attn_output))
        
        # Apply feed-forward network with another residual connection
        mlp_output = self.mlp(adapted_fmri_embedding)
        adapted_fmri_embedding = self.norm2(adapted_fmri_embedding + self.dropout2(mlp_output))

        unchanged_image_embedding = clip_image_embedding
        
        return adapted_fmri_embedding, unchanged_image_embedding


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Information Noise-Contrastive Estimation) loss for multimodal alignment.
    Implements bidirectional contrastive learning between fMRI and image embeddings.
    """
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Scaling factor for softmax (controls hardness of negatives)
        """
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, fmri_embeddings: torch.Tensor, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional InfoNCE loss.
        
        Args:
            fmri_embeddings: fMRI embeddings of shape (batch_size, embedding_dim)
            image_embeddings: Image embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            Combined InfoNCE loss (scalar)
        """
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


class FMRIClipAlignmentModel(nn.Module):
    """
    Complete multimodal system orchestrating fMRI encoder, CLIP vision encoder, and adapter.
    Learns to align fMRI brain activity with CLIP visual embeddings via contrastive learning.
    """
    def __init__(self, 
                 num_voxels: int, 
                 embedding_dim: int, 
                 num_attention_heads: int = 8, 
                 temperature: float = 0.07,
                 clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Args:
            num_voxels: Dimension of input fMRI vectors
            embedding_dim: Dimension of embedding space
            num_attention_heads: Number of attention heads in adapter
            temperature: Temperature parameter for InfoNCE loss
            clip_model_name: Pre-trained CLIP model identifier
        """
        super(FMRIClipAlignmentModel, self).__init__()
        
        # Initialize all components
        self.fmri_encoder = FMRIEncoder(num_voxels, embedding_dim)
        self.clip_vision_encoder = CLIPVisionEncoderWrapper(clip_model_name)
        
        # Ensure embedding dimensions match
        clip_embed_dim = self.clip_vision_encoder.embedding_dim
        if embedding_dim != clip_embed_dim:
            print(f"Warning: Specified embedding_dim ({embedding_dim}) != CLIP embedding_dim ({clip_embed_dim})")
            print(f"Using CLIP embedding_dim: {clip_embed_dim}")
            embedding_dim = clip_embed_dim
            # Reinitialize fMRI encoder with correct dimensions
            self.fmri_encoder = FMRIEncoder(num_voxels, embedding_dim)
        
        self.fmri_image_adapter = FMRIImageAdapter(embedding_dim, num_attention_heads)
        self.contrastive_loss = InfoNCELoss(temperature)
        
        self.embedding_dim = embedding_dim
        
    def forward(self, fmri_data: torch.Tensor, image_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing contrastive loss for fMRI-image alignment.
        
        Args:
            fmri_data: fMRI vectors of shape (batch_size, num_voxels)
            image_data: Images of shape (batch_size, 3, 224, 224)
            
        Returns:
            InfoNCE contrastive loss (scalar)
        """
        # Extract embeddings from both modalities
        fmri_embedding = self.fmri_encoder(fmri_data)
        clip_image_embedding = self.clip_vision_encoder(image_data)
        
        # Align embeddings through adapter
        aligned_fmri_embedding, unchanged_image_embedding = self.fmri_image_adapter(
            fmri_embedding, clip_image_embedding
        )
        
        # Compute and return contrastive loss
        loss = self.contrastive_loss(aligned_fmri_embedding, unchanged_image_embedding)
        return loss
    
    def get_embeddings(self, fmri_data: torch.Tensor, image_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract aligned embeddings without computing loss (for evaluation).
        
        Args:
            fmri_data: fMRI vectors of shape (batch_size, num_voxels)
            image_data: Images of shape (batch_size, 3, 224, 224)
            
        Returns:
            Tuple of (aligned_fmri_embeddings, unchanged_image_embedding)
        """
        with torch.no_grad():
            fmri_embedding = self.fmri_encoder(fmri_data)
            clip_image_embedding = self.clip_vision_encoder(image_data)
            
            aligned_fmri_embedding, unchanged_image_embedding = self.fmri_image_adapter(
                fmri_embedding, clip_image_embedding
            )
            
        return aligned_fmri_embedding, unchanged_image_embedding


def train_model(model, train_loader, val_loader, config):
    """
    Enhanced training loop with gradient accumulation.
    """
    # Extract parameters from config
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config.get('weight_decay', 0.01)
    gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
    warmup_epochs = config.get('warmup_epochs', 5)
    early_stopping_patience = config.get('early_stopping_patience', 15)
    use_mixed_precision = config.get('use_mixed_precision', False)
    eval_every_n_epochs = config.get('eval_every_n_epochs', 5)
    accumulation_steps = config.get('accumulation_steps', 4)  # New: gradient accumulation
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Schedulers
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 1.0
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-warmup_epochs)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)
    
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Starting training for {num_epochs} epochs (effective batch size: {config['batch_size']*accumulation_steps})")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        # Track total batches (for proper averaging with accumulation)
        total_batches_processed = 0
        
        for batch_idx, (images, fmri_vectors) in enumerate(train_loader, 1):
            # Move data and add noise
            images = images.to(device)
            fmri_vectors = fmri_vectors.to(device)
            if config.get('fmri_noise_std', 0) > 0:
                noise = torch.randn_like(fmri_vectors) * config['fmri_noise_std']
                fmri_vectors = fmri_vectors + noise
            
            with torch.autograd.set_detect_anomaly(True):
                # Forward pass
                with torch.cuda.amp.autocast(enabled=use_mixed_precision):
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
            
            # Log every 10 batches (accounting for accumulation)
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
                        with torch.cuda.amp.autocast():
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


def create_dataset(expertise_group, config, ROI_mask):
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

    for subj in expertise_group:
        subject = f'sub-{subj}'
        bold, stimulus_ids = load_data(
            **config["data_params"],
            subject = subject,
        )  

        roi_mask_bool = ROI_mask.astype(bool)
        bold = bold[:, roi_mask_bool]
        bold_nan = np.isnan(bold) 
        bold[bold_nan] = 0.
        fmri_vectors.extend(bold) 

        images_ids = stimulus_ids
        images_to_use = [all_images[i] for i in images_ids.flatten()]
        images.extend(images_to_use)
    

    fmri_vectors = np.array(fmri_vectors)
    images = np.array(images)

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
    """
    Main training script integrating with your data pipeline
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Your data configuration - adjust these paths as needed
    num_runs = 6
    tr = 2
    data_params = dict(
        path = shared_dataset_root / "hdf5s/tc2see-fsaverage-surfs.hdf5",  # Update this path
        tr_offset = num_runs / tr,
        run_normalize='linear_trend',
        interpolation=False,
    )
    
    # Configuration
    config = {
        'experiment_version': '100_epochs_all_subs',
        'accumulation_steps': 4, 
        'batch_size': 64,
        'num_epochs': 100,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'learning_rate': 5e-5,
        'data_params': data_params,
        'embedding_dim': 512,
        'num_attention_heads': 8,
        'temperature': 0.05,
        'clip_model_name': 'openai/clip-vit-base-patch32',
        'weight_decay': 0.01,
        'gradient_clip_norm': 1.0,
        'warmup_epochs': 5,
        'early_stopping_patience': 15,
        'fmri_noise_std': 0.01,
        'use_mixed_precision': True,
        'eval_every_n_epochs': 5,
        'save_embeddings': True,
        'lr_scheduler': 'cosine_with_warmup',
    }
    
    # Your expertise groups
    expertise_groups = {
        # 'low': ['06', '15', '20', '21', '22', '23', '28', '29', '32', '33', '39'],
        # 'high': ['08', '09', '11', '12', '16', '24', '26', '35', '38'],
        'all_subs': ['05', '06', '07', '08', '09', '11', '12', '14', '15', '16', '17', 
                '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', 
                '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40'] # no sub 10, figure out why
    }
    
    # Your ROI combinations
    ROI_combos = {
        "all_rois": [roi for roi in range(1, 181)],
        # "V1": [1],
        "V2": [4],
        # "V3": [5],
        "V4": [6]
        # "V7": [16],
        # "V8": [7],
        # "LO1": [20],
        # "LO2": [21],
        "PIT": [22],
        # "FFC": [18],
        # "VVC": [163],
        "A1": [24],
        # "Pir": [110],
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create ROI masks using your function
    ROI_masks = create_ROI_masks(ROI_combos)
    
    # Train a model for each ROI combination
    for group, participant_list in expertise_groups.items():
        print(f"\n====== {group} scoring group ======\n")
        for ROI_combo, ROI_mask in ROI_masks.items():
            print(f"\n{'='*30} {group} {'='*30}")
            print(f"Training model for ROI: {ROI_combo}")
            print(f"Number of voxels: {sum(ROI_mask)}")
            print(f"{'='*60}")
            
            # Create dataset using your function
            images, fmri_vectors = create_dataset(participant_list, config, ROI_mask)
            
            print(f"Dataset created: {len(images)} samples")
            print(f"Image shape: {images[0].shape}")
            print(f"fMRI shape: {fmri_vectors[0].shape}")
            
            # Split data
            num_train = int(config['train_split'] * len(images))
            num_val = int(config['val_split'] * len(images))
            
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
            num_voxels = sum(ROI_mask)
            
            print(f"\nInitializing model for {num_voxels} voxels...")
            model = FMRIClipAlignmentModel(
                num_voxels=num_voxels,
                embedding_dim=config['embedding_dim'],
                num_attention_heads=config['num_attention_heads'],
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
                num_attention_heads=config['num_attention_heads'],
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