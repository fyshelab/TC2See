import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import os
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
import torchvision.transforms as T
from PIL import Image
import nibabel as nib
import glob
from pathlib import Path
import os
import copy

def evaluate_alignment_quality(
    model, 
    test_loader, 
    device, 
    save_dir,
    experiment_version, 
    k_values=[1, 5, 10], 
    save_prefix=""
):
    """
    Comprehensive evaluation of fMRI-image alignment quality with saved visualizations.
    
    Args:
        model: Trained FMRIClipAlignmentModel
        test_loader: Test data loader
        device: Computing device
        save_dir: where to save visualizations
        experiment_version: Suffix for saving output files
        k_values: List of k values for Recall@K evaluation
        save_prefix: Prefix for saving output files
        
    Returns:
        Dictionary containing evaluation metrics and saves visualizations to disk
    """
    model.eval()
    
    visualizations_dir = save_dir / "visulizations"
    # Create directory for saving results if it doesn't exist
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Extract all embeddings
    all_fmri_embeddings = []
    all_image_embeddings = []
    
    print("Extracting embeddings from test set...")
    with torch.no_grad():
        for images, fmri_vectors in test_loader:
            images = images.to(device)
            fmri_vectors = fmri_vectors.to(device)
            
            fmri_emb, img_emb = model.get_embeddings(fmri_vectors, images)
            
            all_fmri_embeddings.append(fmri_emb.cpu())
            all_image_embeddings.append(img_emb.cpu())
    
    # Concatenate all embeddings
    fmri_embeddings = torch.cat(all_fmri_embeddings, dim=0)
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    
    # Normalize embeddings
    fmri_embeddings = F.normalize(fmri_embeddings, p=2, dim=-1)
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    
    print(f"Evaluating alignment on {len(fmri_embeddings)} test samples...")
    
    # Compute similarity matrices
    fmri_to_image_sim = fmri_embeddings @ image_embeddings.T
    image_to_fmri_sim = image_embeddings @ fmri_embeddings.T
    
    metrics = {}
    
    # 1. Retrieval Metrics (Recall@K and MRR)
    print("\nComputing retrieval metrics...")
    metrics.update(compute_retrieval_metrics(fmri_to_image_sim, image_to_fmri_sim, k_values))
    
    # 2. Alignment Quality Metrics
    print("\nComputing alignment quality metrics...")
    metrics.update(compute_alignment_metrics(fmri_embeddings, image_embeddings, fmri_to_image_sim))
    
    # 3. Visualization
    print("\nGenerating visualizations...")
    generate_visualizations(fmri_embeddings, image_embeddings, fmri_to_image_sim, save_prefix, visualizations_dir)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_file = save_dir / f"{save_prefix}_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to: {metrics_file}")
    
    return metrics

def compute_retrieval_metrics(fmri_to_image_sim, image_to_fmri_sim, k_values):
    """Compute retrieval metrics (Recall@K and MRR) for both directions"""
    metrics = {}
    
    # fMRI to Image retrieval
    fmri_to_img_recalls = {}
    for k in k_values:
        _, top_k_indices = torch.topk(fmri_to_image_sim, k, dim=1)
        correct = torch.any(top_k_indices == torch.arange(len(fmri_to_image_sim)).unsqueeze(1), dim=1).sum().item()
        recall_at_k = correct / len(fmri_to_image_sim)
        fmri_to_img_recalls[f'fMRI_to_Image_R@{k}'] = recall_at_k
        print(f"fMRI→Image Recall@{k}: {recall_at_k:.4f}")
    
    # Image to fMRI retrieval
    img_to_fmri_recalls = {}
    for k in k_values:
        _, top_k_indices = torch.topk(image_to_fmri_sim, k, dim=1)
        correct = torch.any(top_k_indices == torch.arange(len(image_to_fmri_sim)).unsqueeze(1), dim=1).sum().item()
        recall_at_k = correct / len(image_to_fmri_sim)
        img_to_fmri_recalls[f'Image_to_fMRI_R@{k}'] = recall_at_k
        print(f"Image→fMRI Recall@{k}: {recall_at_k:.4f}")
    
    metrics.update(fmri_to_img_recalls)
    metrics.update(img_to_fmri_recalls)
    
    # Mean reciprocal rank (MRR)
    def compute_mrr(similarity_matrix):
        ranks = []
        for i in range(similarity_matrix.shape[0]):
            # Get rank of correct match (diagonal element)
            similarities = similarity_matrix[i]
            sorted_indices = torch.argsort(similarities, descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)
        return torch.tensor(ranks).mean().item()
    
    metrics['fMRI_to_Image_MRR'] = compute_mrr(fmri_to_image_sim)
    metrics['Image_to_fMRI_MRR'] = compute_mrr(image_to_fmri_sim)
    
    print(f"fMRI→Image MRR: {metrics['fMRI_to_Image_MRR']:.4f}")
    print(f"Image→fMRI MRR: {metrics['Image_to_fMRI_MRR']:.4f}")
    
    return metrics

def compute_alignment_metrics(fmri_embeddings, image_embeddings, fmri_to_image_sim):
    """Compute various alignment quality metrics"""
    metrics = {}
    
    # Intra-modal vs inter-modal similarities
    fmri_self_sim = fmri_embeddings @ fmri_embeddings.T
    image_self_sim = image_embeddings @ image_embeddings.T
    
    # Remove diagonal (self-similarities)
    mask = ~torch.eye(len(fmri_embeddings), dtype=bool)
    
    intra_fmri_sims = fmri_self_sim[mask].numpy()
    intra_image_sims = image_self_sim[mask].numpy()
    inter_modal_sims = fmri_to_image_sim[mask].numpy()
    correct_matches = torch.diag(fmri_to_image_sim).numpy()
    
    metrics['intra_fMRI_sim_mean'] = np.mean(intra_fmri_sims)
    metrics['intra_fMRI_sim_std'] = np.std(intra_fmri_sims)
    metrics['intra_Image_sim_mean'] = np.mean(intra_image_sims)
    metrics['intra_Image_sim_std'] = np.std(intra_image_sims)
    metrics['inter_modal_sim_mean'] = np.mean(inter_modal_sims)
    metrics['inter_modal_sim_std'] = np.std(inter_modal_sims)
    metrics['correct_match_sim_mean'] = np.mean(correct_matches)
    metrics['correct_match_sim_std'] = np.std(correct_matches)
    
    print(f"Intra-fMRI similarity: {metrics['intra_fMRI_sim_mean']:.4f} ± {metrics['intra_fMRI_sim_std']:.4f}")
    print(f"Intra-Image similarity: {metrics['intra_Image_sim_mean']:.4f} ± {metrics['intra_Image_sim_std']:.4f}")
    print(f"Inter-modal similarity: {metrics['inter_modal_sim_mean']:.4f} ± {metrics['inter_modal_sim_std']:.4f}")
    print(f"Correct matches similarity: {metrics['correct_match_sim_mean']:.4f} ± {metrics['correct_match_sim_std']:.4f}")
    
    # Separation Quality
    metrics['separation_score'] = metrics['correct_match_sim_mean'] - metrics['inter_modal_sim_mean']
    print(f"Separation score (higher is better): {metrics['separation_score']:.4f}")
    
    return metrics

def generate_visualizations(fmri_embeddings, image_embeddings, similarity_matrix, save_prefix, visualizations_dir):
    """Generate and save various visualizations of the alignment"""
    # 1. Similarity distributions plot
    plt.figure(figsize=(10, 6))
    diagonal_sims = torch.diag(similarity_matrix).numpy()
    off_diagonal_sims = similarity_matrix[~torch.eye(similarity_matrix.shape[0], dtype=bool)].numpy()
    
    sns.kdeplot(off_diagonal_sims, label='Negative Pairs', fill=True)
    sns.kdeplot(diagonal_sims, label='Positive Pairs', fill=True)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.savefig(visualizations_dir / f"{save_prefix}_similarity_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dimensionality reduction visualization
    plot_embedding_space(fmri_embeddings, image_embeddings, save_prefix, visualizations_dir)
    
    # 3. Rank distribution
    plt.figure(figsize=(10, 6))
    ranks = []
    for i in range(similarity_matrix.shape[0]):
        similarities = similarity_matrix[i]
        sorted_indices = torch.argsort(similarities, descending=True)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    plt.hist(ranks, bins=min(50, max(ranks)), alpha=0.7)
    plt.xlabel('Rank of Correct Match')
    plt.ylabel('Frequency')
    plt.title('Rank Distribution')
    plt.axvline(x=np.mean(ranks), color='red', linestyle='--', label=f'Mean Rank: {np.mean(ranks):.1f}')
    plt.legend()
    plt.savefig(visualizations_dir / f"{save_prefix}_rank_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Similarity matrix heatmap (sample of first 100)
    sample_size = min(100, similarity_matrix.shape[0])
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix[:sample_size, :sample_size].numpy(), cmap='viridis')
    plt.title('fMRI-Image Similarity Matrix (Sample)')
    plt.xlabel('Image Index')
    plt.ylabel('fMRI Index')
    plt.savefig(visualizations_dir / f"{save_prefix}_similarity_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_embedding_space(fmri_embeddings, image_embeddings, save_prefix, visualizations_dir):
    """Visualize the embedding space using dimensionality reduction"""
    # Combine embeddings for joint dimensionality reduction
    combined_embeddings = torch.cat([fmri_embeddings, image_embeddings], dim=0)
    labels = ['fMRI'] * len(fmri_embeddings) + ['Image'] * len(image_embeddings)
    
    # Reduce dimensionality using PCA first (for efficiency), then t-SNE
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(combined_embeddings.numpy())
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(pca_result)
    
    # Create dataframe for plotting
    plot_df = pd.DataFrame({
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        'label': labels,
        'type': ['Embedding'] * len(tsne_result)
    })
    
    # Plot t-SNE visualization
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x='x',
        y='y',
        hue='label',
        style='type',
        palette=['blue', 'orange'],
        alpha=0.6,
        s=50
    )
    plt.title('t-SNE Visualization of fMRI and Image Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Modality')
    plt.savefig(visualizations_dir / f"{save_prefix}_tsne_embedding.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also plot PCA visualization (first 2 components)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=pca_result[:, 0],
        y=pca_result[:, 1],
        hue=labels,
        palette=['blue', 'orange'],
        alpha=0.6,
        s=50
    )
    plt.title('PCA Visualization of fMRI and Image Embeddings')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend(title='Modality')
    plt.savefig(visualizations_dir / f"{save_prefix}_pca_embedding.png", dpi=300, bbox_inches='tight')
    plt.close()