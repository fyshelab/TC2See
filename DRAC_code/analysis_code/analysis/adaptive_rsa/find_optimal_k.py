import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import pickle
import json
from pathlib import Path


dataset_root = Path("/project/6029407/jamesmck/TC2See/DRAC_code/data")
results_dir = Path("/project/6029407/jamesmck/TC2See/DRAC_code/results")

avg_duplicates = True
rdm_dist = "correlation"
mod_im_combos = [("ViT-B=16", "bird"), ("ViT-B=16", "bg_mask"), ("ViT-B=16", "max_dist"), 
                 ("ViT-B=32", "bird"), ("ViT-B=32", "bg_mask"), ("ViT-B=32", "max_dist")]
model_name, image_type = mod_im_combos[3]  
clustering_approach = "agglomerative"  # or "kmeans"
linkage = 'complete' # 'ward', 'complete', 'average'


# Your existing code
with h5py.File(dataset_root / f'{model_name}-features_{image_type}.hdf5', 'r') as f:
    clip_embeddings = f['embedding'][:]

print(f"Loaded {clip_embeddings.shape[0]} embeddings with {clip_embeddings.shape[1]} dimensions")

# Step 1: Determine optimal number of clusters
def find_optimal_clusters(embeddings, clustering_approach="kmeans", max_k=20, min_k=2, linkage='ward'):
    """
    Find optimal number of clusters using elbow method and silhouette analysis
    """
    print(f"Finding optimal number of clusters using {clustering_approach}...")
    
    # Test different numbers of clusters
    k_range = range(min_k, min(max_k + 1, len(embeddings) // 2))  # Don't exceed half the data points
    
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"Testing k={k}...")
        
        if clustering_approach == "kmeans":
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(sil_score)
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
            
        elif clustering_approach == "agglomerative":
            
            # Fit Agglomerative Clustering
            agglomerative = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            cluster_labels = agglomerative.fit_predict(embeddings)
            
            # Calculate inertia manually (within-cluster sum of squares)
            inertia = 0
            for cluster_id in range(k):
                cluster_mask = cluster_labels == cluster_id
                if np.sum(cluster_mask) > 0:  # Check if cluster is not empty
                    cluster_points = embeddings[cluster_mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    inertia += np.sum((cluster_points - cluster_center) ** 2)
            
            inertias.append(inertia)
            sil_score = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(sil_score)
            
            print(f"k={k}: Inertia={inertia:.2f}, Silhouette={sil_score:.3f}")
    
    return list(k_range), inertias, silhouette_scores

# Find optimal clusters
k_values, inertias, silhouette_scores = find_optimal_clusters(clip_embeddings, clustering_approach, 20, 2, linkage)


plt.figure(figsize=(12, 5))

# Plot 1: Elbow method
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)


# Plot 2: Silhouette method
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plot_file = Path(__file__).parent / f'{model_name}_{clustering_approach}_{linkage}_clustering_analysis.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Saved clustering analysis plot to: {plot_file}")