# cluster_and_rank.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import inspect

# Monkey patch for Python 3.12 compatibility
if not hasattr(inspect, "ArgSpec"):
    from collections import namedtuple
    inspect.ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
    
import sys
import types

# Monkeypatch to bypass parametric_umap
sys.modules["umap.parametric_umap"] = types.ModuleType("umap.parametric_umap")
setattr(sys.modules["umap.parametric_umap"], "ParametricUMAP", None)
setattr(sys.modules["umap.parametric_umap"], "load_ParametricUMAP", None)


from umap.umap_ import UMAP
from utils.textrank import rank_summaries
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score
import hdbscan


df = pd.read_csv("data/cleaned_summaries.csv")
bert_embeddings = np.load("models/bert_embeddings.npy")

# Find the optimal K for KMeans clustering
def find_best_k(X):
    best_k, best_score = 0, -1
    for k in range(4, 15):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k

k = find_best_k(bert_embeddings)
kmeans = KMeans(n_clusters=k, random_state=42)
df['kmeans_cluster'] = kmeans.fit_predict(bert_embeddings)

kmeans_silhouette_score = silhouette_score(bert_embeddings, df['kmeans_cluster'])
print(f"Silhouette Score for KMeans: {kmeans_silhouette_score:.4f}")

# Perform UMAP
umap_model = UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
umap_result = umap_model.fit_transform(bert_embeddings)
print("Shape of UMAP output:", umap_result.shape)
df['umap_1'] = umap_result[:, 0]
df['umap_2'] = umap_result[:, 1]

cluster_sizes = df['kmeans_cluster'].value_counts()  # You can also use 'hdbscan_cluster'

# Create a dictionary to map cluster labels to their sizes
cluster_sizes_dict = cluster_sizes.to_dict()

# Plot the bubble chart
plt.figure(figsize=(10, 8))

# Plot each point with cluster colors and bubble size proportional to the cluster size
sns.scatterplot(
    data=df, x='umap_1', y='umap_2', hue='kmeans_cluster', size='kmeans_cluster', 
    sizes=(20, 200), palette='Set1', legend='full', alpha=0.7)

# Set plot details
plt.title('Bubble Chart of Clusters (KMeans)', fontsize=16)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

# Perform HDBSCAN
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, metric='euclidean')
df['hdbscan_cluster'] = hdbscan_model.fit_predict(umap_result)

# Calculate Silhouette Score for HDBSCAN (excluding noise points)
hdbscan_labels = df['hdbscan_cluster']
# Exclude noise points with label -1
hdbscan_labels_filtered = hdbscan_labels[hdbscan_labels != -1]
umap_result_filtered = umap_result[hdbscan_labels != -1]
hdbscan_silhouette_score = silhouette_score(umap_result_filtered, hdbscan_labels_filtered)
print(f"Silhouette Score for HDBSCAN: {hdbscan_silhouette_score:.4f}")

num_noise_points = len(df[df['hdbscan_cluster'] == -1])

# Print the number of noise points
print(f"Number of noise points: {num_noise_points}")


cluster_sizes = df['hdbscan_cluster'].value_counts()  # You can also use 'hdbscan_cluster'

# Create a dictionary to map cluster labels to their sizes
cluster_sizes_dict = cluster_sizes.to_dict()

# Plot the bubble chart
plt.figure(figsize=(10, 8))

# Plot each point with cluster colors and bubble size proportional to the cluster size
sns.scatterplot(
    data=df, x='umap_1', y='umap_2', hue='hdbscan_cluster', size='hdbscan_cluster', 
    sizes=(20, 200), palette='Set1', legend='full', alpha=0.7)

# Set plot details
plt.title('Bubble Chart of Clusters (HDBSCAN)', fontsize=16)
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')

# Show the plot
plt.show()

# TextRank ranking for each cluster (based on KMeans)
df['rank_in_cluster'] = -1
for clust in df['kmeans_cluster'].unique():
    cluster_texts = df[df['kmeans_cluster'] == clust]['clean_summary'].tolist()
    idxs = df[df['kmeans_cluster'] == clust].index.tolist()
    ranks = rank_summaries(cluster_texts)
    for rank, i in enumerate(ranks):
        df.at[idxs[i], 'rank_in_cluster'] = rank

# Save the clustered summaries
df.to_csv("data/summary_ranks.csv", index=False)
df.to_csv("data/clustered_summaries.csv", index=False)

# Optionally, print the cluster counts
print(f"KMeans clusters: {len(df['kmeans_cluster'].unique())}")
print(f"HDBSCAN clusters: {len(df['hdbscan_cluster'].unique()) - (1 if -1 in df['hdbscan_cluster'].values else 0)}")  # Exclude noise (-1)

from sklearn.metrics.pairwise import cosine_similarity 

def mean_intra_cluster_cosine_similarity(embeddings, cluster_labels):
    """
    Compute the Mean Intra-cluster Cosine Similarity for each cluster.
    :param embeddings: The embeddings of all points (e.g., BERT embeddings)
    :param cluster_labels: The cluster labels corresponding to each point
    :return: A single mean cosine similarity value across all clusters
    """
    unique_clusters = np.unique(cluster_labels)
    all_similarities = []  # To store all cosine similarities for non-noise points

    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points
            continue
        
        # Get the indices of the points in the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        
        # Get the embeddings for the points in the current cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        # Calculate the cosine similarity between all pairs in the cluster
        sim_matrix = cosine_similarity(cluster_embeddings)
        
        # Get the upper triangle of the matrix (exclude diagonal elements)
        upper_triangle_indices = np.triu_indices_from(sim_matrix, k=1)
        upper_triangle_similarities = sim_matrix[upper_triangle_indices]
        
        # Append the similarities to the list
        all_similarities.extend(upper_triangle_similarities)

    # Calculate the overall mean similarity
    mean_similarity = np.mean(all_similarities) if all_similarities else 0
    return mean_similarity

# Calculate Mean Intra-cluster Cosine Similarity for HDBSCAN (excluding noise points)
mean_hdbscan_similarity = mean_intra_cluster_cosine_similarity(bert_embeddings, df['hdbscan_cluster'])
print(f"Mean Intra-cluster Cosine Similarity for HDBSCAN Clustering: {mean_hdbscan_similarity:.4f}")