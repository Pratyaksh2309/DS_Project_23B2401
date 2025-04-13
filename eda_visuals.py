
# eda_visuals.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import joblib
import os
import networkx as nx

# Create 'plots' folder if it doesn't exist
os.makedirs("plots", exist_ok=True)

# === Load Data ===
df = pd.read_csv('data/clustered_summaries.csv')  # Use clustered version to get UMAP + cluster info
X = joblib.load('models/tfidf_vectors.pkl')

# === 1. Similarity Heatmap ===
sim_matrix = cosine_similarity(X[:30])
plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap="Blues")
plt.title("Similarity Between First 30 Summaries")
plt.savefig("plots/similarity_heatmap.png")
plt.show()

# === 2. Global Word Cloud ===
text = ' '.join(df['clean_summary'])
wc = WordCloud(width=1000, height=500, background_color='white').generate(text)
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of All Sessions")
plt.savefig("plots/global_wordcloud.png")
plt.show()

# === 3. UMAP Scatter Plot (Colored by UMAP Cluster) ===
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='umap_1', y='umap_2',
    hue='kmeans_cluster',  # UMAP colored by KMeans clusters
    palette='tab10',
    data=df,
    legend='full'
)
plt.title("UMAP Projection of Session Clusters (KMeans)")
plt.legend(title="KMeans Cluster")
plt.savefig("plots/umap_kmeans_clusters.png")
plt.show()

# === 4. UMAP Scatter Plot (Colored by HDBSCAN Cluster) ===
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='umap_1', y='umap_2',
    hue='hdbscan_cluster',  # UMAP colored by HDBSCAN clusters
    palette='tab10',
    data=df,
    legend='full'
)
plt.title("UMAP Projection of Session Clusters (HDBSCAN)")
plt.legend(title="HDBSCAN Cluster")
plt.savefig("plots/umap_hdbscan_clusters.png")
plt.show()

# === 5. Cluster-Specific Word Clouds for KMeans ===
for cluster_id in sorted(df['kmeans_cluster'].unique()):
    # Skip noise points (cluster label -1)
    if cluster_id == -1:
        continue
    cluster_text = ' '.join(df[df['kmeans_cluster'] == cluster_id]['clean_summary'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for KMeans Cluster {cluster_id}")
    plt.savefig(f"plots/wordcloud_kmeans_cluster_{cluster_id}.png")
    plt.show()

# === 6. Cluster-Specific Word Clouds for HDBSCAN ===
for cluster_id in sorted(df['hdbscan_cluster'].unique()):
    # Skip noise points (cluster label -1)
    if cluster_id == -1:
        continue
    cluster_text = ' '.join(df[df['hdbscan_cluster'] == cluster_id]['clean_summary'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for HDBSCAN Cluster {cluster_id}")
    plt.savefig(f"plots/wordcloud_hdbscan_cluster_{cluster_id}.png")
    plt.show()

# === 7. Network Graph of Similar Sessions ===
# You can choose between TF-IDF or BERT
sim_matrix = cosine_similarity(X[:30])  #
