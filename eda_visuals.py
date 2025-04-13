# # eda_visuals.py
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics.pairwise import cosine_similarity
# from wordcloud import WordCloud
# import joblib

# df = pd.read_csv('data/cleaned_summaries.csv')
# X = joblib.load('models/tfidf_vectors.pkl')

# # Heatmap of similarity
# sim_matrix = cosine_similarity(X[:30])
# sns.heatmap(sim_matrix, cmap="Blues")
# plt.title("Similarity Between First 30 Summaries")
# plt.show()

# # WordCloud
# text = ' '.join(df['clean_summary'])
# wc = WordCloud(width=1000, height=500, background_color='white').generate(text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wc, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud of All Sessions")
# plt.show()


# eda_visuals.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import joblib
import os

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

# === 3. UMAP Scatter Plot (Colored by Cluster) ===
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='umap_1', y='umap_2',
    hue='cluster',
    palette='tab10',
    data=df,
    legend='full'
)
plt.title("UMAP Projection of Session Clusters")
plt.legend(title="Cluster")
plt.savefig("plots/umap_clusters.png")
plt.show()

# === 4. Cluster-Specific Word Clouds ===
for cluster_id in sorted(df['cluster'].unique()):
    cluster_text = ' '.join(df[df['cluster'] == cluster_id]['clean_summary'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Cluster {cluster_id}")
    plt.savefig(f"plots/wordcloud_cluster_{cluster_id}.png")
    plt.show()
