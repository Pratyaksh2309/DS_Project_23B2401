# # cluster_and_rank.py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import umap.umap_ as umap
# import seaborn as sns

# bert_embeddings = np.load('models/bert_embeddings.npy')
# df = pd.read_csv('data/cleaned_summaries.csv')

# # Find best cluster count
# def optimal_k(X):
#     best_score = -1
#     best_k = 0
#     for k in range(4, 15):
#         km = KMeans(n_clusters=k, random_state=42)
#         labels = km.fit_predict(X)
#         score = silhouette_score(X, labels)
#         print(f"K={k}, Silhouette={score:.4f}")
#         if score > best_score:
#             best_k = k
#     return best_k

# k = optimal_k(bert_embeddings)
# kmeans = KMeans(n_clusters=k, random_state=42)
# df['cluster'] = kmeans.fit_predict(bert_embeddings)

# # UMAP visualization
# umap_model = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
# umap_result = umap_model.fit_transform(bert_embeddings)

# plt.figure(figsize=(10, 7))
# sns.scatterplot(x=umap_result[:, 0], y=umap_result[:, 1], hue=df['cluster'], palette='tab10')
# plt.title("UMAP Clusters of Session Summaries")
# plt.savefig("data/umap_clusters.png")
# plt.show()

# df.to_csv('data/clustered_summaries.csv', index=False)

# from utils.textrank import rank_summaries

# # Rank summaries within each cluster
# all_ranks = []
# for clust in df['cluster'].unique():
#     texts = df[df['cluster'] == clust]['clean_summary'].tolist()
#     indices = df[df['cluster'] == clust].index.tolist()
    
#     ranked_ids = rank_summaries(texts)
#     for rank, i in enumerate(ranked_ids):
#         all_ranks.append((indices[i], rank))

# # Save ranking to DataFrame
# df['rank_in_cluster'] = -1
# for idx, rank in all_ranks:
#     df.at[idx, 'rank_in_cluster'] = rank

# df.to_csv('data/summary_ranks.csv', index=False)

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


df = pd.read_csv("data/cleaned_summaries.csv")
bert_embeddings = np.load("models/bert_embeddings.npy")

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
df['cluster'] = kmeans.fit_predict(bert_embeddings)

# UMAP
umap_model = UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
umap_result = umap_model.fit_transform(bert_embeddings)
df['umap_1'] = umap_result[:, 0]
df['umap_2'] = umap_result[:, 1]

# TextRank ranking
df['rank_in_cluster'] = -1
for clust in df['cluster'].unique():
    cluster_texts = df[df['cluster'] == clust]['clean_summary'].tolist()
    idxs = df[df['cluster'] == clust].index.tolist()
    ranks = rank_summaries(cluster_texts)
    for rank, i in enumerate(ranks):
        df.at[idxs[i], 'rank_in_cluster'] = rank

df.to_csv("data/summary_ranks.csv", index=False)
df.to_csv("data/clustered_summaries.csv", index=False)
