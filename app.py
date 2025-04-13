import streamlit as st

# 🛠 Must be first Streamlit command
st.set_page_config(page_title="Session Summary Engine", layout="wide")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import io
from PIL import Image

# ---------- CACHE MODEL ---------- 
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

bert_model = load_bert_model()

# ---------- LOAD DATA ---------- 
df = pd.read_csv("data/summary_ranks.csv")
bert_embeddings = np.load("models/bert_embeddings.npy")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ---------- PAGE SETUP ---------- 
st.title("📘 Session Summary Explorer")
st.write("Explore and search through summarized sessions using clustering and semantic similarity.")

# ---------- 1. SEARCH BY KEYWORDS ---------- 
st.header("🔍 Search by Keywords")
query = st.text_input("Enter topic keywords (comma separated):")

if st.button("Find Best Session"):
    if query.strip():
        query_embed = bert_model.encode([query])
        sims = cosine_similarity(query_embed, bert_embeddings).flatten()
        best_idx = sims.argmax()
        best_cluster = df.loc[best_idx]['hdbscan_cluster']
        top3 = df[df['hdbscan_cluster'] == best_cluster].sort_values('rank_in_cluster').head(3)
        st.success(f"Best-matching cluster: {best_cluster}")
        for i, row in top3.iterrows():
            st.markdown(f"### Summary {i+1}")
            st.text_area("", row['Session_Summary'], height=200)
    else:
        st.warning("Please enter keywords.")

# ---------- 2. SESSION WORDCLOUD ---------- 
st.header("☁️ Session-wise Word Cloud")
session_idx = st.slider("Select a session index:", min_value=0, max_value=len(df)-1, step=1)
text = df.iloc[session_idx]['clean_summary']
wc = WordCloud(width=800, height=400, background_color='white').generate(text)
fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
st.pyplot(fig_wc)

# Optional download
buf = io.BytesIO()
fig_wc.savefig(buf, format='png')
st.download_button("Download Session WordCloud", buf.getvalue(), file_name=f"session_{session_idx}_wordcloud.png")

# ---------- 3. UMAP VISUALIZATION ---------- 
st.header("🗺️ UMAP Clustering")
try:
    umap_df = pd.read_csv("data/clustered_summaries.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Option to choose between UMAP clustering and HDBSCAN clustering visualization
    clustering_method = st.selectbox("Choose clustering method", ['KMeans (UMAP)', 'HDBSCAN'])
    
    if clustering_method == 'KMeans (UMAP)':
        sns.scatterplot(data=umap_df, x='umap_1', y='umap_2', hue='kmeans_cluster', palette='tab10', legend='full')
        st.pyplot(fig)
    elif clustering_method == 'HDBSCAN':
        sns.scatterplot(data=umap_df, x='umap_1', y='umap_2', hue='hdbscan_cluster', palette='tab10', legend='full')
        st.pyplot(fig)

except:
    st.warning("UMAP visualization unavailable.")

# ---------- 4. CLUSTER WORDCLOUD ---------- 
st.header("🎯 Cluster Word Cloud")
selected_cluster = st.selectbox("Choose a cluster to visualize:", sorted(df['hdbscan_cluster'].unique()))
cluster_text = ' '.join(df[df['hdbscan_cluster'] == selected_cluster]['clean_summary'])
wc_cluster = WordCloud(width=1000, height=400, background_color='white').generate(cluster_text)

fig_cwc, ax_cwc = plt.subplots(figsize=(12, 6))
plt.imshow(wc_cluster, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig_cwc)

# Optional download
buf_cluster = io.BytesIO()
fig_cwc.savefig(buf_cluster, format='png')
st.download_button("Download Cluster WordCloud", buf_cluster.getvalue(), file_name=f"cluster_{selected_cluster}_wordcloud.png")

# ---------- 5. TOP SUMMARIES PER CLUSTER ---------- 
st.header("📌 Top Ranked Summaries in Cluster")
top_k = st.slider("Top K summaries", 1, 5, 3)
top_cluster_df = df[df['hdbscan_cluster'] == selected_cluster].sort_values('rank_in_cluster').head(top_k)
for i, row in top_cluster_df.iterrows():
    st.markdown(f"### Rank {int(row['rank_in_cluster']) + 1}")
    st.text_area("", row['Session_Summary'], height=200)

# ---------- 6. CUSTOM TEXT MATCHING ---------- 
st.header("📝 Find Similar Session for Your Own Text")
user_text = st.text_area("Paste your own content here:")
if st.button("Match Custom Text"):
    if user_text.strip():
        user_vec = bert_model.encode([user_text])
        sims = cosine_similarity(user_vec, bert_embeddings).flatten()
        top3 = sims.argsort()[-3:][::-1]
        for i, idx in enumerate(top3):
            st.markdown(f"### Match {i+1}")
            st.text_area("", df.iloc[idx]['Session_Summary'], height=200)
    else:
        st.warning("Please enter custom text.")
