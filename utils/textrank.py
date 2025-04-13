# utils/textrank.py
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def rank_summaries(texts):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(texts)
    sim_matrix = cosine_similarity(X)
    
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in ranked]
