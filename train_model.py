# train_model.py
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_excel("Session-Summary-for-E6-project.xlsx")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_summary'] = df['Session_Summary'].apply(clean_text)
df.to_csv("data/cleaned_summaries.csv", index=False)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_summary'])
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# BERT
model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings = model.encode(df['clean_summary'].tolist(), show_progress_bar=True)
np.save("models/bert_embeddings.npy", bert_embeddings)
print("Shape of BERT embeddings:", bert_embeddings.shape)