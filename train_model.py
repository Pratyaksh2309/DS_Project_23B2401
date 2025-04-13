# # train_model.py
# import pandas as pd
# import re
# import nltk
# import string
# import joblib

# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load data
# df = pd.read_excel("Session-Summary-for-E6-project (1).xlsx")

# # Preprocess
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     tokens = nltk.word_tokenize(text)
#     tokens = [w for w in tokens if w not in stopwords.words('english')]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(w) for w in tokens]
#     return ' '.join(tokens)

# df['clean_summary'] = df['Session_Summary'].apply(clean_text)

# # TF-IDF
# vectorizer = TfidfVectorizer(max_features=2000)
# X = vectorizer.fit_transform(df['clean_summary'])

# # Save models and data
# joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
# joblib.dump(X, 'models/tfidf_vectors.pkl')
# df.to_csv('data/cleaned_summaries.csv', index=False)

# print("✅ Training complete. Models and cleaned data saved.")

# from sentence_transformers import SentenceTransformer
# import numpy as np

# # BERT model
# bert = SentenceTransformer('all-MiniLM-L6-v2')
# bert_embeddings = bert.encode(df['clean_summary'].tolist(), show_progress_bar=True)

# # Save BERT embeddings
# np.save("models/bert_embeddings.npy", bert_embeddings)


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
