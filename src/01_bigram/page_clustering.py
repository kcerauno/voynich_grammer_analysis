import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math
from collections import Counter, defaultdict
from pathlib import Path
import os
import argparse

# Create results directory if it doesn't exist
os.makedirs('hypothesis/01_bigram/results', exist_ok=True)

# Load data from SQLite
conn = sqlite3.connect('data/voynich.db')
query = """
SELECT page, category, scribe, language, word 
FROM words_enriched 
WHERE word IS NOT NULL AND word != ''
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Group by page
page_data = df.groupby(['page', 'category', 'scribe', 'language'])['word'].apply(list).reset_index()

# Filter pages with fewer than min_words words to avoid excessive noise
min_words = 20
page_data = page_data[page_data['word'].apply(len) >= min_words].reset_index(drop=True)

print(f"Total pages with >= {min_words} words: {len(page_data)}")
print("\nCategories distribution:")
print(page_data['category'].value_counts())

# Build global vocab
all_words = df['word'].tolist()
global_charset = set(c for w in all_words for c in w)
vocab = sorted(list(global_charset))

def compute_features(words, vocab, alpha=0.01):
    V = len(vocab)
    raw = defaultdict(Counter)
    bos_raw = Counter()
    
    for w in words:
        if not w: continue
        seq = ['^'] + list(w) + ['$']
        bos_raw[seq[1]] += 1
        for a, b in zip(seq, seq[1:]):
            raw[a][b] += 1

    features = []
    
    # 1. BOS distribution (probability of each char being first)
    bos_total = sum(bos_raw.values()) + alpha * V
    for c in vocab:
        prob = (bos_raw.get(c, 0) + alpha) / bos_total
        features.append(prob)
        
    # 2. Transition matrix (flattened)
    contexts = ['^'] + vocab
    vocab_with_eos = vocab + ['$']
    V_next = len(vocab_with_eos)
    
    for ctx in contexts:
        ctx_total = sum(raw[ctx].values()) + alpha * V_next
        for c in vocab_with_eos:
            prob = (raw[ctx].get(c, 0) + alpha) / ctx_total
            features.append(prob)
            
    return np.array(features)

# Extract features
print("\nComputing features...")
X = np.array([compute_features(words, vocab) for words in page_data['word']])
cat_labels = page_data['category'].values
scribe_labels = page_data['scribe'].astype(str).values
lang_labels = page_data['language'].astype(str).values
pages = page_data['page'].values

print(f"Feature matrix shape: {X.shape}")

# Optional: standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction
print("\nPerforming PCA...")
# Reduce to 50 dimensions first (or keep all if n_samples < 50)
n_components = min(50, X.shape[0], X.shape[1])
pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_):.3f}")

print("\nPerforming clustering visualization...")
# Try UMAP, fallback to TSNE
try:
    import umap
    print("Using UMAP")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    method_name = "UMAP"
except ImportError:
    print("UMAP not found, falling back to t-SNE")
    # Perplexity should be smaller than number of samples
    perplexity = min(30, max(5, int(X.shape[0] / 5)))
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    method_name = "t-SNE"

X_2d = reducer.fit_transform(X_pca)

def plot_clusters(labels, title, filename):
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        x=X_2d[:, 0], y=X_2d[:, 1],
        hue=labels,
        palette="tab10",
        s=60, alpha=0.8
    )

    # Annotate some points (e.g. random 10%)
    np.random.seed(42)
    annotate_idx = np.random.choice(len(pages), size=min(40, len(pages)), replace=False)
    for idx in annotate_idx:
        plt.annotate(pages[idx], (X_2d[idx, 0], X_2d[idx, 1]), 
                     fontsize=8, alpha=0.6, xytext=(3, 3), textcoords='offset points')

    plt.title(f"Page-level Bigram Feature Clustering - {title} ({method_name})")
    plt.xlabel(f"{method_name} Component 1")
    plt.ylabel(f"{method_name} Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=title)
    plt.tight_layout()

    out_path = f'hypothesis/01_bigram/results/{filename}'
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.close()

plot_clusters(cat_labels, "Category", "page_clusters_category.png")
plot_clusters(scribe_labels, "Scribe", "page_clusters_scribe.png")
plot_clusters(lang_labels, "Currier Language", "page_clusters_language.png")

