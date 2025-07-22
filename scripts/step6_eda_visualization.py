import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import re

MERGED_CSV = 'data/radiomics/merged_radiomics_clinical.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def sanitize_filename(s):
    return re.sub(r'[^A-Za-z0-9_]+', '_', s)

# Load data
data = pd.read_csv(MERGED_CSV)

# 1. Plot distributions of key clinical features
clinical_features = ['Age', 'Sex', 'Baseline mRS', 'ADMIT NIH', 'Days f/u', 'Last mRS']
for feat in clinical_features:
    if feat in data.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(data[feat].dropna(), kde=True, bins=20)
        plt.title(f'Distribution of {feat}')
        plt.tight_layout()
        safe_feat = sanitize_filename(feat)
        plt.savefig(os.path.join(RESULTS_DIR, f'step6_dist_{safe_feat}.png'))
        plt.close()

# 2. Plot distributions of a few radiomics features
radiomics_features = [col for col in data.columns if col.startswith('original_')][:6]
for feat in radiomics_features:
    plt.figure(figsize=(6,4))
    sns.histplot(data[feat].dropna(), kde=True, bins=20)
    plt.title(f'Distribution of {feat}')
    plt.tight_layout()
    safe_feat = sanitize_filename(feat)
    plt.savefig(os.path.join(RESULTS_DIR, f'step6_dist_{safe_feat}.png'))
    plt.close()

# 3. Correlation heatmap (numeric features only)
numeric_data = data.select_dtypes(include=[np.number])
plt.figure(figsize=(12,10))
corr = numeric_data.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap (numeric features)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step6_corr_heatmap.png'))
plt.close()

# 4. PCA and t-SNE colored by Last mRS
features = [col for col in numeric_data.columns if col not in ['Last mRS']]
X = numeric_data[features]
X = X.fillna(X.mean())  # Fill missing values with mean
X = X.dropna(axis=1, how='all')  # Drop columns that are all NaN
X = X.dropna(axis=0, how='any')  # Drop rows with any remaining NaN
y = numeric_data['Last mRS']

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plt.figure(figsize=(7,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Last mRS')
plt.title('PCA of Features colored by Last mRS')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step6_pca_lastmrs.png'))
plt.close()

# t-SNE (may take a minute)
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
X_tsne = tsne.fit_transform(X)
plt.figure(figsize=(7,6))
scatter = plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Last mRS')
plt.title('t-SNE of Features colored by Last mRS')
plt.xlabel('tSNE-1')
plt.ylabel('tSNE-2')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step6_tsne_lastmrs.png'))
plt.close()

print("Step 6 complete: EDA and visualizations saved to results/.") 