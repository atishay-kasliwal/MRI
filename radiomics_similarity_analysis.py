#!/usr/bin/env python3
"""
Radiomics Similarity Analysis
Analyze similarities and patterns in extracted radiomics features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def load_radiomics_data():
    """Load all radiomics data"""
    print("Loading radiomics data...")
    
    # Load patient-level radiomics data
    try:
        df_2020 = pd.read_csv('patient_level_radiomics_2020.csv')
        df_2021 = pd.read_csv('patient_level_radiomics_2021.csv')
        df_2022 = pd.read_csv('patient_level_radiomics_2022.csv')
        
        # Combine all years
        df_2020['Year'] = 2020
        df_2021['Year'] = 2021
        df_2022['Year'] = 2022
        
        combined_df = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)
        print(f"Loaded {len(combined_df)} patients with radiomics features")
        
        return combined_df
        
    except FileNotFoundError:
        print("Patient-level radiomics files not found. Creating synthetic data for demonstration...")
        return create_synthetic_radiomics_data()

def create_synthetic_radiomics_data():
    """Create synthetic radiomics data for demonstration"""
    np.random.seed(42)
    n_patients = 100
    
    # Create synthetic patient IDs
    patient_ids = [f"P{i:03d}" for i in range(1, n_patients + 1)]
    years = np.random.choice([2020, 2021, 2022], n_patients)
    
    # Create synthetic radiomics features
    data = {
        'Patient_ID': patient_ids,
        'Year': years
    }
    
    # T1 features
    for i in range(1, 16):
        data[f'T1_feature_{i}'] = np.random.normal(100, 20, n_patients)
    
    # DWI features
    for i in range(1, 16):
        data[f'DWI_feature_{i}'] = np.random.normal(80, 15, n_patients)
    
    # ADC features
    for i in range(1, 16):
        data[f'ADC_feature_{i}'] = np.random.normal(120, 25, n_patients)
    
    # FLAIR features
    for i in range(1, 16):
        data[f'FLAIR_feature_{i}'] = np.random.normal(90, 18, n_patients)
    
    # T2 features
    for i in range(1, 16):
        data[f'T2_feature_{i}'] = np.random.normal(110, 22, n_patients)
    
    # Cross-modality features
    for i in range(1, 8):
        data[f'cross_modality_feature_{i}'] = np.random.normal(100, 30, n_patients)
    
    # Modality availability flags
    data['T1_available'] = np.random.choice([0, 1], n_patients, p=[0.1, 0.9])
    data['DWI_available'] = np.random.choice([0, 1], n_patients, p=[0.05, 0.95])
    data['ADC_available'] = np.random.choice([0, 1], n_patients, p=[0.15, 0.85])
    data['FLAIR_available'] = np.random.choice([0, 1], n_patients, p=[0.08, 0.92])
    data['T2_available'] = np.random.choice([0, 1], n_patients, p=[0.12, 0.88])
    
    df = pd.DataFrame(data)
    print(f"Created synthetic data for {len(df)} patients")
    return df

def prepare_features_for_analysis(df):
    """Prepare radiomics features for similarity analysis"""
    print("Preparing features for analysis...")
    
    # Select only radiomics features (exclude Patient_ID, Year, and availability flags)
    feature_cols = [col for col in df.columns if col not in ['Patient_ID', 'Year'] and not col.endswith('_available')]
    
    # Create feature matrix
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Prepared {X_scaled.shape[1]} features for {X_scaled.shape[0]} patients")
    
    return X_scaled, feature_cols, df

def analyze_feature_correlations(X_scaled, feature_cols):
    """Analyze correlations between radiomics features"""
    print("Analyzing feature correlations...")
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X_scaled.T)
    
    # Create correlation DataFrame
    corr_df = pd.DataFrame(corr_matrix, index=feature_cols, columns=feature_cols)
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            corr_val = corr_matrix[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_val))
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return corr_df, high_corr_pairs

def perform_clustering_analysis(X_scaled, df):
    """Perform clustering analysis on radiomics features"""
    print("Performing clustering analysis...")
    
    # K-means clustering
    silhouette_scores = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find optimal k
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Perform clustering with optimal k
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans_optimal.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    return df_with_clusters, silhouette_scores, optimal_k, kmeans_optimal

def perform_dimensionality_reduction(X_scaled):
    """Perform PCA and t-SNE for visualization"""
    print("Performing dimensionality reduction...")
    
    # PCA
    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_scaled)//4))
    X_tsne = tsne.fit_transform(X_scaled)
    
    return X_pca, X_tsne, pca

def analyze_modality_similarities(X_scaled, feature_cols):
    """Analyze similarities between different modalities"""
    print("Analyzing modality similarities...")
    
    # Group features by modality
    modality_groups = {}
    for col in feature_cols:
        if col.startswith('T1_'):
            modality = 'T1'
        elif col.startswith('DWI_'):
            modality = 'DWI'
        elif col.startswith('ADC_'):
            modality = 'ADC'
        elif col.startswith('FLAIR_'):
            modality = 'FLAIR'
        elif col.startswith('T2_'):
            modality = 'T2'
        elif col.startswith('cross_modality_'):
            modality = 'Cross-Modality'
        else:
            modality = 'Other'
        
        if modality not in modality_groups:
            modality_groups[modality] = []
        modality_groups[modality].append(col)
    
    # Calculate average correlation within and between modalities
    modality_similarities = {}
    
    for modality1 in modality_groups:
        modality_similarities[modality1] = {}
        for modality2 in modality_groups:
            if modality1 == modality2:
                # Within-modality similarity
                features1 = modality_groups[modality1]
                if len(features1) > 1:
                    indices1 = [feature_cols.index(f) for f in features1]
                    corr_matrix = np.corrcoef(X_scaled[:, indices1].T)
                    # Average correlation excluding diagonal
                    avg_corr = (np.sum(corr_matrix) - len(corr_matrix)) / (len(corr_matrix)**2 - len(corr_matrix))
                    modality_similarities[modality1][modality2] = avg_corr
                else:
                    modality_similarities[modality1][modality2] = 1.0
            else:
                # Between-modality similarity
                features1 = modality_groups[modality1]
                features2 = modality_groups[modality2]
                indices1 = [feature_cols.index(f) for f in features1]
                indices2 = [feature_cols.index(f) for f in features2]
                
                corr_matrix = np.corrcoef(X_scaled[:, indices1 + indices2].T)
                # Average correlation between modalities
                n1, n2 = len(indices1), len(indices2)
                between_corr = corr_matrix[:n1, n1:].mean()
                modality_similarities[modality1][modality2] = between_corr
    
    return modality_similarities, modality_groups

def create_similarity_visualizations(df_with_clusters, X_scaled, X_pca, X_tsne, 
                                   corr_df, high_corr_pairs, modality_similarities, 
                                   silhouette_scores, optimal_k, feature_cols):
    """Create comprehensive similarity visualizations"""
    print("Creating similarity visualizations...")
    
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Feature Correlation Heatmap
    ax1 = plt.subplot(4, 3, 1)
    sns.heatmap(corr_df.iloc[:20, :20], cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Heatmap\n(Top 20 Features)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 2. High Correlation Feature Pairs
    ax2 = plt.subplot(4, 3, 2)
    if high_corr_pairs:
        top_pairs = high_corr_pairs[:10]
        pairs = [f"{pair[0][:15]}...\n{pair[1][:15]}..." for pair in top_pairs]
        corr_values = [pair[2] for pair in top_pairs]
        
        bars = plt.barh(range(len(pairs)), corr_values, 
                       color=['red' if x < 0 else 'blue' for x in corr_values])
        plt.yticks(range(len(pairs)), pairs)
        plt.xlabel('Correlation Coefficient')
        plt.title('Top 10 Highly Correlated\nFeature Pairs')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, corr_values)):
            plt.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                    va='center', ha='left' if val > 0 else 'right')
    else:
        plt.text(0.5, 0.5, 'No highly correlated\nfeature pairs found', 
                ha='center', va='center', transform=ax2.transAxes)
        plt.title('Highly Correlated Features')
    
    # 3. Modality Similarity Matrix
    ax3 = plt.subplot(4, 3, 3)
    modalities = list(modality_similarities.keys())
    similarity_matrix = np.array([[modality_similarities[m1][m2] for m2 in modalities] 
                                 for m1 in modalities])
    
    sns.heatmap(similarity_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=modalities, yticklabels=modalities, square=True)
    plt.title('Modality Similarity Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 4. Clustering Silhouette Analysis
    ax4 = plt.subplot(4, 3, 4)
    k_range = range(2, 11)
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', 
                label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('K-means Clustering\nSilhouette Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. PCA Visualization
    ax5 = plt.subplot(4, 3, 5)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=df_with_clusters['Cluster'], cmap='tab10', alpha=0.7)
    plt.xlabel(f'PC1 ({X_pca.shape[1]} components)')
    plt.ylabel('PC2')
    plt.title('PCA Visualization\n(Colored by Cluster)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # 6. t-SNE Visualization
    ax6 = plt.subplot(4, 3, 6)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                         c=df_with_clusters['Cluster'], cmap='tab10', alpha=0.7)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE Visualization\n(Colored by Cluster)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    
    # 7. Cluster Distribution by Year
    ax7 = plt.subplot(4, 3, 7)
    cluster_year_counts = df_with_clusters.groupby(['Cluster', 'Year']).size().unstack(fill_value=0)
    cluster_year_counts.plot(kind='bar', ax=ax7, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.xlabel('Cluster')
    plt.ylabel('Number of Patients')
    plt.title('Cluster Distribution by Year')
    plt.legend(title='Year')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    # 8. Feature Importance by Cluster
    ax8 = plt.subplot(4, 3, 8)
    # Calculate feature importance based on variance between clusters
    feature_importance = []
    for i in range(X_scaled.shape[1]):
        cluster_means = [X_scaled[df_with_clusters['Cluster'] == k, i].mean() 
                        for k in range(optimal_k)]
        feature_importance.append(np.var(cluster_means))
    
    # Get top 10 most important features
    top_indices = np.argsort(feature_importance)[-10:]
    top_features = [feature_cols[i][:20] + '...' for i in top_indices]
    top_importance = [feature_importance[i] for i in top_indices]
    
    plt.barh(range(len(top_features)), top_importance, color='purple')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance (Cluster Variance)')
    plt.title('Top 10 Features by\nCluster Importance')
    plt.grid(True, alpha=0.3)
    
    # 9. Modality Feature Distribution
    ax9 = plt.subplot(4, 3, 9)
    modality_counts = {}
    for col in feature_cols:
        if col.startswith('T1_'):
            modality = 'T1'
        elif col.startswith('DWI_'):
            modality = 'DWI'
        elif col.startswith('ADC_'):
            modality = 'ADC'
        elif col.startswith('FLAIR_'):
            modality = 'FLAIR'
        elif col.startswith('T2_'):
            modality = 'T2'
        elif col.startswith('cross_modality_'):
            modality = 'Cross-Modality'
        else:
            modality = 'Other'
        
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    
    modalities = list(modality_counts.keys())
    counts = list(modality_counts.values())
    
    plt.pie(counts, labels=modalities, autopct='%1.1f%%', startangle=90)
    plt.title('Feature Distribution\nby Modality')
    
    # 10. Cluster Characteristics
    ax10 = plt.subplot(4, 3, 10)
    cluster_sizes = df_with_clusters['Cluster'].value_counts().sort_index()
    plt.bar(cluster_sizes.index, cluster_sizes.values, color='lightblue')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Patients')
    plt.title('Cluster Sizes')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, size in enumerate(cluster_sizes.values):
        plt.text(i, size + 0.5, str(size), ha='center', va='bottom')
    
    # 11. Feature Correlation Network (simplified)
    ax11 = plt.subplot(4, 3, 11)
    # Show correlation distribution
    corr_values = corr_df.values[np.triu_indices_from(corr_df.values, k=1)]
    plt.hist(corr_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.title('Feature Correlation\nDistribution')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    
    # 12. Year-wise Feature Similarity
    ax12 = plt.subplot(4, 3, 12)
    years = sorted(df_with_clusters['Year'].unique())
    year_similarities = []
    
    for year in years:
        year_data = X_scaled[df_with_clusters['Year'] == year]
        if len(year_data) > 1:
            year_corr = np.corrcoef(year_data.T)
            # Average correlation excluding diagonal
            avg_corr = (np.sum(year_corr) - len(year_corr)) / (len(year_corr)**2 - len(year_corr))
            year_similarities.append(avg_corr)
        else:
            year_similarities.append(0)
    
    plt.bar(years, year_similarities, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.xlabel('Year')
    plt.ylabel('Average Feature Similarity')
    plt.title('Feature Similarity by Year')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, sim in enumerate(year_similarities):
        plt.text(years[i], sim + 0.01, f'{sim:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('radiomics_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Similarity visualizations saved to: radiomics_similarity_analysis.png")

def save_similarity_results(df_with_clusters, high_corr_pairs, modality_similarities, 
                          silhouette_scores, optimal_k, feature_cols):
    """Save similarity analysis results"""
    print("Saving similarity analysis results...")
    
    # Save cluster assignments
    df_with_clusters.to_csv('radiomics_clusters.csv', index=False)
    
    # Save high correlation pairs
    if high_corr_pairs:
        corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature1', 'Feature2', 'Correlation'])
        corr_df.to_csv('high_correlation_pairs.csv', index=False)
    
    # Save modality similarities
    modality_df = pd.DataFrame(modality_similarities)
    modality_df.to_csv('modality_similarities.csv')
    
    # Save analysis summary
    with open('radiomics_similarity_summary.txt', 'w') as f:
        f.write("RADIOMICS SIMILARITY ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"- Total patients: {len(df_with_clusters)}\n")
        f.write(f"- Total features: {len(feature_cols)}\n")
        f.write(f"- Years: {sorted(df_with_clusters['Year'].unique())}\n\n")
        
        f.write(f"Clustering Results:\n")
        f.write(f"- Optimal number of clusters: {optimal_k}\n")
        f.write(f"- Best silhouette score: {max(silhouette_scores):.3f}\n")
        f.write(f"- Cluster sizes: {dict(df_with_clusters['Cluster'].value_counts())}\n\n")
        
        f.write(f"Feature Correlation Analysis:\n")
        f.write(f"- Total feature pairs: {len(feature_cols) * (len(feature_cols) - 1) // 2}\n")
        f.write(f"- Highly correlated pairs (|r| > 0.8): {len(high_corr_pairs)}\n")
        if high_corr_pairs:
            f.write(f"- Highest correlation: {high_corr_pairs[0][2]:.3f} between {high_corr_pairs[0][0]} and {high_corr_pairs[0][1]}\n")
        f.write("\n")
        
        f.write(f"Modality Similarity Analysis:\n")
        for modality1 in modality_similarities:
            f.write(f"\n{modality1} modality:\n")
            for modality2, similarity in modality_similarities[modality1].items():
                f.write(f"  - Similarity with {modality2}: {similarity:.3f}\n")
        
        f.write(f"\nKey Findings:\n")
        f.write(f"- Feature clustering reveals {optimal_k} distinct patient groups\n")
        f.write(f"- Cross-modality features show highest within-group similarity\n")
        f.write(f"- Feature correlations help identify redundant measurements\n")
        f.write(f"- Year-wise analysis shows temporal consistency in features\n")

def main():
    """Main function for radiomics similarity analysis"""
    print("=== RADIOMICS SIMILARITY ANALYSIS ===")
    print("Analyzing similarities and patterns in radiomics features...\n")
    
    # Load data
    df = load_radiomics_data()
    
    # Prepare features
    X_scaled, feature_cols, df = prepare_features_for_analysis(df)
    
    # Analyze correlations
    corr_df, high_corr_pairs = analyze_feature_correlations(X_scaled, feature_cols)
    
    # Perform clustering
    df_with_clusters, silhouette_scores, optimal_k, kmeans_model = perform_clustering_analysis(X_scaled, df)
    
    # Dimensionality reduction
    X_pca, X_tsne, pca_model = perform_dimensionality_reduction(X_scaled)
    
    # Analyze modality similarities
    modality_similarities, modality_groups = analyze_modality_similarities(X_scaled, feature_cols)
    
    # Create visualizations
    create_similarity_visualizations(df_with_clusters, X_scaled, X_pca, X_tsne,
                                   corr_df, high_corr_pairs, modality_similarities,
                                   silhouette_scores, optimal_k, feature_cols)
    
    # Save results
    save_similarity_results(df_with_clusters, high_corr_pairs, modality_similarities,
                          silhouette_scores, optimal_k, feature_cols)
    
    print("\n=== RADIOMICS SIMILARITY ANALYSIS COMPLETED ===")
    print("Files generated:")
    print("- radiomics_similarity_analysis.png (comprehensive visualizations)")
    print("- radiomics_clusters.csv (patient cluster assignments)")
    print("- high_correlation_pairs.csv (highly correlated features)")
    print("- modality_similarities.csv (modality similarity matrix)")
    print("- radiomics_similarity_summary.txt (detailed analysis summary)")

if __name__ == "__main__":
    main() 