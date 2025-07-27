#!/usr/bin/env python3
"""
Comprehensive Radiomics Analysis for MRI
Top 10 Most Applied Radiomics Analysis Techniques with Appealing Visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set enhanced style for better aesthetics
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveRadiomicsAnalyzer:
    """Comprehensive Radiomics Analysis with Top 10 Techniques"""
    
    def __init__(self):
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_enhanced_synthetic_data(self):
        """Load enhanced synthetic radiomics data with realistic patterns"""
        print("🔬 Loading enhanced synthetic radiomics data...")
        
        np.random.seed(42)
        n_patients = 200
        
        # Create synthetic patient IDs and metadata
        patient_ids = [f"P{i:03d}" for i in range(1, n_patients + 1)]
        ages = np.random.normal(65, 15, n_patients)
        sexes = np.random.choice(['M', 'F'], n_patients, p=[0.55, 0.45])
        years = np.random.choice([2020, 2021, 2022], n_patients, p=[0.3, 0.4, 0.3])
        
        # Create base data structure
        data = {
            'Patient_ID': patient_ids,
            'Age': ages,
            'Sex': sexes,
            'Year': years
        }
        
        # Generate realistic radiomics features with correlations
        modalities = {
            'T1': {'base': 100, 'std': 20, 'features': 20},
            'T2': {'base': 120, 'std': 25, 'features': 20},
            'FLAIR': {'base': 90, 'std': 18, 'features': 20},
            'DWI': {'base': 80, 'std': 15, 'features': 20},
            'ADC': {'base': 140, 'std': 30, 'features': 20}
        }
        
        # Create correlated features within each modality
        for modality, params in modalities.items():
            base_val = params['base']
            std_val = params['std']
            n_features = params['features']
            
            # Create base feature with age correlation
            age_factor = np.random.normal(0, 0.3, n_patients)
            
            for i in range(1, n_features + 1):
                feature_name = f'{modality}_feature_{i}'
                
                if i == 1:
                    # Base feature with age correlation
                    data[feature_name] = base_val + np.random.normal(0, std_val, n_patients) + age_factor * 10
                else:
                    # Correlate with previous feature
                    correlation = 0.6 + np.random.uniform(0, 0.3)
                    data[feature_name] = (correlation * data[f'{modality}_feature_{i-1}'] + 
                                        (1-correlation) * (base_val + np.random.normal(0, std_val, n_patients)))
        
        # Add cross-modality features
        for i in range(1, 11):
            data[f'cross_modality_feature_{i}'] = (
                data[f'T1_feature_{i}'] * 0.25 + 
                data[f'T2_feature_{i}'] * 0.25 + 
                data[f'FLAIR_feature_{i}'] * 0.25 + 
                data[f'DWI_feature_{i}'] * 0.25 + 
                np.random.normal(0, 5, n_patients)
            )
        
        # Add clinical outcomes (synthetic)
        data['Lesion_Volume'] = np.random.gamma(2, 50, n_patients)
        data['Edema_Score'] = np.random.choice([0, 1, 2, 3], n_patients, p=[0.3, 0.4, 0.2, 0.1])
        data['Hemorrhage'] = np.random.choice([0, 1], n_patients, p=[0.8, 0.2])
        
        self.data = pd.DataFrame(data)
        print(f"✅ Loaded {len(self.data)} patients with {len(self.data.columns)-7} radiomics features")
        return self.data
    
    def prepare_features(self):
        """Prepare radiomics features for analysis"""
        print("🔧 Preparing features for analysis...")
        
        # Select only radiomics features
        feature_cols = [col for col in self.data.columns if 'feature' in col]
        self.features = self.data[feature_cols].copy()
        
        # Handle missing values
        self.features = self.features.fillna(self.features.mean())
        
        # Standardize features
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.features),
            columns=self.features.columns,
            index=self.features.index
        )
        
        print(f"✅ Prepared {self.features_scaled.shape[1]} features for {self.features_scaled.shape[0]} patients")
        return self.features_scaled
    
    def technique_1_feature_correlation_analysis(self):
        """Technique 1: Comprehensive Feature Correlation Analysis"""
        print("📊 Technique 1: Feature Correlation Analysis")
        
        # Calculate correlation matrix
        corr_matrix = self.features_scaled.corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        self.results['correlation'] = {
            'matrix': corr_matrix,
            'high_pairs': high_corr_pairs,
            'avg_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        }
        
        return self.results['correlation']
    
    def technique_2_clustering_analysis(self):
        """Technique 2: Multi-Algorithm Clustering Analysis"""
        print("🎯 Technique 2: Multi-Algorithm Clustering Analysis")
        
        # K-means clustering
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.features_scaled)
            silhouette_avg = silhouette_score(self.features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Perform clustering with optimal k
        kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_optimal.fit_predict(self.features_scaled)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(self.features_scaled)
        
        # Hierarchical clustering
        linkage_matrix = linkage(self.features_scaled, method='ward')
        hierarchical_labels = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
        
        self.results['clustering'] = {
            'kmeans': {'labels': kmeans_labels, 'optimal_k': optimal_k, 'silhouette_scores': silhouette_scores},
            'dbscan': {'labels': dbscan_labels, 'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)},
            'hierarchical': {'labels': hierarchical_labels, 'linkage_matrix': linkage_matrix}
        }
        
        return self.results['clustering']
    
    def technique_3_dimensionality_reduction(self):
        """Technique 3: Multi-Method Dimensionality Reduction"""
        print("📉 Technique 3: Multi-Method Dimensionality Reduction")
        
        # PCA
        pca = PCA(n_components=0.95)
        pca_result = pca.fit_transform(self.features_scaled)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.features_scaled)//4))
        tsne_result = tsne.fit_transform(self.features_scaled)
        
        # UMAP
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        umap_result = umap_reducer.fit_transform(self.features_scaled)
        
        # NMF (requires non-negative data)
        features_minmax = MinMaxScaler().fit_transform(self.features_scaled)
        nmf = NMF(n_components=min(10, self.features_scaled.shape[1]), random_state=42)
        nmf_result = nmf.fit_transform(features_minmax)
        
        self.results['dimensionality'] = {
            'pca': {'result': pca_result, 'explained_variance': pca.explained_variance_ratio_},
            'tsne': {'result': tsne_result},
            'umap': {'result': umap_result},
            'nmf': {'result': nmf_result, 'components': nmf.components_}
        }
        
        return self.results['dimensionality']
    
    def technique_4_feature_importance_analysis(self):
        """Technique 4: Multi-Method Feature Importance Analysis"""
        print("⭐ Technique 4: Multi-Method Feature Importance Analysis")
        
        # Create synthetic target for demonstration
        target = np.random.choice([0, 1], len(self.features_scaled), p=[0.6, 0.4])
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.features_scaled, target)
        rf_importance = rf.feature_importances_
        
        # Statistical tests
        f_scores, f_pvalues = f_classif(self.features_scaled, target)
        mi_scores = mutual_info_classif(self.features_scaled, target, random_state=42)
        
        # Variance-based importance
        variance_importance = self.features_scaled.var()
        
        # Correlation with target
        target_corr = [abs(np.corrcoef(self.features_scaled[col], target)[0, 1]) 
                      for col in self.features_scaled.columns]
        
        self.results['feature_importance'] = {
            'random_forest': rf_importance,
            'f_scores': f_scores,
            'mutual_info': mi_scores,
            'variance': variance_importance,
            'target_correlation': target_corr,
            'feature_names': self.features_scaled.columns.tolist()
        }
        
        return self.results['feature_importance']
    
    def technique_5_outlier_detection(self):
        """Technique 5: Multi-Method Outlier Detection"""
        print("🔍 Technique 5: Multi-Method Outlier Detection")
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest_labels = iso_forest.fit_predict(self.features_scaled)
        
        # One-Class SVM
        oc_svm = OneClassSVM(nu=0.1)
        oc_svm_labels = oc_svm.fit_predict(self.features_scaled)
        
        # Statistical outlier detection (Z-score)
        z_scores = np.abs(stats.zscore(self.features_scaled))
        z_score_outliers = (z_scores > 3).any(axis=1)
        
        # IQR-based outlier detection
        Q1 = self.features_scaled.quantile(0.25)
        Q3 = self.features_scaled.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((self.features_scaled < (Q1 - 1.5 * IQR)) | 
                       (self.features_scaled > (Q3 + 1.5 * IQR))).any(axis=1)
        
        self.results['outlier_detection'] = {
            'isolation_forest': iso_forest_labels,
            'one_class_svm': oc_svm_labels,
            'z_score': z_score_outliers,
            'iqr': iqr_outliers
        }
        
        return self.results['outlier_detection']
    
    def technique_6_modality_analysis(self):
        """Technique 6: Comprehensive Modality Analysis"""
        print("🏥 Technique 6: Comprehensive Modality Analysis")
        
        # Group features by modality
        modality_groups = {}
        for col in self.features_scaled.columns:
            if col.startswith('T1_'):
                modality = 'T1'
            elif col.startswith('T2_'):
                modality = 'T2'
            elif col.startswith('FLAIR_'):
                modality = 'FLAIR'
            elif col.startswith('DWI_'):
                modality = 'DWI'
            elif col.startswith('ADC_'):
                modality = 'ADC'
            elif col.startswith('cross_modality_'):
                modality = 'Cross-Modality'
            else:
                modality = 'Other'
            
            if modality not in modality_groups:
                modality_groups[modality] = []
            modality_groups[modality].append(col)
        
        # Calculate modality statistics
        modality_stats = {}
        for modality, features in modality_groups.items():
            modality_data = self.features_scaled[features]
            modality_stats[modality] = {
                'n_features': len(features),
                'mean_correlation': modality_data.corr().values[np.triu_indices_from(modality_data.corr().values, k=1)].mean(),
                'variance': modality_data.var().mean(),
                'skewness': modality_data.skew().mean(),
                'kurtosis': modality_data.kurtosis().mean()
            }
        
        self.results['modality_analysis'] = {
            'groups': modality_groups,
            'statistics': modality_stats
        }
        
        return self.results['modality_analysis']
    
    def technique_7_temporal_analysis(self):
        """Technique 7: Temporal and Longitudinal Analysis"""
        print("📅 Technique 7: Temporal and Longitudinal Analysis")
        
        # Year-wise analysis
        year_stats = {}
        for year in sorted(self.data['Year'].unique()):
            year_data = self.data[self.data['Year'] == year]
            year_features = self.features_scaled.loc[year_data.index]
            
            year_stats[year] = {
                'n_patients': len(year_data),
                'mean_features': year_features.mean().mean(),
                'std_features': year_features.std().mean(),
                'feature_correlation': year_features.corr().values[np.triu_indices_from(year_features.corr().values, k=1)].mean()
            }
        
        # Age-based analysis
        age_bins = pd.cut(self.data['Age'], bins=5)
        age_stats = {}
        for age_bin in age_bins.unique():
            if pd.isna(age_bin):
                continue
            age_data = self.data[age_bins == age_bin]
            age_features = self.features_scaled.loc[age_data.index]
            
            age_stats[str(age_bin)] = {
                'n_patients': len(age_data),
                'mean_features': age_features.mean().mean(),
                'std_features': age_features.std().mean()
            }
        
        self.results['temporal_analysis'] = {
            'year_stats': year_stats,
            'age_stats': age_stats
        }
        
        return self.results['temporal_analysis']
    
    def technique_8_statistical_analysis(self):
        """Technique 8: Comprehensive Statistical Analysis"""
        print("📈 Technique 8: Comprehensive Statistical Analysis")
        
        # Distribution analysis
        distribution_stats = {}
        for col in self.features_scaled.columns:
            feature_data = self.features_scaled[col]
            distribution_stats[col] = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'skewness': feature_data.skew(),
                'kurtosis': feature_data.kurtosis(),
                'shapiro_p': stats.shapiro(feature_data)[1],
                'anderson_stat': stats.anderson(feature_data)[0]
            }
        
        # Correlation with clinical variables
        clinical_correlations = {}
        for col in self.features_scaled.columns:
            feature_data = self.features_scaled[col]
            clinical_correlations[col] = {
                'age_corr': np.corrcoef(feature_data, self.data['Age'])[0, 1],
                'volume_corr': np.corrcoef(feature_data, self.data['Lesion_Volume'])[0, 1]
            }
        
        # Feature stability analysis
        feature_stability = {}
        for col in self.features_scaled.columns:
            feature_data = self.features_scaled[col]
            # Calculate stability as inverse of coefficient of variation
            feature_stability[col] = feature_data.mean() / feature_data.std() if feature_data.std() > 0 else 0
        
        self.results['statistical_analysis'] = {
            'distribution': distribution_stats,
            'clinical_correlations': clinical_correlations,
            'stability': feature_stability
        }
        
        return self.results['statistical_analysis']
    
    def technique_9_network_analysis(self):
        """Technique 9: Network and Graph Analysis"""
        print("🕸️ Technique 9: Network and Graph Analysis")
        
        # Create correlation network
        corr_matrix = self.features_scaled.corr()
        
        # Find network connections (high correlations)
        network_edges = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    network_edges.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        # Calculate network metrics
        network_metrics = {
            'n_nodes': len(corr_matrix.columns),
            'n_edges': len(network_edges),
            'density': len(network_edges) / (len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) / 2),
            'avg_degree': 2 * len(network_edges) / len(corr_matrix.columns)
        }
        
        # Community detection (simplified)
        communities = {}
        for modality in ['T1', 'T2', 'FLAIR', 'DWI', 'ADC', 'Cross-Modality']:
            modality_features = [col for col in self.features_scaled.columns if col.startswith(f'{modality}_')]
            if modality_features:
                communities[modality] = modality_features
        
        self.results['network_analysis'] = {
            'edges': network_edges,
            'metrics': network_metrics,
            'communities': communities
        }
        
        return self.results['network_analysis']
    
    def technique_10_quality_assessment(self):
        """Technique 10: Comprehensive Quality Assessment"""
        print("✅ Technique 10: Comprehensive Quality Assessment")
        
        # Data quality metrics
        quality_metrics = {
            'completeness': 1 - self.features.isnull().sum().sum() / (self.features.shape[0] * self.features.shape[1]),
            'consistency': self.features.std().mean() / self.features.mean().mean(),
            'reliability': 1 - (self.features == 0).sum().sum() / (self.features.shape[0] * self.features.shape[1])
        }
        
        # Feature quality scores
        feature_quality = {}
        for col in self.features_scaled.columns:
            feature_data = self.features_scaled[col]
            feature_quality[col] = {
                'variance': feature_data.var(),
                'skewness': abs(feature_data.skew()),
                'outlier_ratio': (abs(stats.zscore(feature_data)) > 3).mean(),
                'quality_score': 1 - (abs(stats.zscore(feature_data)) > 3).mean()
            }
        
        # Overall quality score
        overall_quality = np.mean([feature_quality[col]['quality_score'] for col in feature_quality])
        
        self.results['quality_assessment'] = {
            'metrics': quality_metrics,
            'feature_quality': feature_quality,
            'overall_quality': overall_quality
        }
        
        return self.results['quality_assessment']
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations for all techniques"""
        print("🎨 Creating comprehensive visualizations...")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(30, 40))
        fig.patch.set_facecolor('#f8f9fa')
        
        # Define custom colors
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        # 1. Feature Correlation Heatmap
        ax1 = plt.subplot(5, 3, 1)
        corr_matrix = self.results['correlation']['matrix']
        sns.heatmap(corr_matrix.iloc[:30, :30], cmap='RdBu_r', center=0, 
                   square=True, cbar_kws={'shrink': 0.8}, annot=False)
        plt.title('Feature Correlation Heatmap\n(Top 30 Features)', fontsize=14, fontweight='bold', pad=20)
        
        # 2. High Correlation Pairs
        ax2 = plt.subplot(5, 3, 2)
        high_pairs = self.results['correlation']['high_pairs'][:10]
        pairs = [f"{pair[0][:15]}...\n{pair[1][:15]}..." for pair in high_pairs]
        corr_values = [pair[2] for pair in high_pairs]
        
        bars = plt.barh(range(len(pairs)), corr_values, 
                       color=['#FF6B6B' if x < 0 else '#4ECDC4' for x in corr_values],
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.yticks(range(len(pairs)), pairs, fontsize=9)
        plt.xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        plt.title('Top 10 High Correlations', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 3. Clustering Silhouette Analysis
        ax3 = plt.subplot(5, 3, 3)
        silhouette_scores = self.results['clustering']['kmeans']['silhouette_scores']
        k_range = range(2, 11)
        plt.plot(k_range, silhouette_scores, 'o-', linewidth=3, markersize=10, 
                color='#FF6B6B', markerfacecolor='white', markeredgewidth=2)
        plt.axvline(x=self.results['clustering']['kmeans']['optimal_k'], 
                   color='#4ECDC4', linestyle='--', linewidth=3)
        plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
        plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        plt.title('K-means Silhouette Analysis', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 4. PCA Visualization
        ax4 = plt.subplot(5, 3, 4)
        pca_result = self.results['dimensionality']['pca']['result']
        kmeans_labels = self.results['clustering']['kmeans']['labels']
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                            c=kmeans_labels, cmap='Set2', alpha=0.8, s=60,
                            edgecolors='black', linewidth=0.5)
        plt.xlabel('PC1', fontsize=12, fontweight='bold')
        plt.ylabel('PC2', fontsize=12, fontweight='bold')
        plt.title('PCA Visualization\n(Colored by Cluster)', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(scatter, label='Cluster', shrink=0.8)
        plt.grid(True, alpha=0.3)
        
        # 5. t-SNE Visualization
        ax5 = plt.subplot(5, 3, 5)
        tsne_result = self.results['dimensionality']['tsne']['result']
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                            c=kmeans_labels, cmap='Set2', alpha=0.8, s=60,
                            edgecolors='black', linewidth=0.5)
        plt.xlabel('t-SNE 1', fontsize=12, fontweight='bold')
        plt.ylabel('t-SNE 2', fontsize=12, fontweight='bold')
        plt.title('t-SNE Visualization\n(Colored by Cluster)', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(scatter, label='Cluster', shrink=0.8)
        plt.grid(True, alpha=0.3)
        
        # 6. UMAP Visualization
        ax6 = plt.subplot(5, 3, 6)
        umap_result = self.results['dimensionality']['umap']['result']
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                            c=kmeans_labels, cmap='Set2', alpha=0.8, s=60,
                            edgecolors='black', linewidth=0.5)
        plt.xlabel('UMAP 1', fontsize=12, fontweight='bold')
        plt.ylabel('UMAP 2', fontsize=12, fontweight='bold')
        plt.title('UMAP Visualization\n(Colored by Cluster)', fontsize=14, fontweight='bold', pad=20)
        plt.colorbar(scatter, label='Cluster', shrink=0.8)
        plt.grid(True, alpha=0.3)
        
        # 7. Feature Importance (Random Forest)
        ax7 = plt.subplot(5, 3, 7)
        rf_importance = self.results['feature_importance']['random_forest']
        feature_names = self.results['feature_importance']['feature_names']
        top_indices = np.argsort(rf_importance)[-15:]
        top_features = [feature_names[i][:20] + '...' for i in top_indices]
        top_importance = [rf_importance[i] for i in top_indices]
        
        bars = plt.barh(range(len(top_features)), top_importance, 
                       color='#DDA0DD', alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.yticks(range(len(top_features)), top_features, fontsize=9)
        plt.xlabel('Random Forest Importance', fontsize=12, fontweight='bold')
        plt.title('Top 15 Features\n(Random Forest)', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        
        # 8. Modality Analysis
        ax8 = plt.subplot(5, 3, 8)
        modality_stats = self.results['modality_analysis']['statistics']
        modalities = list(modality_stats.keys())
        mean_correlations = [modality_stats[m]['mean_correlation'] for m in modalities]
        
        bars = plt.bar(modalities, mean_correlations, 
                      color=colors[:len(modalities)], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        plt.xlabel('Modality', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Correlation', fontsize=12, fontweight='bold')
        plt.title('Modality Similarity Analysis', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 9. Temporal Analysis
        ax9 = plt.subplot(5, 3, 9)
        year_stats = self.results['temporal_analysis']['year_stats']
        years = list(year_stats.keys())
        year_correlations = [year_stats[y]['feature_correlation'] for y in years]
        
        bars = plt.bar(years, year_correlations, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        plt.xlabel('Year', fontsize=12, fontweight='bold')
        plt.ylabel('Feature Correlation', fontsize=12, fontweight='bold')
        plt.title('Temporal Consistency', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 10. Outlier Detection
        ax10 = plt.subplot(5, 3, 10)
        outlier_results = self.results['outlier_detection']
        methods = ['Isolation Forest', 'One-Class SVM', 'Z-Score', 'IQR']
        outlier_counts = [
            (outlier_results['isolation_forest'] == -1).sum(),
            (outlier_results['one_class_svm'] == -1).sum(),
            outlier_results['z_score'].sum(),
            outlier_results['iqr'].sum()
        ]
        
        bars = plt.bar(methods, outlier_counts, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        plt.xlabel('Detection Method', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Outliers', fontsize=12, fontweight='bold')
        plt.title('Outlier Detection Results', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 11. Quality Assessment
        ax11 = plt.subplot(5, 3, 11)
        quality_metrics = self.results['quality_assessment']['metrics']
        metrics = list(quality_metrics.keys())
        values = list(quality_metrics.values())
        
        bars = plt.bar(metrics, values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        plt.xlabel('Quality Metric', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Data Quality Assessment', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 12. Network Analysis
        ax12 = plt.subplot(5, 3, 12)
        network_metrics = self.results['network_analysis']['metrics']
        metric_names = list(network_metrics.keys())
        metric_values = list(network_metrics.values())
        
        bars = plt.bar(metric_names, metric_values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        plt.xlabel('Network Metric', fontsize=12, fontweight='bold')
        plt.ylabel('Value', fontsize=12, fontweight='bold')
        plt.title('Network Analysis Metrics', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 13. Statistical Distribution
        ax13 = plt.subplot(5, 3, 13)
        # Show distribution of feature skewness
        skewness_values = [self.results['statistical_analysis']['distribution'][col]['skewness'] 
                          for col in self.features_scaled.columns]
        plt.hist(skewness_values, bins=30, alpha=0.8, color='#FFEAA7', 
                edgecolor='black', linewidth=0.5)
        plt.xlabel('Skewness', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Feature Skewness Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.axvline(x=0, color='#FF6B6B', linestyle='--', alpha=0.8, linewidth=2)
        plt.grid(True, alpha=0.3)
        
        # 14. Feature Stability
        ax14 = plt.subplot(5, 3, 14)
        stability_values = list(self.results['statistical_analysis']['stability'].values())
        plt.hist(stability_values, bins=30, alpha=0.8, color='#DDA0DD', 
                edgecolor='black', linewidth=0.5)
        plt.xlabel('Stability Score', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Feature Stability Distribution', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        
        # 15. Summary Statistics
        ax15 = plt.subplot(5, 3, 15)
        summary_data = {
            'Total Patients': len(self.data),
            'Total Features': len(self.features_scaled.columns),
            'Optimal Clusters': self.results['clustering']['kmeans']['optimal_k'],
            'High Correlations': len(self.results['correlation']['high_pairs']),
            'Overall Quality': f"{self.results['quality_assessment']['overall_quality']:.3f}"
        }
        
        y_positions = np.arange(len(summary_data))
        plt.barh(y_positions, [1] * len(summary_data), color='#F7DC6F', alpha=0.3)
        plt.yticks(y_positions, summary_data.keys(), fontsize=10)
        plt.xticks([])
        plt.title('Analysis Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Add value labels
        for i, value in enumerate(summary_data.values()):
            plt.text(0.5, i, str(value), ha='center', va='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('comprehensive_radiomics_analysis.png', dpi=300, bbox_inches='tight',
                    facecolor='#f8f9fa')
        plt.close()
        
        print("✅ Comprehensive visualizations saved to: comprehensive_radiomics_analysis.png")
    
    def run_complete_analysis(self):
        """Run all 10 radiomics analysis techniques"""
        print("🚀 Starting Comprehensive Radiomics Analysis...")
        print("=" * 60)
        
        # Load and prepare data
        self.load_enhanced_synthetic_data()
        self.prepare_features()
        
        # Run all 10 techniques
        print("\n📊 Running Top 10 Radiomics Analysis Techniques:")
        print("-" * 50)
        
        self.technique_1_feature_correlation_analysis()
        self.technique_2_clustering_analysis()
        self.technique_3_dimensionality_reduction()
        self.technique_4_feature_importance_analysis()
        self.technique_5_outlier_detection()
        self.technique_6_modality_analysis()
        self.technique_7_temporal_analysis()
        self.technique_8_statistical_analysis()
        self.technique_9_network_analysis()
        self.technique_10_quality_assessment()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        # Save results
        self.save_analysis_results()
        
        print("\n✅ Comprehensive Radiomics Analysis Completed Successfully!")
        print("=" * 60)
    
    def save_analysis_results(self):
        """Save comprehensive analysis results"""
        print("💾 Saving analysis results...")
        
        # Save cluster assignments
        cluster_df = self.data.copy()
        cluster_df['KMeans_Cluster'] = self.results['clustering']['kmeans']['labels']
        cluster_df['DBSCAN_Cluster'] = self.results['clustering']['dbscan']['labels']
        cluster_df['Hierarchical_Cluster'] = self.results['clustering']['hierarchical']['labels']
        cluster_df.to_csv('comprehensive_radiomics_clusters.csv', index=False)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'Feature': self.results['feature_importance']['feature_names'],
            'RandomForest_Importance': self.results['feature_importance']['random_forest'],
            'F_Score': self.results['feature_importance']['f_scores'],
            'Mutual_Info': self.results['feature_importance']['mutual_info'],
            'Variance': self.results['feature_importance']['variance'],
            'Target_Correlation': self.results['feature_importance']['target_correlation']
        })
        importance_df.to_csv('comprehensive_feature_importance.csv', index=False)
        
        # Save correlation pairs
        corr_df = pd.DataFrame(self.results['correlation']['high_pairs'], 
                              columns=['Feature1', 'Feature2', 'Correlation'])
        corr_df.to_csv('comprehensive_high_correlations.csv', index=False)
        
        # Save comprehensive summary
        with open('comprehensive_radiomics_summary.txt', 'w') as f:
            f.write("COMPREHENSIVE RADIOMICS ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset Information:\n")
            f.write(f"- Total patients: {len(self.data)}\n")
            f.write(f"- Total features: {len(self.features_scaled.columns)}\n")
            f.write(f"- Years: {sorted(self.data['Year'].unique())}\n\n")
            
            f.write(f"Analysis Results:\n")
            f.write(f"- Optimal clusters (K-means): {self.results['clustering']['kmeans']['optimal_k']}\n")
            f.write(f"- High correlation pairs: {len(self.results['correlation']['high_pairs'])}\n")
            f.write(f"- Average correlation: {self.results['correlation']['avg_correlation']:.3f}\n")
            f.write(f"- Overall quality score: {self.results['quality_assessment']['overall_quality']:.3f}\n\n")
            
            f.write(f"Top 10 Analysis Techniques Applied:\n")
            f.write(f"1. Feature Correlation Analysis\n")
            f.write(f"2. Multi-Algorithm Clustering Analysis\n")
            f.write(f"3. Multi-Method Dimensionality Reduction\n")
            f.write(f"4. Multi-Method Feature Importance Analysis\n")
            f.write(f"5. Multi-Method Outlier Detection\n")
            f.write(f"6. Comprehensive Modality Analysis\n")
            f.write(f"7. Temporal and Longitudinal Analysis\n")
            f.write(f"8. Comprehensive Statistical Analysis\n")
            f.write(f"9. Network and Graph Analysis\n")
            f.write(f"10. Comprehensive Quality Assessment\n")
        
        print("✅ Analysis results saved successfully!")

def main():
    """Main function to run comprehensive radiomics analysis"""
    
    # Create analyzer instance
    analyzer = ComprehensiveRadiomicsAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    print("\n📁 Generated Files:")
    print("- comprehensive_radiomics_analysis.png (15-panel visualization)")
    print("- comprehensive_radiomics_clusters.csv (cluster assignments)")
    print("- comprehensive_feature_importance.csv (feature importance)")
    print("- comprehensive_high_correlations.csv (high correlations)")
    print("- comprehensive_radiomics_summary.txt (analysis summary)")

if __name__ == "__main__":
    main() 