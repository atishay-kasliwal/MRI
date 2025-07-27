import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import nibabel as nib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set golden theme colors
GOLDEN_COLORS = {
    'primary_gold': '#B8860B',
    'secondary_gold': '#DAA520', 
    'light_gold': '#F4A460',
    'pale_gold': '#F5DEB3',
    'dark_gold': '#8B6914',
    'black': '#000000',
    'white': '#FFFFFF',
    'grey': '#808080',
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

# Set matplotlib style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['axes.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['axes.edgecolor'] = GOLDEN_COLORS['dark_gold']
plt.rcParams['axes.labelcolor'] = GOLDEN_COLORS['black']
plt.rcParams['xtick.color'] = GOLDEN_COLORS['black']
plt.rcParams['ytick.color'] = GOLDEN_COLORS['black']
plt.rcParams['text.color'] = GOLDEN_COLORS['black']

class AdvancedRadiomicsVisualizer:
    """
    Advanced Radiomics Feature Extraction and Visualization
    Implements latest visualization techniques for radiomics analysis
    """
    
    def __init__(self, base_path="/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip"):
        """Initialize the advanced visualizer"""
        self.base_path = base_path
        self.years = [2020, 2021, 2022, 2023, 2024]
        self.modalities = ['T1', 'T2', 'FLAIR', 'DWI', 'ADC']
        self.feature_data = {}
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        
        print("=== ADVANCED RADIOMICS VISUALIZER INITIALIZED ===")
        print(f"Processing years: {self.years}")
        print(f"Modalities: {self.modalities}")
    
    def extract_basic_features(self):
        """Extract basic features from imaging data"""
        print("\n=== EXTRACTING BASIC FEATURES ===")
        
        all_features = []
        feature_metadata = []
        
        for year in self.years:
            year_path = os.path.join(self.base_path, str(year))
            if os.path.exists(year_path):
                # Handle nested year folders
                if os.path.exists(os.path.join(year_path, str(year))):
                    year_path = os.path.join(year_path, str(year))
                
                patients = [d for d in os.listdir(year_path) if d.startswith('DE-IDENTIFIED')]
                
                for patient in patients:
                    patient_path = os.path.join(year_path, patient)
                    outcome_path = os.path.join(patient_path, 'outcome')
                    
                    if os.path.exists(outcome_path):
                        files = os.listdir(outcome_path)
                        mask_files = [f for f in files if 'mask' in f.lower() and f.endswith('.nii.gz')]
                        scan_files = [f for f in files if f.startswith('CORRECT') and f.endswith('.nii.gz')]
                        
                        if mask_files and scan_files:
                            # Extract basic features from each modality
                            for scan_file in scan_files:
                                try:
                                    scan_path = os.path.join(outcome_path, scan_file)
                                    modality = scan_file.split('_')[1].replace('CORRECT', '')
                                    
                                    # Load image data
                                    img = nib.load(scan_path)
                                    data = img.get_fdata()
                                    
                                    # Extract basic radiomics-like features
                                    features = self._extract_basic_radiomics_features(data)
                                    
                                    # Add metadata
                                    features['PatientID'] = patient
                                    features['Year'] = year
                                    features['Modality'] = modality
                                    features['ScanFile'] = scan_file
                                    
                                    all_features.append(features)
                                    feature_metadata.append({
                                        'PatientID': patient,
                                        'Year': year,
                                        'Modality': modality,
                                        'ScanFile': scan_file
                                    })
                                    
                                except Exception as e:
                                    print(f"Error processing {scan_file}: {e}")
        
        if all_features:
            self.feature_df = pd.DataFrame(all_features)
            self.metadata_df = pd.DataFrame(feature_metadata)
            
            print(f"✅ Extracted features from {len(self.feature_df)} scans")
            print(f"Features per scan: {len(self.feature_df.columns) - 4}")  # Exclude metadata columns
            
            return self.feature_df
        else:
            print("❌ No features extracted")
            return None
    
    def _extract_basic_radiomics_features(self, data):
        """Extract basic radiomics-like features from image data"""
        features = {}
        
        # Remove NaN and infinite values
        data_clean = data[~np.isnan(data) & ~np.isinf(data)]
        
        if len(data_clean) == 0:
            return features
        
        # First-order statistics
        features['mean_intensity'] = np.mean(data_clean)
        features['std_intensity'] = np.std(data_clean)
        features['min_intensity'] = np.min(data_clean)
        features['max_intensity'] = np.max(data_clean)
        features['median_intensity'] = np.median(data_clean)
        features['skewness'] = self._calculate_skewness(data_clean)
        features['kurtosis'] = self._calculate_kurtosis(data_clean)
        features['variance'] = np.var(data_clean)
        features['energy'] = np.sum(data_clean**2)
        features['entropy'] = self._calculate_entropy(data_clean)
        
        # Shape features
        features['volume'] = len(data_clean)
        features['surface_area'] = self._estimate_surface_area(data)
        features['sphericity'] = self._calculate_sphericity(data)
        features['compactness'] = self._calculate_compactness(data)
        
        # Texture-like features
        features['contrast'] = self._calculate_contrast(data)
        features['homogeneity'] = self._calculate_homogeneity(data)
        features['correlation'] = self._calculate_correlation(data)
        
        # Histogram features
        hist, bins = np.histogram(data_clean, bins=50)
        features['histogram_mean'] = np.mean(hist)
        features['histogram_std'] = np.std(hist)
        features['histogram_entropy'] = self._calculate_entropy(hist)
        
        return features
    
    def _calculate_skewness(self, data):
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_entropy(self, data):
        """Calculate entropy"""
        if len(data) == 0:
            return 0
        hist, _ = np.histogram(data, bins=min(50, len(data)//10))
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0
        p = hist / np.sum(hist)
        return -np.sum(p * np.log2(p + 1e-10))
    
    def _estimate_surface_area(self, data):
        """Estimate surface area from 3D data"""
        # Simple estimation based on non-zero voxels
        non_zero = np.sum(data > 0)
        return non_zero ** (2/3)  # Approximation
    
    def _calculate_sphericity(self, data):
        """Calculate sphericity"""
        volume = np.sum(data > 0)
        if volume == 0:
            return 0
        surface_area = self._estimate_surface_area(data)
        return (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
    
    def _calculate_compactness(self, data):
        """Calculate compactness"""
        volume = np.sum(data > 0)
        if volume == 0:
            return 0
        surface_area = self._estimate_surface_area(data)
        return volume / (surface_area ** (3/2))
    
    def _calculate_contrast(self, data):
        """Calculate contrast"""
        if len(data) < 2:
            return 0
        return np.max(data) - np.min(data)
    
    def _calculate_homogeneity(self, data):
        """Calculate homogeneity"""
        if len(data) == 0:
            return 0
        return 1 / (1 + np.var(data))
    
    def _calculate_correlation(self, data):
        """Calculate correlation-like measure"""
        if len(data) < 2:
            return 0
        # Calculate correlation with position indices
        indices = np.arange(len(data))
        return np.corrcoef(data.flatten(), indices)[0, 1] if not np.isnan(np.corrcoef(data.flatten(), indices)[0, 1]) else 0
    
    def create_advanced_visualizations(self, output_path='advanced_radiomics_visualizations.pdf'):
        """Create advanced radiomics visualizations"""
        print(f"\n=== CREATING ADVANCED VISUALIZATIONS ===")
        
        if not hasattr(self, 'feature_df') or self.feature_df is None:
            print("❌ No feature data available. Run extract_basic_features() first.")
            return
        
        with PdfPages(output_path) as pdf:
            
            # 1. Feature Distribution Analysis
            self._create_feature_distribution_analysis(pdf)
            
            # 2. Dimensionality Reduction Visualizations
            self._create_dimensionality_reduction_analysis(pdf)
            
            # 3. Feature Correlation Analysis
            self._create_correlation_analysis(pdf)
            
            # 4. Temporal Feature Evolution
            self._create_temporal_analysis(pdf)
            
            # 5. Modality Comparison
            self._create_modality_comparison(pdf)
            
            # 6. Feature Stability Analysis
            self._create_stability_analysis(pdf)
            
            # 7. Advanced Clustering Analysis
            self._create_clustering_analysis(pdf)
            
            # 8. Feature Importance Analysis
            self._create_feature_importance_analysis(pdf)
        
        print(f"✅ Advanced visualizations saved to {output_path}")
    
    def _create_feature_distribution_analysis(self, pdf):
        """Create comprehensive feature distribution analysis"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Advanced Feature Distribution Analysis', fontsize=20, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Get numeric features
        numeric_cols = self.feature_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Year']]
        
        # Select key features for visualization
        key_features = ['mean_intensity', 'std_intensity', 'entropy', 'skewness', 'kurtosis', 
                       'volume', 'sphericity', 'contrast', 'homogeneity']
        
        for i, feature in enumerate(key_features[:9]):
            row, col = i // 3, i % 3
            
            if feature in numeric_cols:
                data = self.feature_df[feature].dropna()
                
                # Histogram with KDE
                axes[row, col].hist(data, bins=30, alpha=0.7, color=GOLDEN_COLORS['primary_gold'], 
                                  edgecolor=GOLDEN_COLORS['dark_gold'], density=True)
                
                # Add KDE
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                axes[row, col].plot(x_range, kde(x_range), color=GOLDEN_COLORS['red'], linewidth=2)
                
                axes[row, col].set_title(f'{feature.replace("_", " ").title()}', 
                                       fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
                axes[row, col].set_xlabel('Value', fontweight='bold')
                axes[row, col].set_ylabel('Density', fontweight='bold')
                axes[row, col].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
                
                # Add statistics
                mean_val = np.mean(data)
                std_val = np.std(data)
                axes[row, col].axvline(mean_val, color=GOLDEN_COLORS['green'], linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[row, col].legend()
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_dimensionality_reduction_analysis(self, pdf):
        """Create dimensionality reduction visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dimensionality Reduction Analysis', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Prepare data
        numeric_cols = self.feature_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Year']]
        
        X = self.feature_df[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Color by year
        years = self.feature_df['Year'].values
        unique_years = np.unique(years)
        colors = [GOLDEN_COLORS['primary_gold'], GOLDEN_COLORS['secondary_gold'], 
                 GOLDEN_COLORS['light_gold'], GOLDEN_COLORS['blue'], GOLDEN_COLORS['green']]
        
        # 1. PCA Analysis
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        for i, year in enumerate(unique_years):
            mask = years == year
            axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             c=colors[i % len(colors)], alpha=0.7, s=50, label=f'Year {year}')
        
        axes[0, 0].set_title(f'PCA Analysis (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})', 
                           fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 0].set_xlabel('PC1', fontweight='bold')
        axes[0, 0].set_ylabel('PC2', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # 2. t-SNE Analysis
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled)
        
        for i, year in enumerate(unique_years):
            mask = years == year
            axes[0, 1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                             c=colors[i % len(colors)], alpha=0.7, s=50, label=f'Year {year}')
        
        axes[0, 1].set_title('t-SNE Analysis', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 1].set_xlabel('t-SNE 1', fontweight='bold')
        axes[0, 1].set_ylabel('t-SNE 2', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # 3. MDS Analysis
        mds = MDS(n_components=2, random_state=42)
        X_mds = mds.fit_transform(X_scaled)
        
        for i, year in enumerate(unique_years):
            mask = years == year
            axes[1, 0].scatter(X_mds[mask, 0], X_mds[mask, 1], 
                             c=colors[i % len(colors)], alpha=0.7, s=50, label=f'Year {year}')
        
        axes[1, 0].set_title('MDS Analysis', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 0].set_xlabel('MDS 1', fontweight='bold')
        axes[1, 0].set_ylabel('MDS 2', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # 4. Explained variance plot
        pca_full = PCA()
        pca_full.fit(X_scaled)
        explained_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        axes[1, 1].plot(range(1, len(explained_var) + 1), explained_var, 
                       marker='o', linewidth=2, color=GOLDEN_COLORS['primary_gold'])
        axes[1, 1].axhline(y=0.95, color=GOLDEN_COLORS['red'], linestyle='--', label='95% Variance')
        axes[1, 1].axhline(y=0.90, color=GOLDEN_COLORS['orange'], linestyle='--', label='90% Variance')
        axes[1, 1].set_title('PCA Explained Variance', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 1].set_xlabel('Number of Components', fontweight='bold')
        axes[1, 1].set_ylabel('Cumulative Explained Variance', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_correlation_analysis(self, pdf):
        """Create feature correlation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Get numeric features
        numeric_cols = self.feature_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['Year']]
        
        # 1. Correlation heatmap
        corr_matrix = self.feature_df[feature_cols].corr()
        
        im = axes[0, 0].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        axes[0, 0].set_title('Feature Correlation Heatmap', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 0].set_xticks(range(len(feature_cols)))
        axes[0, 0].set_xticklabels([col[:10] for col in feature_cols], rotation=45, ha='right')
        axes[0, 0].set_yticks(range(len(feature_cols)))
        axes[0, 0].set_yticklabels([col[:10] for col in feature_cols])
        
        # Add correlation values
        for i in range(len(feature_cols)):
            for j in range(len(feature_cols)):
                text = axes[0, 0].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                                      ha="center", va="center", color="black", fontsize=8)
        
        # 2. Top correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((corr_matrix.iloc[i, j], corr_matrix.columns[i], corr_matrix.columns[j]))
        
        corr_pairs.sort(key=lambda x: abs(x[0]), reverse=True)
        top_correlations = corr_pairs[:10]
        
        features1 = [pair[1][:15] for pair in top_correlations]
        features2 = [pair[2][:15] for pair in top_correlations]
        correlations = [pair[0] for pair in top_correlations]
        
        y_pos = np.arange(len(top_correlations))
        colors = [GOLDEN_COLORS['red'] if x < 0 else GOLDEN_COLORS['green'] for x in correlations]
        
        bars = axes[0, 1].barh(y_pos, correlations, color=colors, alpha=0.7)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels([f'{f1}\n{f2}' for f1, f2 in zip(features1, features2)])
        axes[0, 1].set_title('Top 10 Feature Correlations', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 1].set_xlabel('Correlation Coefficient', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # 3. Feature variance analysis
        feature_vars = self.feature_df[feature_cols].var().sort_values(ascending=False)
        top_variance_features = feature_vars.head(10)
        
        axes[1, 0].bar(range(len(top_variance_features)), top_variance_features.values, 
                      color=GOLDEN_COLORS['blue'], alpha=0.7)
        axes[1, 0].set_title('Top 10 Features by Variance', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 0].set_xlabel('Features', fontweight='bold')
        axes[1, 0].set_ylabel('Variance', fontweight='bold')
        axes[1, 0].set_xticks(range(len(top_variance_features)))
        axes[1, 0].set_xticklabels([col[:10] for col in top_variance_features.index], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # 4. Feature stability across years
        stability_scores = []
        feature_names = []
        
        for feature in feature_cols[:10]:  # Top 10 features
            year_means = []
            for year in self.years:
                year_data = self.feature_df[self.feature_df['Year'] == year][feature]
                if len(year_data) > 0:
                    year_means.append(year_data.mean())
            
            if len(year_means) > 1:
                stability = 1 / (1 + np.std(year_means))  # Higher stability = lower std
                stability_scores.append(stability)
                feature_names.append(feature[:10])
        
        axes[1, 1].bar(range(len(stability_scores)), stability_scores, 
                      color=GOLDEN_COLORS['purple'], alpha=0.7)
        axes[1, 1].set_title('Feature Stability Across Years', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 1].set_xlabel('Features', fontweight='bold')
        axes[1, 1].set_ylabel('Stability Score', fontweight='bold')
        axes[1, 1].set_xticks(range(len(stability_scores)))
        axes[1, 1].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

def main():
    """Main function to run advanced radiomics visualizations"""
    print("=== ADVANCED RADIOMICS VISUALIZATIONS ===")
    print("Creating advanced feature analysis and visualizations...\n")
    
    # Initialize visualizer
    visualizer = AdvancedRadiomicsVisualizer()
    
    # Extract features
    feature_df = visualizer.extract_basic_features()
    
    if feature_df is not None:
        # Create advanced visualizations
        visualizer.create_advanced_visualizations('advanced_radiomics_visualizations.pdf')
        
        print("\n=== VISUALIZATION COMPLETED ===")
        print("Generated analyses include:")
        print("1. Feature Distribution Analysis")
        print("2. Dimensionality Reduction Analysis")
        print("3. Feature Correlation Analysis")
        print("4. Temporal Feature Evolution")
        print("5. Modality Comparison")
        print("6. Feature Stability Analysis")
        print("7. Advanced Clustering Analysis")
        print("8. Feature Importance Analysis")
        print("\nKey insights:")
        print("- Advanced statistical analysis")
        print("- Modern visualization techniques")
        print("- Feature stability assessment")
        print("- Temporal evolution patterns")

if __name__ == "__main__":
    main() 