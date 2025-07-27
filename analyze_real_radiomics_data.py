#!/usr/bin/env python3
"""
Analyze Real Radiomics Data
Analyzes the extracted radiomics features from real MRI data and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_combine_radiomics_data():
    """Load and combine radiomics data from all years"""
    
    # Paths to radiomics files
    radiomics_2020 = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2020/all_patients_basic_radiomics_2020.csv')
    radiomics_2021 = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2021/all_patients_basic_radiomics_2021.csv')
    radiomics_2022 = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2022/2022/all_patients_basic_radiomics_2022.csv')
    
    # Load data
    print("Loading radiomics data...")
    df_2020 = pd.read_csv(radiomics_2020)
    df_2021 = pd.read_csv(radiomics_2021)
    df_2022 = pd.read_csv(radiomics_2022)
    
    # Add year column
    df_2020['Year'] = 2020
    df_2021['Year'] = 2021
    df_2022['Year'] = 2022
    
    # Combine data
    combined_df = pd.concat([df_2020, df_2021, df_2022], ignore_index=True)
    
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Years: {combined_df['Year'].value_counts().to_dict()}")
    print(f"Modalities: {combined_df['Modality'].value_counts().to_dict()}")
    
    return combined_df

def create_comprehensive_visualizations(df):
    """Create comprehensive visualizations of the radiomics data"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Data distribution by year and modality
    ax1 = plt.subplot(3, 4, 1)
    year_modality_counts = df.groupby(['Year', 'Modality']).size().unstack(fill_value=0)
    year_modality_counts.plot(kind='bar', ax=ax1)
    plt.title('Data Distribution by Year and Modality')
    plt.xlabel('Year')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=0)
    plt.legend(title='Modality')
    plt.grid(True, alpha=0.3)
    
    # 2. Feature correlation heatmap (top features)
    ax2 = plt.subplot(3, 4, 2)
    # Select numeric features for correlation
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f not in ['Year']]
    
    # Calculate correlation matrix for top features
    top_features = ['mean', 'std', 'volume', 'entropy', 'energy', 'skewness', 'kurtosis']
    available_features = [f for f in top_features if f in numeric_features]
    
    if len(available_features) > 1:
        correlation_matrix = df[available_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix')
    
    # 3. Volume distribution by modality
    ax3 = plt.subplot(3, 4, 3)
    if 'volume' in df.columns:
        df.boxplot(column='volume', by='Modality', ax=ax3)
        plt.title('Volume Distribution by Modality')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 4. Mean intensity by modality
    ax4 = plt.subplot(3, 4, 4)
    if 'mean' in df.columns:
        df.boxplot(column='mean', by='Modality', ax=ax4)
        plt.title('Mean Intensity by Modality')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 5. Entropy distribution by modality
    ax5 = plt.subplot(3, 4, 5)
    if 'entropy' in df.columns:
        df.boxplot(column='entropy', by='Modality', ax=ax5)
        plt.title('Entropy Distribution by Modality')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 6. Energy distribution by modality
    ax6 = plt.subplot(3, 4, 6)
    if 'energy' in df.columns:
        # Log scale for energy due to large values
        df['log_energy'] = np.log10(df['energy'] + 1)
        df.boxplot(column='log_energy', by='Modality', ax=ax6)
        plt.title('Log Energy Distribution by Modality')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 7. Skewness distribution by modality
    ax7 = plt.subplot(3, 4, 7)
    if 'skewness' in df.columns:
        df.boxplot(column='skewness', by='Modality', ax=ax7)
        plt.title('Skewness Distribution by Modality')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 8. Kurtosis distribution by modality
    ax8 = plt.subplot(3, 4, 8)
    if 'kurtosis' in df.columns:
        df.boxplot(column='kurtosis', by='Modality', ax=ax8)
        plt.title('Kurtosis Distribution by Modality')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 9. Feature importance (variance across modalities)
    ax9 = plt.subplot(3, 4, 9)
    if len(available_features) > 0:
        # Calculate feature variance across modalities
        feature_variance = []
        feature_names = []
        
        for feature in available_features:
            if feature not in ['Year']:
                variance = df.groupby('Modality')[feature].var().mean()
                feature_variance.append(variance)
                feature_names.append(feature)
        
        if feature_variance:
            # Sort by variance
            sorted_indices = np.argsort(feature_variance)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_variance = [feature_variance[i] for i in sorted_indices]
            
            plt.barh(range(len(sorted_features)), sorted_variance)
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Average Variance Across Modalities')
            plt.title('Feature Discriminative Power')
            plt.grid(True, alpha=0.3)
    
    # 10. Patient distribution by year
    ax10 = plt.subplot(3, 4, 10)
    patient_counts = df.groupby('Year')['PatientID'].nunique()
    patient_counts.plot(kind='bar', ax=ax10)
    plt.title('Number of Patients by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    # 11. Modality comparison (mean values)
    ax11 = plt.subplot(3, 4, 11)
    if 'mean' in df.columns:
        modality_means = df.groupby('Modality')['mean'].mean()
        modality_means.plot(kind='bar', ax=ax11)
        plt.title('Average Mean Intensity by Modality')
        plt.xlabel('Modality')
        plt.ylabel('Mean Intensity')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 12. Volume vs Energy scatter plot
    ax12 = plt.subplot(3, 4, 12)
    if 'volume' in df.columns and 'energy' in df.columns:
        for modality in df['Modality'].unique():
            modality_data = df[df['Modality'] == modality]
            plt.scatter(modality_data['volume'], np.log10(modality_data['energy'] + 1), 
                       alpha=0.6, label=modality)
        plt.xlabel('Volume')
        plt.ylabel('Log Energy')
        plt.title('Volume vs Energy by Modality')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_modality_comparison_plots(df):
    """Create detailed modality comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Features to compare
    features_to_compare = ['mean', 'std', 'entropy', 'volume', 'skewness', 'kurtosis']
    
    for i, feature in enumerate(features_to_compare):
        if feature in df.columns:
            row = i // 3
            col = i % 3
            
            # Create violin plot
            sns.violinplot(data=df, x='Modality', y=feature, ax=axes[row, col])
            axes[row, col].set_title(f'{feature.capitalize()} by Modality')
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_year_comparison_plots(df):
    """Create year comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Features to compare
    features_to_compare = ['mean', 'std', 'entropy', 'volume', 'skewness', 'kurtosis']
    
    for i, feature in enumerate(features_to_compare):
        if feature in df.columns:
            row = i // 3
            col = i % 3
            
            # Create box plot
            df.boxplot(column=feature, by='Year', ax=axes[row, col])
            axes[row, col].set_title(f'{feature.capitalize()} by Year')
            axes[row, col].set_xlabel('Year')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_statistical_summary(df):
    """Create statistical summary of the data"""
    
    print("\n=== RADIOMICS DATA STATISTICAL SUMMARY ===\n")
    
    # Basic statistics
    print("Dataset Overview:")
    print(f"Total samples: {len(df)}")
    print(f"Unique patients: {df['PatientID'].nunique()}")
    print(f"Years: {sorted(df['Year'].unique())}")
    print(f"Modalities: {sorted(df['Modality'].unique())}")
    
    print("\nSamples by Year:")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} samples")
    
    print("\nSamples by Modality:")
    modality_counts = df['Modality'].value_counts()
    for modality, count in modality_counts.items():
        print(f"  {modality}: {count} samples")
    
    print("\nPatients by Year:")
    patients_by_year = df.groupby('Year')['PatientID'].nunique()
    for year, count in patients_by_year.items():
        print(f"  {year}: {count} patients")
    
    # Feature statistics
    print("\nFeature Statistics (across all modalities):")
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f not in ['Year']]
    
    for feature in numeric_features[:10]:  # Show first 10 features
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        min_val = df[feature].min()
        max_val = df[feature].max()
        print(f"  {feature}: mean={mean_val:.2f}, std={std_val:.2f}, range=[{min_val:.2f}, {max_val:.2f}]")
    
    return {
        'total_samples': len(df),
        'unique_patients': df['PatientID'].nunique(),
        'years': sorted(df['Year'].unique()),
        'modalities': sorted(df['Modality'].unique()),
        'year_counts': year_counts.to_dict(),
        'modality_counts': modality_counts.to_dict(),
        'patients_by_year': patients_by_year.to_dict()
    }

def main():
    """Main function to analyze radiomics data"""
    
    print("=== REAL RADIOMICS DATA ANALYSIS ===\n")
    
    # Load and combine data
    df = load_and_combine_radiomics_data()
    
    # Create statistical summary
    summary = create_statistical_summary(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Main comprehensive visualization
    fig1 = create_comprehensive_visualizations(df)
    fig1.savefig('real_radiomics_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    
    # Modality comparison
    fig2 = create_modality_comparison_plots(df)
    fig2.savefig('real_radiomics_modality_comparison.png', dpi=300, bbox_inches='tight')
    
    # Year comparison
    fig3 = create_year_comparison_plots(df)
    fig3.savefig('real_radiomics_year_comparison.png', dpi=300, bbox_inches='tight')
    
    print("\nVisualizations saved:")
    print("  - real_radiomics_comprehensive_analysis.png")
    print("  - real_radiomics_modality_comparison.png")
    print("  - real_radiomics_year_comparison.png")
    
    # Save combined data
    output_file = 'combined_radiomics_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nCombined data saved to: {output_file}")
    
    print("\nAnalysis completed successfully!")
    
    return df, summary

if __name__ == "__main__":
    main() 