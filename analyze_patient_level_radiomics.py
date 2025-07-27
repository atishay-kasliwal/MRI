#!/usr/bin/env python3
"""
Analyze Patient-Level Radiomics Data
Analyzes the patient-level radiomics features that combine all MRI modalities per patient
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_and_combine_patient_level_data():
    """Load and combine patient-level radiomics data from all years"""
    
    # Paths to patient-level radiomics files
    radiomics_2020 = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2020/patient_level_radiomics_2020.csv')
    radiomics_2021 = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2021/patient_level_radiomics_2021.csv')
    radiomics_2022 = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2022/2022/patient_level_radiomics_2022.csv')
    
    # Load data
    print("Loading patient-level radiomics data...")
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
    print(f"Total patients: {combined_df['PatientID'].nunique()}")
    
    return combined_df

def create_patient_level_visualizations(df):
    """Create comprehensive visualizations of the patient-level radiomics data"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Patient distribution by year
    ax1 = plt.subplot(3, 4, 1)
    year_counts = df['Year'].value_counts().sort_index()
    year_counts.plot(kind='bar', ax=ax1)
    plt.title('Patient Distribution by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Patients')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    # 2. Feature correlation heatmap (top features)
    ax2 = plt.subplot(3, 4, 2)
    # Select numeric features for correlation
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f not in ['Year']]
    
    # Calculate correlation matrix for top features
    top_features = ['T1_mean', 'DWI_mean', 'ADC_mean', 'FLAIR_mean', 'T2_mean',
                   'T1_volume', 'cross_modality_mean_mean', 'cross_modality_entropy_mean']
    available_features = [f for f in top_features if f in numeric_features]
    
    if len(available_features) > 1:
        correlation_matrix = df[available_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax2, cbar_kws={'shrink': 0.8})
        plt.title('Top Feature Correlation Matrix')
    
    # 3. Volume distribution by year
    ax3 = plt.subplot(3, 4, 3)
    if 'T1_volume' in df.columns:
        df.boxplot(column='T1_volume', by='Year', ax=ax3)
        plt.title('Tumor Volume Distribution by Year')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 4. Mean intensity by modality
    ax4 = plt.subplot(3, 4, 4)
    modality_means = []
    modality_names = []
    for modality in ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']:
        feature_name = f'{modality}_mean'
        if feature_name in df.columns:
            modality_means.append(df[feature_name].mean())
            modality_names.append(modality)
    
    if modality_means:
        plt.bar(modality_names, modality_means)
        plt.title('Average Mean Intensity by Modality')
        plt.xlabel('Modality')
        plt.ylabel('Mean Intensity')
        plt.grid(True, alpha=0.3)
    
    # 5. Cross-modality mean intensity
    ax5 = plt.subplot(3, 4, 5)
    if 'cross_modality_mean_mean' in df.columns:
        df.boxplot(column='cross_modality_mean_mean', by='Year', ax=ax5)
        plt.title('Cross-Modality Mean Intensity by Year')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 6. Cross-modality entropy
    ax6 = plt.subplot(3, 4, 6)
    if 'cross_modality_entropy_mean' in df.columns:
        df.boxplot(column='cross_modality_entropy_mean', by='Year', ax=ax6)
        plt.title('Cross-Modality Entropy by Year')
        plt.suptitle('')  # Remove default title
        plt.grid(True, alpha=0.3)
    
    # 7. Modality availability
    ax7 = plt.subplot(3, 4, 7)
    modality_availability = []
    modality_names = []
    for modality in ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']:
        feature_name = f'has_{modality}'
        if feature_name in df.columns:
            availability = df[feature_name].mean() * 100
            modality_availability.append(availability)
            modality_names.append(modality)
    
    if modality_availability:
        plt.bar(modality_names, modality_availability)
        plt.title('Modality Availability (%)')
        plt.xlabel('Modality')
        plt.ylabel('Availability (%)')
        plt.grid(True, alpha=0.3)
    
    # 8. Number of modalities per patient
    ax8 = plt.subplot(3, 4, 8)
    if 'num_modalities' in df.columns:
        df['num_modalities'].value_counts().sort_index().plot(kind='bar', ax=ax8)
        plt.title('Number of Modalities per Patient')
        plt.xlabel('Number of Modalities')
        plt.ylabel('Number of Patients')
        plt.grid(True, alpha=0.3)
    
    # 9. Feature importance (variance across patients)
    ax9 = plt.subplot(3, 4, 9)
    if len(available_features) > 0:
        # Calculate feature variance across patients
        feature_variance = []
        feature_names = []
        
        for feature in available_features:
            variance = df[feature].var()
            feature_variance.append(variance)
            feature_names.append(feature)
        
        if feature_variance:
            # Sort by variance
            sorted_indices = np.argsort(feature_variance)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices]
            sorted_variance = [feature_variance[i] for i in sorted_indices]
            
            plt.barh(range(len(sorted_features)), sorted_variance)
            plt.yticks(range(len(sorted_features)), [f.split('_')[0] for f in sorted_features])
            plt.xlabel('Variance Across Patients')
            plt.title('Feature Discriminative Power')
            plt.grid(True, alpha=0.3)
    
    # 10. T1 vs DWI mean intensity scatter
    ax10 = plt.subplot(3, 4, 10)
    if 'T1_mean' in df.columns and 'DWI_mean' in df.columns:
        plt.scatter(df['T1_mean'], df['DWI_mean'], alpha=0.6)
        plt.xlabel('T1 Mean Intensity')
        plt.ylabel('DWI Mean Intensity')
        plt.title('T1 vs DWI Mean Intensity')
        plt.grid(True, alpha=0.3)
    
    # 11. Volume vs cross-modality entropy
    ax11 = plt.subplot(3, 4, 11)
    if 'T1_volume' in df.columns and 'cross_modality_entropy_mean' in df.columns:
        plt.scatter(df['T1_volume'], df['cross_modality_entropy_mean'], alpha=0.6)
        plt.xlabel('Tumor Volume')
        plt.ylabel('Cross-Modality Entropy')
        plt.title('Volume vs Cross-Modality Entropy')
        plt.grid(True, alpha=0.3)
    
    # 12. Year vs cross-modality features
    ax12 = plt.subplot(3, 4, 12)
    if 'cross_modality_mean_mean' in df.columns and 'cross_modality_entropy_mean' in df.columns:
        for year in df['Year'].unique():
            year_data = df[df['Year'] == year]
            plt.scatter(year_data['cross_modality_mean_mean'], 
                       year_data['cross_modality_entropy_mean'], 
                       alpha=0.6, label=f'Year {year}')
        plt.xlabel('Cross-Modality Mean')
        plt.ylabel('Cross-Modality Entropy')
        plt.title('Cross-Modality Features by Year')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_modality_comparison_plots(df):
    """Create detailed modality comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Features to compare across modalities
    features_to_compare = ['mean', 'std', 'entropy', 'volume', 'skewness', 'kurtosis']
    
    for i, feature in enumerate(features_to_compare):
        row = i // 3
        col = i % 3
        
        # Collect data for each modality
        modality_data = []
        modality_names = []
        
        for modality in ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']:
            feature_name = f'{modality}_{feature}'
            if feature_name in df.columns:
                modality_data.append(df[feature_name].values)
                modality_names.append(modality)
        
        if modality_data:
            axes[row, col].boxplot(modality_data, labels=modality_names)
            axes[row, col].set_title(f'{feature.capitalize()} by Modality')
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_cross_modality_analysis(df):
    """Create cross-modality analysis plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Cross-modality features
    cross_features = ['cross_modality_mean_mean', 'cross_modality_std_mean', 
                     'cross_modality_entropy_mean', 'cross_modality_energy_mean',
                     'cross_modality_mean_std', 'cross_modality_mean_range']
    
    for i, feature in enumerate(cross_features):
        row = i // 3
        col = i % 3
        
        if feature in df.columns:
            df.boxplot(column=feature, by='Year', ax=axes[row, col])
            axes[row, col].set_title(f'{feature.replace("_", " ").title()} by Year')
            axes[row, col].set_xlabel('Year')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_statistical_summary(df):
    """Create statistical summary of the patient-level data"""
    
    print("\n=== PATIENT-LEVEL RADIOMICS DATA STATISTICAL SUMMARY ===\n")
    
    # Basic statistics
    print("Dataset Overview:")
    print(f"Total patients: {len(df)}")
    print(f"Years: {sorted(df['Year'].unique())}")
    print(f"Features per patient: {len(df.columns) - 2}")  # Exclude PatientID and Year
    
    print("\nPatients by Year:")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} patients")
    
    # Modality availability
    print("\nModality Availability:")
    for modality in ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']:
        feature_name = f'has_{modality}'
        if feature_name in df.columns:
            availability = df[feature_name].mean() * 100
            print(f"  {modality}: {availability:.1f}%")
    
    # Number of modalities
    if 'num_modalities' in df.columns:
        print(f"\nAverage modalities per patient: {df['num_modalities'].mean():.1f}")
        print(f"Modalities distribution:")
        modality_dist = df['num_modalities'].value_counts().sort_index()
        for num_mod, count in modality_dist.items():
            print(f"  {num_mod} modalities: {count} patients")
    
    # Feature statistics
    print("\nFeature Statistics (across all patients):")
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [f for f in numeric_features if f not in ['Year']]
    
    # Group features by category
    feature_categories = {}
    for feature in numeric_features:
        if feature.startswith('cross_modality_'):
            category = 'cross_modality'
        elif feature.startswith('has_') or feature == 'num_modalities':
            category = 'modality_info'
        else:
            modality = feature.split('_')[0]
            category = f'{modality}_features'
        
        if category not in feature_categories:
            feature_categories[category] = []
        feature_categories[category].append(feature)
    
    for category, features in feature_categories.items():
        print(f"\n{category.replace('_', ' ').title()} ({len(features)} features):")
        for feature in features[:5]:  # Show first 5 features
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            print(f"  {feature}: mean={mean_val:.2f}, std={std_val:.2f}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more features")
    
    return {
        'total_patients': len(df),
        'years': sorted(df['Year'].unique()),
        'year_counts': year_counts.to_dict(),
        'feature_categories': feature_categories,
        'total_features': len(numeric_features)
    }

def main():
    """Main function to analyze patient-level radiomics data"""
    
    print("=== PATIENT-LEVEL RADIOMICS DATA ANALYSIS ===\n")
    
    # Load and combine data
    df = load_and_combine_patient_level_data()
    
    # Create statistical summary
    summary = create_statistical_summary(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Main comprehensive visualization
    fig1 = create_patient_level_visualizations(df)
    fig1.savefig('patient_level_radiomics_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    
    # Modality comparison
    fig2 = create_modality_comparison_plots(df)
    fig2.savefig('patient_level_modality_comparison.png', dpi=300, bbox_inches='tight')
    
    # Cross-modality analysis
    fig3 = create_cross_modality_analysis(df)
    fig3.savefig('patient_level_cross_modality_analysis.png', dpi=300, bbox_inches='tight')
    
    print("\nVisualizations saved:")
    print("  - patient_level_radiomics_comprehensive_analysis.png")
    print("  - patient_level_modality_comparison.png")
    print("  - patient_level_cross_modality_analysis.png")
    
    # Save combined data
    output_file = 'combined_patient_level_radiomics_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\nCombined patient-level data saved to: {output_file}")
    
    print("\nAnalysis completed successfully!")
    
    return df, summary

if __name__ == "__main__":
    main() 