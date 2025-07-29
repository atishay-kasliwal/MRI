#!/usr/bin/env python3
"""
Dataset Summary
Shows comprehensive statistics for the merged radiomics-clinical dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

def print_dataset_summary():
    """Print comprehensive dataset summary"""
    
    print("=== FINAL DATASET SUMMARY ===")
    
    # Check if merged file exists
    merged_file = "merged_radiomics_clinical_data.csv"
    if not Path(merged_file).exists():
        print(f"Error: {merged_file} not found. Please run the merge script first.")
        return
    
    # Load merged dataset
    df = pd.read_csv(merged_file)
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"  Total patients: {len(df)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Years covered: {sorted(df['Year'].unique())}")
    
    # Feature breakdown
    radiomics_cols = [col for col in df.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    clinical_cols = [col for col in df.columns if col not in radiomics_cols and col not in ['PatientID', 'Year', 'AvailableModalities']]
    
    print(f"\n🔬 FEATURE BREAKDOWN")
    print(f"  Radiomics features: {len(radiomics_cols)}")
    print(f"  Clinical features: {len(clinical_cols)}")
    print(f"  Metadata features: 3 (PatientID, Year, AvailableModalities)")
    
    # Modality breakdown
    print(f"\n📈 MODALITY BREAKDOWN")
    modality_counts = df['AvailableModalities'].value_counts()
    for modalities, count in modality_counts.items():
        print(f"  {modalities}: {count} patients")
    
    # Year breakdown
    print(f"\n📅 YEAR BREAKDOWN")
    year_counts = df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count} patients")
    
    # mRS Analysis
    print(f"\n🏥 mRS ANALYSIS")
    mrs_columns = [col for col in df.columns if 'mrs' in col.lower()]
    for col in mrs_columns:
        valid_mrs = df[col].dropna()
        valid_mrs_numeric = pd.to_numeric(valid_mrs, errors='coerce').dropna()
        if len(valid_mrs_numeric) > 0:
            print(f"  {col}:")
            print(f"    Valid values: {len(valid_mrs_numeric)}")
            print(f"    Range: {valid_mrs_numeric.min()}-{valid_mrs_numeric.max()}")
            print(f"    Mean: {valid_mrs_numeric.mean():.2f}")
            print(f"    Distribution: {valid_mrs_numeric.value_counts().sort_index().to_dict()}")
    
    # Clinical features summary
    print(f"\n💊 CLINICAL FEATURES SUMMARY")
    print(f"  Available clinical features: {len(clinical_cols)}")
    
    # Show some key clinical features
    key_clinical = ['Sex', 'Age', 'Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    for feature in key_clinical:
        if feature in df.columns:
            valid_values = df[feature].dropna()
            if len(valid_values) > 0:
                print(f"  {feature}: {len(valid_values)} valid values")
    
    # Radiomics features summary
    print(f"\n🔬 RADIOMICS FEATURES SUMMARY")
    print(f"  Total radiomics features: {len(radiomics_cols)}")
    
    # Count features by modality
    modality_feature_counts = {}
    for col in radiomics_cols:
        for modality in ['T1', 'T2', 'FLAIR', 'DWI', 'ADC']:
            if col.startswith(f"{modality}_"):
                modality_feature_counts[modality] = modality_feature_counts.get(modality, 0) + 1
                break
    
    for modality, count in modality_feature_counts.items():
        print(f"  {modality}: {count} features")
    
    # Data quality
    print(f"\n✅ DATA QUALITY")
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    print(f"  Data completeness: {completeness:.1f}%")
    print(f"  Missing values: {missing_cells:,} out of {total_cells:,} cells")
    
    # File sizes
    print(f"\n💾 FILE SIZES")
    files = ["mrs_radiomics_combined.csv", "merged_radiomics_clinical_data.csv"]
    for file in files:
        if Path(file).exists():
            size_mb = Path(file).stat().st_size / (1024 * 1024)
            print(f"  {file}: {size_mb:.1f} MB")
    
    print(f"\n🎉 MERGE COMPLETED SUCCESSFULLY!")
    print(f"Your final dataset is ready for analysis: {merged_file}")

if __name__ == "__main__":
    print_dataset_summary() 