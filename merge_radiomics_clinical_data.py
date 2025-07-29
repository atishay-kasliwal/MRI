#!/usr/bin/env python3
"""
Merge Radiomics with Clinical Data
Merges extracted radiomics features with clinical follow-up data based on ANON patient IDs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def merge_radiomics_clinical_data():
    """
    Merge radiomics data with clinical data based on ANON patient IDs
    """
    print("=== MERGING RADIOMICS WITH CLINICAL DATA ===")
    
    # Load the combined radiomics dataset
    radiomics_file = "mrs_radiomics_combined.csv"
    if not Path(radiomics_file).exists():
        print(f"Error: {radiomics_file} not found. Please run the radiomics extraction first.")
        return
    
    print(f"Loading radiomics data from {radiomics_file}...")
    radiomics_df = pd.read_csv(radiomics_file)
    print(f"  ✓ Loaded {len(radiomics_df)} patients with radiomics features")
    
    # Load clinical data for all years
    clinical_data = {}
    clinical_path = Path("clinical folow up ")
    
    years = [2020, 2021, 2022, 2023, 2024]
    
    for year in years:
        # Handle special case for 2022 file name
        if year == 2022:
            csv_file = clinical_path / f"MRI SCAN- MRN NUMBER.xlsx - Copy of {year}_Patients (1).csv"
        else:
            csv_file = clinical_path / f"MRI SCAN- MRN NUMBER.xlsx - Copy of {year}_Patients.csv"
        
        if csv_file.exists():
            try:
                if year == 2024:
                    # 2024 has no headers
                    df = pd.read_csv(csv_file, header=None)
                    # Define column names based on 2023 structure
                    columns = [
                        'MRN ANON', 'True MRN', 'WF Identifier', 'MRN', 'Comment', 'Service date', 'Sex', 'Age', 'Race',
                        'Diabetes', 'Hypertension', 'AFIB', 'Hyper-lipidemia', 'CHF', 'CAD', 'Hemoglobin A1c', 'Prior Stroke',
                        'Smoking hx', 'Pre-admission anti-thrombotics', 'Baseline mRS', 'Year', 'Etiology', 'Location',
                        'ADMIT NIH', 'DISCHARGE NIH', '90 DAY NIH', 'LAST NIH', 'ADMIT GCS', 'DISCHARGE GCS', '90 DAY GCS',
                        'LAST GCS', 'ADMIT ASPECTS', 'DISCHARGE ASPECTS', '90 DAY ASPECTS', 'LAST ASPECTS', 'ADMIT CTA',
                        'DISCHARGE CTA', '90 DAY CTA', 'LAST CTA', 'ADMIT CTP', 'DISCHARGE CTP', '90 DAY CTP', 'LAST CTP',
                        'ADMIT DWI', 'DISCHARGE DWI', '90 DAY DWI', 'LAST DWI', 'ADMIT FLAIR', 'DISCHARGE FLAIR', '90 DAY FLAIR',
                        'LAST FLAIR', 'ADMIT T1', 'DISCHARGE T1', '90 DAY T1', 'LAST T1', 'ADMIT T2', 'DISCHARGE T2', '90 DAY T2',
                        'LAST T2', 'ADMIT T2*', 'DISCHARGE T2*', '90 DAY T2*', 'LAST T2*', 'ADMIT SWI', 'DISCHARGE SWI', '90 DAY SWI',
                        'LAST SWI', 'ADMIT MRA', 'DISCHARGE MRA', '90 DAY MRA', 'LAST MRA', 'ADMIT MRV', 'DISCHARGE MRV', '90 DAY MRV',
                        'LAST MRV', 'ADMIT CTA', 'DISCHARGE CTA', '90 DAY CTA', 'LAST CTA', 'ADMIT CTP', 'DISCHARGE CTP', '90 DAY CTP',
                        'LAST CTP', 'ADMIT DSA', 'DISCHARGE DSA', '90 DAY DSA', 'LAST DSA', 'ADMIT ECHO', 'DISCHARGE ECHO', '90 DAY ECHO',
                        'LAST ECHO', 'ADMIT EKG', 'DISCHARGE EKG', '90 DAY EKG', 'LAST EKG', 'ADMIT HOLTER', 'DISCHARGE HOLTER', '90 DAY HOLTER',
                        'LAST HOLTER', 'ADMIT TEE', 'DISCHARGE TEE', '90 DAY TEE', 'LAST TEE', 'ADMIT TTE', 'DISCHARGE TTE', '90 DAY TTE',
                        'LAST TTE', 'ADMIT CXR', 'DISCHARGE CXR', '90 DAY CXR', 'LAST CXR', 'ADMIT CT', 'DISCHARGE CT', '90 DAY CT',
                        'LAST CT', 'ADMIT MRI', 'DISCHARGE MRI', '90 DAY MRI', 'LAST MRI', 'ADMIT US', 'DISCHARGE US', '90 DAY US',
                        'LAST US', 'ADMIT XRAY', 'DISCHARGE XRAY', '90 DAY XRAY', 'LAST XRAY', 'ADMIT OTHER', 'DISCHARGE OTHER', '90 DAY OTHER',
                        'LAST OTHER', 'Discharge mRS', '90 days mRS', 'Last mRS'
                    ]
                    df.columns = columns[:len(df.columns)]
                else:
                    df = pd.read_csv(csv_file)
                
                clinical_data[year] = df
                print(f"  ✓ Loaded {len(df)} patients from {year}")
            except Exception as e:
                print(f"  ✗ Error loading {year}: {e}")
    
    # Create merged dataset
    print("\nMerging radiomics with clinical data...")
    
    merged_data = []
    
    for idx, radiomics_row in radiomics_df.iterrows():
        patient_id = radiomics_row['PatientID']
        year = radiomics_row['Year']
        
        # Find matching clinical data
        clinical_row = None
        if year in clinical_data:
            df = clinical_data[year]
            
            # Try to find the patient in clinical data
            for id_col in ['MRN ANON', 'ANON MRN ', 'WF Identifier', 'MRN']:
                if id_col in df.columns:
                    # Find exact match
                    matches = df[df[id_col] == patient_id]
                    if len(matches) > 0:
                        clinical_row = matches.iloc[0]
                        break
                    
                    # Try without ANON prefix
                    clean_id = str(patient_id).replace('ANON', '').strip()
                    matches = df[df[id_col] == clean_id]
                    if len(matches) > 0:
                        clinical_row = matches.iloc[0]
                        break
        
        if clinical_row is not None:
            # Merge radiomics with clinical data
            merged_row = radiomics_row.copy()
            
            # Add clinical features (avoid duplicates)
            for col in clinical_row.index:
                if col not in merged_row.index:
                    merged_row[col] = clinical_row[col]
            
            merged_data.append(merged_row)
        else:
            print(f"  ⚠ No clinical data found for {patient_id} ({year})")
    
    if merged_data:
        # Create merged DataFrame
        merged_df = pd.DataFrame(merged_data)
        
        # Save merged dataset
        output_file = "merged_radiomics_clinical_data.csv"
        merged_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Successfully merged {len(merged_df)} patients")
        print(f"✓ Saved merged dataset to {output_file}")
        
        # Print summary statistics
        print(f"\nDataset Summary:")
        print(f"  Total patients: {len(merged_df)}")
        print(f"  Total features: {len(merged_df.columns)}")
        print(f"  Years covered: {merged_df['Year'].unique()}")
        
        # Show feature breakdown
        radiomics_cols = [col for col in merged_df.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
        clinical_cols = [col for col in merged_df.columns if col not in radiomics_cols and col not in ['PatientID', 'Year', 'AvailableModalities']]
        
        print(f"  Radiomics features: {len(radiomics_cols)}")
        print(f"  Clinical features: {len(clinical_cols)}")
        
        # Show mRS distribution
        mrs_columns = [col for col in merged_df.columns if 'mrs' in col.lower()]
        if mrs_columns:
            print(f"\nmRS Distribution:")
            for col in mrs_columns:
                valid_mrs = merged_df[col].dropna()
                valid_mrs_numeric = pd.to_numeric(valid_mrs, errors='coerce').dropna()
                if len(valid_mrs_numeric) > 0:
                    print(f"  {col}: {len(valid_mrs_numeric)} valid values, range: {valid_mrs_numeric.min()}-{valid_mrs_numeric.max()}")
                else:
                    print(f"  {col}: {len(valid_mrs)} values (non-numeric data)")
        
        return merged_df
    else:
        print("✗ No data to merge")
        return None

if __name__ == "__main__":
    merge_radiomics_clinical_data() 