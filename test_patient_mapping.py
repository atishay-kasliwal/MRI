#!/usr/bin/env python3
"""
Test script to verify patient ID mapping between clinical data and MRI data
"""

import pandas as pd
from pathlib import Path

def test_patient_mapping():
    """Test patient ID mapping"""
    
    # Load 2024 clinical data
    clinical_file = "clinical folow up /MRI SCAN- MRN NUMBER.xlsx - Copy of 2024_Patients.csv"
    df = pd.read_csv(clinical_file, header=None)
    
    # Define column names based on 2023 structure
    columns = [
        'MRN ANON', 'True MRN', 'WF Identifier', 'MRN', 'Comment', 'Service date', 'Sex', 'Age', 'Race',
        'Diabetes', 'Hypertension', 'AFIB', 'Hyper-lipidemia', 'CHF', 'CAD', 'Hemoglobin A1c', 'Prior Stroke',
        'Smoking hx', 'Pre-admission anti-thrombotics', 'Baseline mRS', 'Year', 'Etiology', 'Location',
        'ADMIT NIH', 'Decision based on:', 'ASPECT', 'Tandem occlusion', 'Vessel 1', 'Vessel 2', 'Vessel 3',
        'IVTPA', 'Onset-to-puncture (min)', 'Anesthesia', 'Approach (fem/rad/other)', 'Angioplasty',
        'Technique (ADAPT/stentriever/solumbra/ot', 'Balloon guide used?', 'Device  1 name', 'Device 2 name',
        'Device 3 name', 'Stentriever used?', 'Balloon used?', 'Stent', 'Stent Name', 'Reason for stent',
        'IAtPA', 'Heparin dosing', 'Integrilin dosing (mg)', 'Pre/intra-procedural anti-thrombotics',
        'Attempts#', 'Time to Revasc (min)', 'Procedure time (min)', 'Final TICI ', 'Complications?',
        'Complications detail', 'Complication required treatment?', 'Distal embolization (yes/no?)',
        'Hemorrhagic transformation', 'Symptomatic ICH?', 'MRI seq', 'Post-procedure CT',
        'ECASS radiological criteria ', 'PH1/PH2 hemorrhage volume (cc)', 'SAH clot thickness', 'SAH pattern',
        'IVH', 'Craniectomy', '24 hr NIHSS', 'Discharge antithrombotics', 'Discharge NIHSS', 'Discharge mRS',
        'LOS', '90 day NIHSS', '90 days mRS', 'Days f/u', 'Last mRS', 'Comments', '', 'Onset', 'Revasc',
        'Day of week', 'Groin/Radial puncture time'
    ]
    
    df.columns = columns
    
    print("Testing patient ID mapping...")
    print(f"Total patients in 2024: {len(df)}")
    
    # Check MRI data path
    mri_base = Path("/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip")
    
    # Test first few patients
    for i, row in df.head(5).iterrows():
        anon_id = row['MRN ANON']
        numeric_id = anon_id.replace('ANON', '')
        
        print(f"\nPatient {i+1}:")
        print(f"  ANON ID: {anon_id}")
        print(f"  Numeric ID: {numeric_id}")
        
        # Check if MRI data exists
        mri_path = mri_base / "2024" / "2024" / f"{numeric_id}.brainlab"
        if mri_path.exists():
            print(f"  ✓ MRI data found: {mri_path}")
            
            # Check for NIfTI files
            nifti_dir = mri_path / "nifti_output"
            if nifti_dir.exists():
                modalities = ['T1', 'T2', 'FLAIR', 'DWI', 'ADC']
                for modality in modalities:
                    modality_dir = nifti_dir / modality
                    if modality_dir.exists():
                        nii_files = list(modality_dir.glob("*.nii.gz"))
                        print(f"    {modality}: {len(nii_files)} files")
            
            # Check for mask
            mask_dir = mri_path / "mask"
            if mask_dir.exists():
                mask_files = list(mask_dir.glob("*mask*.nii.gz"))
                print(f"    Mask: {len(mask_files)} files")
        else:
            print(f"  ✗ MRI data not found")
        
        # Check mRS data
        mrs_columns = ['Baseline mRS', 'Discharge mRS', '90 days mRS', 'Last mRS']
        for col in mrs_columns:
            if col in df.columns:
                mrs_value = row[col]
                if pd.notna(mrs_value) and str(mrs_value).strip() not in ['', 'NA', 'nan']:
                    print(f"  {col}: {mrs_value}")

if __name__ == "__main__":
    test_patient_mapping() 