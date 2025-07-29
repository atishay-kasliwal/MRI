#!/usr/bin/env python3
"""
mRS Radiomics Extraction Pipeline
Extracts radiomics features for patients with mRS data from clinical follow-up
Processes 5 scans per patient (T1, T2, FLAIR, DWI, ADC) and merges into yearly sheets
"""

import os
import sys
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from radiomics import featureextractor
import SimpleITK as sitk
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

class MRSRadiomicsPipeline:
    """
    Comprehensive pipeline for extracting radiomics features from patients with mRS data
    """
    
    def __init__(self, mri_base_path="/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip", 
                 clinical_data_path="clinical folow up "):
        """
        Initialize the pipeline
        
        Args:
            mri_base_path: Path to the MRI scan data
            clinical_data_path: Path to clinical follow-up data
        """
        self.mri_base_path = Path(mri_base_path)
        self.clinical_data_path = Path(clinical_data_path)
        self.years = [2020, 2021, 2022, 2023, 2024]
        self.modalities = ['T1', 'T2', 'FLAIR', 'DWI', 'ADC']
        
        # Initialize PyRadiomics feature extractor
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        self.extractor.enableAllFeatures()
        
        # Store results for each year
        self.yearly_results = {}
        
        print("=== mRS RADIOMICS PIPELINE INITIALIZED ===")
        print(f"MRI Base Path: {self.mri_base_path}")
        print(f"Clinical Data Path: {self.clinical_data_path}")
        print(f"Processing Years: {self.years}")
        print(f"Modalities: {self.modalities}")
    
    def load_clinical_data(self):
        """Load clinical follow-up data for all years"""
        print("\nLoading clinical follow-up data...")
        
        clinical_data = {}
        
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
        
        for year in self.years:
            # Handle special case for 2022 file name
            if year == 2022:
                csv_file = self.clinical_data_path / f"MRI SCAN- MRN NUMBER.xlsx - Copy of {year}_Patients (1).csv"
            else:
                csv_file = self.clinical_data_path / f"MRI SCAN- MRN NUMBER.xlsx - Copy of {year}_Patients.csv"
            
            if csv_file.exists():
                try:
                    # 2024 file doesn't have headers, others do
                    if year == 2024:
                        df = pd.read_csv(csv_file, header=None)
                        df.columns = columns
                    else:
                        df = pd.read_csv(csv_file)
                    
                    clinical_data[year] = df
                    print(f"  ✓ Loaded {len(df)} patients from {year}")
                except Exception as e:
                    print(f"  ✗ Error loading {year}: {e}")
                    clinical_data[year] = pd.DataFrame()
            else:
                print(f"  ✗ File not found: {csv_file}")
                clinical_data[year] = pd.DataFrame()
        
        return clinical_data
    
    def extract_patient_ids_with_mrs(self, clinical_data):
        """
        Extract patient IDs that have mRS data (Last mRS or 90 days mRS)
        
        Args:
            clinical_data: Dictionary of clinical data by year
            
        Returns:
            Dictionary mapping year to list of patient IDs with mRS data
        """
        print("\nExtracting patient IDs with mRS data...")
        
        patients_with_mrs = {}
        
        for year, df in clinical_data.items():
            if df.empty:
                patients_with_mrs[year] = []
                continue
            
            # Look for mRS columns (case insensitive)
            mrs_columns = []
            for col in df.columns:
                col_lower = col.lower()
                if 'mrs' in col_lower and ('last' in col_lower or '90' in col_lower or 'discharge' in col_lower):
                    mrs_columns.append(col)
            
            print(f"  {year}: Found mRS columns: {mrs_columns}")
            
            # Extract patient IDs with valid mRS data
            valid_patients = []
            
            for idx, row in df.iterrows():
                has_mrs = False
                
                # Check if any mRS column has valid data
                for col in mrs_columns:
                    mrs_value = row[col]
                    if pd.notna(mrs_value) and mrs_value != '' and str(mrs_value).lower() not in ['na', 'nan', 'unknown']:
                        try:
                            mrs_float = float(mrs_value)
                            if 0 <= mrs_float <= 6:  # Valid mRS range
                                has_mrs = True
                                break
                        except (ValueError, TypeError):
                            continue
                
                if has_mrs:
                    # Extract patient ID from various possible column names
                    patient_id = None
                    
                    # Try different possible column names for patient ID
                    id_columns = ['MRN ANON', 'ANON MRN ', 'WF Identifier', 'MRN']
                    for id_col in id_columns:
                        if id_col in df.columns and pd.notna(row[id_col]):
                            patient_id = str(row[id_col]).strip()
                            break
                    
                    if patient_id:
                        valid_patients.append(patient_id)
            
            patients_with_mrs[year] = valid_patients
            print(f"  {year}: {len(valid_patients)} patients with valid mRS data")
        
        return patients_with_mrs
    
    def find_mri_data_path(self, patient_id, year):
        """
        Find the MRI data path for a given patient ID and year
        Handles the DE-IDENTIFIED structure in the MRI data
        
        Args:
            patient_id: Patient identifier
            year: Year of data
            
        Returns:
            Path to patient's MRI data or None if not found
        """
        # Handle different patient ID formats
        patient_id_clean = str(patient_id).replace('ANON', '').strip()
        
        # Try different possible paths based on year structure
        possible_paths = []
        
        # First try local data structure (single patient folder)
        possible_paths.append(self.mri_base_path / f"DE-IDENTIFIED, {patient_id_clean}.brainlab")
        
        # Then try year-specific paths
        if year == 2020:
            possible_paths.extend([
                self.mri_base_path / "2020" / f"DE-IDENTIFIED, {patient_id_clean}.brainlab",
                self.mri_base_path / "2020" / f"DE-IDENTIFIED, ANON{patient_id_clean}.brainlab"
            ])
        elif year == 2021:
            possible_paths.extend([
                self.mri_base_path / "2021" / f"DE-IDENTIFIED, {patient_id_clean}.brainlab",
                self.mri_base_path / "2021" / f"DE-IDENTIFIED, ANON{patient_id_clean}.brainlab"
            ])
        elif year == 2022:
            possible_paths.extend([
                self.mri_base_path / "2022" / "2022" / f"DE-IDENTIFIED, {patient_id_clean}.brainlab",
                self.mri_base_path / "2022" / "2022" / f"DE-IDENTIFIED, ANON{patient_id_clean}.brainlab"
            ])
        elif year == 2023:
            possible_paths.extend([
                self.mri_base_path / "2023" / "2023" / f"DE-IDENTIFIED, {patient_id_clean}.brainlab",
                self.mri_base_path / "2023" / "2023" / f"DE-IDENTIFIED, ANON{patient_id_clean}.brainlab"
            ])
        elif year == 2024:
            possible_paths.extend([
                self.mri_base_path / "2024" / "2024" / f"DE-IDENTIFIED, {patient_id_clean}.brainlab",
                self.mri_base_path / "2024" / "2024" / f"DE-IDENTIFIED, ANON{patient_id_clean}.brainlab"
            ])
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def find_nifti_files(self, patient_path):
        """
        Find NIfTI files for all modalities in patient directory
        
        Args:
            patient_path: Path to patient's MRI data
            
        Returns:
            Dictionary mapping modality to NIfTI file path
        """
        nifti_files = {}
        
        # Check Outcome directory first (for DE-IDENTIFIED structure)
        outcome_dir = patient_path / "Outcome"
        if not outcome_dir.exists():
            outcome_dir = patient_path / "outcome"  # Try lowercase
        
        if outcome_dir.exists():
            for modality in self.modalities:
                # Look for CORRECT modality files
                nii_files = list(outcome_dir.glob(f"CORRECT{modality}_*.nii.gz"))
                if nii_files:
                    nifti_files[modality] = nii_files[0]
        
        # If not found in Outcome, check nifti_output directory
        if not nifti_files:
            nifti_dir = patient_path / "nifti_output"
            if nifti_dir.exists():
                for modality in self.modalities:
                    modality_dir = nifti_dir / modality
                    if modality_dir.exists():
                        # Look for .nii.gz files
                        nii_files = list(modality_dir.glob("*.nii.gz"))
                        if nii_files:
                            # Use the first file found (usually the main one)
                            nifti_files[modality] = nii_files[0]
        
        # If not found in nifti_output, check correctedn4 directory
        if not nifti_files:
            corrected_dir = patient_path / "correctedn4"
            if corrected_dir.exists():
                for modality in self.modalities:
                    nii_files = list(corrected_dir.glob(f"CORRECT{modality}_*.nii.gz"))
                    if nii_files:
                        nifti_files[modality] = nii_files[0]
        
        return nifti_files
    
    def find_mask_file(self, patient_path):
        """
        Find mask file in patient directory
        
        Args:
            patient_path: Path to patient's MRI data
            
        Returns:
            Path to mask file or None if not found
        """
        # Check Outcome directory first (for DE-IDENTIFIED structure)
        outcome_dir = patient_path / "Outcome"
        if not outcome_dir.exists():
            outcome_dir = patient_path / "outcome"  # Try lowercase
        
        if outcome_dir.exists():
            # Look for mask files in Outcome directory
            mask_files = list(outcome_dir.glob("*mask*.nii.gz"))
            if mask_files:
                return mask_files[0]
        
        # Check mask directory
        mask_dir = patient_path / "mask"
        if mask_dir.exists():
            # Look for mask files
            mask_files = list(mask_dir.glob("*mask*.nii.gz"))
            if mask_files:
                return mask_files[0]
        
        # Check nifti_output directory
        nifti_dir = patient_path / "nifti_output"
        if nifti_dir.exists():
            mask_files = list(nifti_dir.glob("*mask*.nii.gz"))
            if mask_files:
                return mask_files[0]
        
        return None
    
    def extract_radiomics_features(self, image_path, mask_path):
        """
        Extract radiomics features from image and mask
        
        Args:
            image_path: Path to NIfTI image file
            mask_path: Path to NIfTI mask file
            
        Returns:
            Dictionary of radiomics features
        """
        try:
            # Extract features using PyRadiomics
            feature_vector = self.extractor.execute(str(image_path), str(mask_path))
            
            # Remove diagnostic features
            features = {k: v for k, v in feature_vector.items() 
                       if not k.startswith('diagnostics_')}
            
            return features
            
        except Exception as e:
            print(f"    Error extracting features: {e}")
            return None
    
    def process_patient(self, patient_id, year, clinical_data):
        """
        Process a single patient and extract radiomics features
        
        Args:
            patient_id: Patient identifier
            year: Year of data
            clinical_data: Clinical data for the year
            
        Returns:
            Dictionary with patient features and clinical data
        """
        # Find MRI data path
        mri_path = self.find_mri_data_path(patient_id, year)
        if not mri_path:
            return None
        
        # Check if the path contains "DE-IDENTIFIED" (skip cases without this labeling)
        if "DE-IDENTIFIED" not in str(mri_path):
            return None
        
        # Find NIfTI files
        nifti_files = self.find_nifti_files(mri_path)
        if not nifti_files:
            return None
        
        # Find mask file
        mask_path = self.find_mask_file(mri_path)
        if not mask_path:
            return None
        
        # Extract clinical data for this patient
        patient_clinical = {}
        df = clinical_data[year]
        
        # Find the row for this patient
        patient_row = None
        for idx, row in df.iterrows():
            # Check different possible ID columns
            for id_col in ['MRN ANON', 'ANON MRN ', 'WF Identifier', 'MRN']:
                if id_col in df.columns:
                    row_id = str(row[id_col]).strip()
                    if row_id == str(patient_id).strip() or row_id == str(patient_id).replace('ANON', '').strip():
                        patient_row = row
                        break
            if patient_row is not None:
                break
        
        if patient_row is not None:
            # Extract all clinical features
            for col in df.columns:
                patient_clinical[col] = patient_row[col]
        
        # Extract radiomics features for each modality
        patient_features = {
            'PatientID': patient_id,
            'Year': year,
            **patient_clinical
        }
        
        available_modalities = []
        
        for modality, nifti_path in nifti_files.items():
            features = self.extract_radiomics_features(nifti_path, mask_path)
            if features:
                # Add modality prefix to feature names
                for feature_name, feature_value in features.items():
                    patient_features[f"{modality}_{feature_name}"] = feature_value
                
                available_modalities.append(modality)
        
        if available_modalities:
            patient_features['AvailableModalities'] = ','.join(available_modalities)
            return patient_features
        else:
            return None
    
    def process_year(self, year, clinical_data, patients_with_mrs):
        """
        Process all patients for a given year
        
        Args:
            year: Year to process
            clinical_data: Clinical data for all years
            patients_with_mrs: Dictionary of patients with mRS data by year
            
        Returns:
            List of patient feature dictionaries
        """
        print(f"\nProcessing year {year}...")
        
        if year not in patients_with_mrs or not patients_with_mrs[year]:
            print(f"  No patients with mRS data for {year}")
            return []
        
        year_results = []
        total_patients = len(patients_with_mrs[year])
        successful_count = 0
        
        # Create progress bar
        with tqdm(total=total_patients, desc=f"Year {year}", unit="patient") as pbar:
            for patient_id in patients_with_mrs[year]:
                patient_features = self.process_patient(patient_id, year, clinical_data)
                if patient_features:
                    year_results.append(patient_features)
                    successful_count += 1
                
                # Update progress bar with success/failure stats
                pbar.set_postfix({
                    'Success': successful_count,
                    'Failed': pbar.n - successful_count + 1,
                    'Success Rate': f"{(successful_count/pbar.n)*100:.1f}%" if pbar.n > 0 else "0%"
                })
                pbar.update(1)
        
        print(f"  ✓ Successfully processed {len(year_results)}/{total_patients} patients for {year}")
        return year_results
    
    def save_yearly_results(self):
        """Save results for each year to CSV files"""
        print("\nSaving yearly results...")
        
        for year, results in self.yearly_results.items():
            if not results:
                print(f"  No results for {year}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Save to CSV
            output_file = f"mrs_radiomics_{year}.csv"
            df.to_csv(output_file, index=False)
            print(f"  ✓ Saved {len(df)} patients to {output_file}")
            
            # Print summary
            print(f"    Features per patient: {len(df.columns) - 3}")  # Exclude PatientID, Year, AvailableModalities
            print(f"    Available modalities: {df['AvailableModalities'].value_counts().to_dict()}")
    
    def create_combined_dataset(self):
        """Create a combined dataset with all years"""
        print("\nCreating combined dataset...")
        
        all_results = []
        for year, results in self.yearly_results.items():
            all_results.extend(results)
        
        if all_results:
            combined_df = pd.DataFrame(all_results)
            combined_file = "mrs_radiomics_combined.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"  ✓ Saved combined dataset with {len(combined_df)} patients to {combined_file}")
            
            # Print summary statistics
            print(f"    Total patients: {len(combined_df)}")
            print(f"    Years covered: {combined_df['Year'].unique()}")
            print(f"    Total features: {len(combined_df.columns) - 3}")
            
            # Show mRS distribution if available
            mrs_columns = [col for col in combined_df.columns if 'mrs' in col.lower()]
            if mrs_columns:
                print(f"    mRS columns found: {mrs_columns}")
                for col in mrs_columns:
                    valid_mrs = combined_df[col].dropna()
                    # Convert to numeric, ignoring errors
                    valid_mrs_numeric = pd.to_numeric(valid_mrs, errors='coerce').dropna()
                    if len(valid_mrs_numeric) > 0:
                        print(f"      {col}: {len(valid_mrs_numeric)} valid values, range: {valid_mrs_numeric.min()}-{valid_mrs_numeric.max()}")
                    else:
                        print(f"      {col}: {len(valid_mrs)} values (non-numeric data)")
        else:
            print("  ✗ No results to combine")
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("=== STARTING mRS RADIOMICS PIPELINE ===")
        
        # Step 1: Load clinical data
        clinical_data = self.load_clinical_data()
        
        # Step 2: Extract patients with mRS data
        patients_with_mrs = self.extract_patient_ids_with_mrs(clinical_data)
        
        # Step 3: Process each year
        for year in self.years:
            year_results = self.process_year(year, clinical_data, patients_with_mrs)
            self.yearly_results[year] = year_results
        
        # Step 4: Save results
        self.save_yearly_results()
        
        # Step 5: Create combined dataset
        self.create_combined_dataset()
        
        print("\n=== mRS RADIOMICS PIPELINE COMPLETED ===")
        print("Generated files:")
        for year in self.years:
            if self.yearly_results[year]:
                print(f"  - mrs_radiomics_{year}.csv")
        print("  - mrs_radiomics_combined.csv")

def main():
    """Main function"""
    # Check if MRI data path exists
    mri_path = "/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip"
    if not os.path.exists(mri_path):
        print(f"Error: MRI data path not found: {mri_path}")
        print("Please ensure the external drive is connected and the path is correct.")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = MRSRadiomicsPipeline()
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 