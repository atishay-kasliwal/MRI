#!/usr/bin/env python3
"""
Patient-Level Radiomics Feature Extraction
Extracts radiomic features from all MRI modalities for each patient and combines them into a single feature vector
"""

import os
import sys
import csv
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from scipy import ndimage
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

def extract_basic_features(image_array, mask_array):
    """
    Extract basic radiomic features from image and mask arrays
    """
    features = {}
    
    # Ensure mask is binary
    mask_binary = (mask_array > 0).astype(np.uint8)
    
    # Get masked image
    masked_image = image_array * mask_binary
    
    # Remove zero values for statistics
    non_zero_values = masked_image[masked_image > 0]
    
    if len(non_zero_values) == 0:
        # Return default values if no valid pixels
        features.update({
            'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
            'median': 0.0, 'skewness': 0.0, 'kurtosis': 0.0,
            'energy': 0.0, 'entropy': 0.0, 'variance': 0.0,
            'volume': 0.0, 'surface_area': 0.0, 'compactness': 0.0,
            'sphericity': 0.0, 'eccentricity': 0.0
        })
        return features
    
    # First-order statistics
    features['mean'] = float(np.mean(non_zero_values))
    features['std'] = float(np.std(non_zero_values))
    features['min'] = float(np.min(non_zero_values))
    features['max'] = float(np.max(non_zero_values))
    features['median'] = float(np.median(non_zero_values))
    features['skewness'] = float(skew(non_zero_values))
    features['kurtosis'] = float(kurtosis(non_zero_values))
    features['variance'] = float(np.var(non_zero_values))
    
    # Energy and entropy
    features['energy'] = float(np.sum(non_zero_values**2))
    
    # Calculate histogram for entropy
    hist, bins = np.histogram(non_zero_values, bins=256, range=(features['min'], features['max']))
    hist = hist[hist > 0]  # Remove zero bins
    if len(hist) > 0:
        prob = hist / np.sum(hist)
        features['entropy'] = float(-np.sum(prob * np.log2(prob + 1e-10)))
    else:
        features['entropy'] = 0.0
    
    # Shape features
    # Volume (number of voxels)
    features['volume'] = float(np.sum(mask_binary))
    
    # Surface area (approximation using edge detection)
    edges = ndimage.binary_erosion(mask_binary) != mask_binary
    features['surface_area'] = float(np.sum(edges))
    
    # Compactness (volume^2/3 / surface_area)
    if features['surface_area'] > 0:
        features['compactness'] = float((features['volume'] ** (2/3)) / features['surface_area'])
    else:
        features['compactness'] = 0.0
    
    # Sphericity (approximation)
    if features['volume'] > 0:
        # Approximate sphericity as (π^(1/3) * (6*volume)^(2/3)) / surface_area
        features['sphericity'] = float((np.pi**(1/3) * (6*features['volume'])**(2/3)) / features['surface_area'])
    else:
        features['sphericity'] = 0.0
    
    # Eccentricity (approximation using bounding box)
    if features['volume'] > 0:
        # Find bounding box
        coords = np.where(mask_binary)
        if len(coords[0]) > 0:
            x_range = np.max(coords[0]) - np.min(coords[0])
            y_range = np.max(coords[1]) - np.min(coords[1])
            z_range = np.max(coords[2]) - np.min(coords[2]) if len(coords) > 2 else 1
            
            max_range = max(x_range, y_range, z_range)
            min_range = min(x_range, y_range, z_range)
            
            if max_range > 0:
                features['eccentricity'] = float(np.sqrt(1 - (min_range**2 / max_range**2)))
            else:
                features['eccentricity'] = 0.0
        else:
            features['eccentricity'] = 0.0
    else:
        features['eccentricity'] = 0.0
    
    return features

def extract_glcm_features(image_array, mask_array, distances=[1], angles=[0]):
    """
    Extract basic GLCM-like features
    """
    features = {}
    
    # Ensure mask is binary
    mask_binary = (mask_array > 0).astype(np.uint8)
    
    # Get masked image
    masked_image = image_array * mask_binary
    
    # Remove zero values
    non_zero_values = masked_image[masked_image > 0]
    
    if len(non_zero_values) == 0:
        features.update({
            'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0,
            'glcm_energy': 0.0, 'glcm_correlation': 0.0
        })
        return features
    
    # Simple texture features based on local variance
    # Calculate local variance using convolution
    kernel = np.ones((3, 3, 3)) / 27  # 3x3x3 averaging kernel
    local_mean = ndimage.convolve(masked_image, kernel, mode='constant', cval=0)
    local_var = ndimage.convolve(masked_image**2, kernel, mode='constant', cval=0) - local_mean**2
    
    # Texture features based on local variance
    local_var_values = local_var[mask_binary > 0]
    if len(local_var_values) > 0:
        features['glcm_contrast'] = float(np.mean(local_var_values))
        features['glcm_homogeneity'] = float(1.0 / (1.0 + np.mean(local_var_values)))
        features['glcm_energy'] = float(np.mean(local_var_values**2))
        features['glcm_correlation'] = float(np.corrcoef(masked_image.flatten(), local_mean.flatten())[0, 1] if len(masked_image.flatten()) > 1 else 0.0)
    else:
        features.update({
            'glcm_contrast': 0.0, 'glcm_homogeneity': 0.0,
            'glcm_energy': 0.0, 'glcm_correlation': 0.0
        })
    
    return features

def extract_radiomics_features(image_path, mask_path):
    """
    Extract radiomics features from image and mask files
    """
    try:
        # Load image and mask
        image = sitk.ReadImage(str(image_path))
        mask = sitk.ReadImage(str(mask_path))
        
        # Convert to numpy arrays
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        
        # Ensure same dimensions
        if image_array.shape != mask_array.shape:
            print(f"Shape mismatch: image {image_array.shape}, mask {mask_array.shape}")
            return None
        
        # Extract features
        features = {}
        
        # Basic features
        basic_features = extract_basic_features(image_array, mask_array)
        features.update(basic_features)
        
        # GLCM-like features
        glcm_features = extract_glcm_features(image_array, mask_array)
        features.update(glcm_features)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def extract_patient_level_features(patient_dir, modalities=['T1', 'DWI', 'ADC', 'FLAIR', 'T2']):
    """
    Extract features from all modalities for a single patient and combine them
    """
    outcome_dir = patient_dir / 'Outcome'
    if not outcome_dir.exists():
        print(f"No Outcome folder for {patient_dir.name}, skipping.")
        return None
    
    # Find mask (assume only one .nii.gz mask file)
    mask_files = list(outcome_dir.glob('*mask*.nii.gz'))
    if not mask_files:
        print(f"No mask found in {outcome_dir}, skipping.")
        return None
    mask_path = mask_files[0]
    
    # Extract features for each modality
    patient_features = {}
    available_modalities = []
    
    for modality in modalities:
        img_files = list(outcome_dir.glob(f'CORRECT{modality}_*.nii.gz'))
        if not img_files:
            print(f"No N4-corrected {modality} for {patient_dir.name}, skipping modality.")
            continue
        
        img_path = img_files[0]
        
        try:
            feature_vector = extract_radiomics_features(img_path, mask_path)
            if feature_vector is None:
                print(f"Failed to extract features for {patient_dir.name} - {modality}")
                continue
            
            # Add modality prefix to feature names
            for feature_name, feature_value in feature_vector.items():
                patient_features[f"{modality}_{feature_name}"] = feature_value
            
            available_modalities.append(modality)
            print(f"  ✓ Extracted features for {patient_dir.name} - {modality}")
            
        except Exception as e:
            print(f"ERROR for {patient_dir.name} - {modality}: {e}")
            continue
    
    # Add patient-level summary features
    if available_modalities:
        # Calculate cross-modality features
        for feature_base in ['mean', 'std', 'entropy', 'energy']:
            values = []
            for modality in available_modalities:
                feature_name = f"{modality}_{feature_base}"
                if feature_name in patient_features:
                    values.append(patient_features[feature_name])
            
            if values:
                patient_features[f"cross_modality_{feature_base}_mean"] = float(np.mean(values))
                patient_features[f"cross_modality_{feature_base}_std"] = float(np.std(values))
                patient_features[f"cross_modality_{feature_base}_range"] = float(np.max(values) - np.min(values))
        
        # Add modality availability flags
        for modality in modalities:
            patient_features[f"has_{modality}"] = 1.0 if modality in available_modalities else 0.0
        
        # Add number of available modalities
        patient_features['num_modalities'] = float(len(available_modalities))
        
        print(f"  ✓ Combined features for {patient_dir.name} ({len(available_modalities)} modalities)")
        return patient_features
    
    return None

def main():
    """Main function to extract patient-level radiomics features"""
    
    # Accept root directory as argument
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1])
    else:
        ROOT = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2020')
    
    MODALITIES = ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']
    
    # Output CSV
    year = ROOT.name if ROOT.name.isdigit() else ROOT.parts[-1]
    OUTPUT_CSV = ROOT / f'patient_level_radiomics_{year}.csv'
    
    results = []
    header = ['PatientID']
    
    print(f"Processing directory: {ROOT}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Modalities: {MODALITIES}")
    
    # Count total patients
    patient_dirs = [d for d in ROOT.iterdir() if d.is_dir()]
    total_patients = len(patient_dirs)
    print(f"Found {total_patients} patient directories")
    
    for i, patient_dir in enumerate(patient_dirs):
        print(f"Processing patient {i+1}/{total_patients}: {patient_dir.name}")
        
        patient_features = extract_patient_level_features(patient_dir, MODALITIES)
        
        if patient_features is None:
            print(f"  Failed to extract features for {patient_dir.name}")
            continue
        
        # Create row with patient ID and all features
        row = [patient_dir.name] + [patient_features.get(k, 0.0) for k in header[1:]]
        
        # If this is the first successful patient, update header
        if len(header) == 1:
            header += list(patient_features.keys())
            # Recreate row with proper order
            row = [patient_dir.name] + [patient_features.get(k, 0.0) for k in header[1:]]
        
        results.append(row)
    
    # Write results to CSV
    if results:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
        
        print(f"\nPatient-level radiomics features saved to {OUTPUT_CSV}")
        print(f"Total patients processed: {len(results)}")
        print(f"Total features per patient: {len(header) - 1}")
        print(f"Feature categories:")
        
        # Count features by category
        feature_categories = {}
        for feature in header[1:]:
            if feature.startswith('cross_modality_'):
                category = 'cross_modality'
            elif feature.startswith('has_') or feature == 'num_modalities':
                category = 'modality_info'
            else:
                modality = feature.split('_')[0]
                category = f'{modality}_features'
            
            feature_categories[category] = feature_categories.get(category, 0) + 1
        
        for category, count in feature_categories.items():
            print(f"  {category}: {count} features")
    else:
        print("No features extracted. Check the data structure and file paths.")

if __name__ == "__main__":
    main() 