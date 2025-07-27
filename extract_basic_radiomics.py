#!/usr/bin/env python3
"""
Basic Radiomics Feature Extraction
Extracts basic radiomic features from MRI data using numpy, scipy, and SimpleITK
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

def main():
    """Main function to extract radiomics features"""
    
    # Accept root directory as argument
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1])
    else:
        ROOT = Path('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/2020')
    
    MODALITIES = ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']
    
    # Output CSV
    year = ROOT.name if ROOT.name.isdigit() else ROOT.parts[-1]
    OUTPUT_CSV = ROOT / f'all_patients_basic_radiomics_{year}.csv'
    
    results = []
    header = ['PatientID', 'Modality']
    
    print(f"Processing directory: {ROOT}")
    print(f"Output file: {OUTPUT_CSV}")
    
    # Count total patients
    patient_dirs = [d for d in ROOT.iterdir() if d.is_dir()]
    total_patients = len(patient_dirs)
    print(f"Found {total_patients} patient directories")
    
    for i, patient_dir in enumerate(patient_dirs):
        print(f"Processing patient {i+1}/{total_patients}: {patient_dir.name}")
        
        outcome_dir = patient_dir / 'Outcome'
        if not outcome_dir.exists():
            print(f"  No Outcome folder for {patient_dir.name}, skipping.")
            continue
        
        # Find mask (assume only one .nii.gz mask file)
        mask_files = list(outcome_dir.glob('*mask*.nii.gz'))
        if not mask_files:
            print(f"  No mask found in {outcome_dir}, skipping.")
            continue
        mask_path = mask_files[0]
        
        for modality in MODALITIES:
            img_files = list(outcome_dir.glob(f'CORRECT{modality}_*.nii.gz'))
            if not img_files:
                print(f"  No N4-corrected {modality} for {patient_dir.name}, skipping.")
                continue
            img_path = img_files[0]
            
            try:
                feature_vector = extract_radiomics_features(img_path, mask_path)
                if feature_vector is None:
                    print(f"  Failed to extract features for {patient_dir.name} - {modality}")
                    continue
                
                if len(header) == 2:
                    header += list(feature_vector.keys())
                
                row = [patient_dir.name, modality] + [feature_vector[k] for k in header[2:]]
                results.append(row)
                print(f"  ✓ Extracted features for {patient_dir.name} - {modality}")
                
            except Exception as e:
                print(f"  ERROR for {patient_dir.name} - {modality}: {e}")
                continue
    
    # Write results to CSV
    if results:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(results)
        
        print(f"\nAll radiomics features saved to {OUTPUT_CSV}")
        print(f"Total feature vectors extracted: {len(results)}")
        print(f"Features per modality: {len(results) // len(MODALITIES) if len(results) > 0 else 0}")
    else:
        print("No features extracted. Check the data structure and file paths.")

if __name__ == "__main__":
    main() 