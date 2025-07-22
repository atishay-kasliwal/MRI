import os
from pathlib import Path
from logging_utils import log_and_print
from dcm2niix_conversion import convert_dicom_to_nifti
from freesurfer_standardization import standardize_to_t1
from n4_bias_correction import n4_bias_correction
from masking import generate_mask
from io_utils import load_nifti
from visualization import plot_slice
import sys
import numpy as np
import csv
from pipeline_utils import is_valid_nifti

# Configuration
DATA_ROOT = Path('/Users/atishaykasliwal/Desktop/mri/data')
MODALITIES = ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']
BATCH_SIZE = 5

def check_file_exists(file_path, description=""):
    """Check if a file exists and is valid"""
    if file_path.exists():
        if file_path.suffix in ['.nii.gz', '.nii']:
            return is_valid_nifti(str(file_path))
        return True
    return False

def should_skip_conversion(patient_dir, modality):
    """Check if DICOM to NIfTI conversion should be skipped"""
    nifti_output = patient_dir / 'nifti_output'
    raw_folder = nifti_output / 'Raw'
    modality_files = list(raw_folder.glob(f'{modality}*.nii.gz'))
    return len(modality_files) > 0 and all(check_file_exists(f) for f in modality_files)

def should_skip_standardization(patient_dir):
    """Check if standardization should be skipped"""
    nifti_output = patient_dir / 'nifti_output'
    final_folder = nifti_output / 'Raw' / 'Final'
    std_files = list(final_folder.glob('*_in_T1.nii.gz'))
    return len(std_files) > 0 and all(check_file_exists(f) for f in std_files)

def should_skip_n4_correction(patient_dir, modality):
    """Check if N4 bias correction should be skipped"""
    correctedn4_folder = patient_dir / 'correctedn4'
    corrected_name = f'CORRECT{modality}_{patient_dir.name.replace(",", "").replace(" ", "_")}.nii.gz'
    corrected_path = correctedn4_folder / corrected_name
    return check_file_exists(corrected_path, f"N4 corrected {modality}")

def should_skip_mask_generation(patient_dir):
    """Check if mask generation should be skipped"""
    patient_name = patient_dir.name.replace(',', '').replace(' ', '_')
    nifti_output = patient_dir / 'nifti_output'
    final_folder = nifti_output / 'Raw' / 'Final'
    mask_folder = patient_dir / 'mask'
    mask_name = f'MASK_{patient_name}.nii.gz'
    mask_path_final = final_folder / mask_name
    mask_path_new = mask_folder / mask_name
    return check_file_exists(mask_path_final, "Final mask") and check_file_exists(mask_path_new, "New mask")

# Accept patient directories as command-line arguments for flexibility
if len(sys.argv) > 1:
    patient_names = sys.argv[1:]
    patient_dirs = [DATA_ROOT / name for name in patient_names if (DATA_ROOT / name).is_dir()]
    if not patient_dirs:
        log_and_print('No valid patient directories found for the provided arguments.', 'error')
        sys.exit(1)
else:
    # Sort patient directories in ascending order
    patient_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])

os.environ['FREESURFER_HOME'] = '/Applications/freesurfer'
os.environ['PATH'] = '/Applications/freesurfer/bin:' + os.environ['PATH']

results = []

# Progress tracking
total_patients = len(patient_dirs)
current_patient = 0

log_and_print(f'\n=== Starting MRI Processing Pipeline ===')
log_and_print(f'Total patients to process: {total_patients}')
log_and_print(f'Processing all patients with skip logic enabled...\n')

for patient_dir in patient_dirs:
    current_patient += 1
    patient_name = patient_dir.name.replace(',', '').replace(' ', '_')
    log_and_print(f'\n=== Processing patient {current_patient}/{total_patients}: {patient_name} ===')
    
    nifti_output = patient_dir / 'nifti_output'
    raw_folder = nifti_output / 'Raw'
    final_folder = raw_folder / 'Final'
    correctedn4_folder = patient_dir / 'correctedn4'
    mask_folder = patient_dir / 'mask'
    
    os.makedirs(final_folder, exist_ok=True)
    os.makedirs(correctedn4_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    # Step 1: DICOM to NIfTI conversion for each modality
    log_and_print(f'  Step 1/5: DICOM to NIfTI conversion...')
    for modality in MODALITIES:
        dicom_folder = patient_dir / modality
        modality_output = nifti_output / modality
        
        if dicom_folder.exists():
            if should_skip_conversion(patient_dir, modality):
                log_and_print(f'  {modality} NIfTI files already exist, skipping conversion.')
            else:
                convert_dicom_to_nifti(str(dicom_folder), str(modality_output), str(raw_folder), log_fn=log_and_print)
        else:
            log_and_print(f'  {modality} folder missing for {patient_name}', 'warning')

    # Step 2: FreeSurfer standardization to T1 space
    log_and_print(f'  Step 2/5: FreeSurfer standardization...')
    t1_files = list(raw_folder.glob('T1*.nii.gz'))
    if not t1_files:
        log_and_print(f'  No T1 NIfTI found for {patient_name}', 'error')
        continue
    
    if should_skip_standardization(patient_dir):
        log_and_print(f'  Standardized files already exist for {patient_name}, skipping standardization.')
    else:
        t1_reference = t1_files[0]
        all_nifti_files = list(raw_folder.glob('*.nii.gz'))
        standardize_to_t1(all_nifti_files, t1_reference, final_folder, log_fn=log_and_print)

    # Step 3: N4 bias correction for all standardized modalities
    log_and_print(f'  Step 3/5: N4 bias correction...')
    std_files = list(final_folder.glob('*_in_T1.nii.gz'))
    n4_corrected_paths = {}
    
    for std_file in std_files:
        modality = std_file.name.split('_')[0]
        corrected_name = f'CORRECT{modality}_{patient_name}.nii.gz'
        corrected_path = correctedn4_folder / corrected_name
        
        if should_skip_n4_correction(patient_dir, modality):
            log_and_print(f'  N4 corrected {modality} already exists, skipping correction.')
            n4_corrected_paths[modality] = corrected_path
        else:
            n4_bias_correction(std_file, corrected_path, log_fn=log_and_print)
            n4_corrected_paths[modality] = corrected_path

    # Step 4: Generate mask from corrected DWI
    log_and_print(f'  Step 4/5: Mask generation...')
    if 'DWI' in n4_corrected_paths:
        if should_skip_mask_generation(patient_dir):
            log_and_print(f'  Masks already exist for {patient_name}, skipping mask generation.')
        else:
            mask_name = f'MASK_{patient_name}.nii.gz'
            mask_path_final = final_folder / mask_name
            mask_path_new = mask_folder / mask_name
            generate_mask(n4_corrected_paths['DWI'], output_path=mask_path_final, log_fn=log_and_print)
            generate_mask(n4_corrected_paths['DWI'], output_path=mask_path_new, log_fn=log_and_print)

    # Step 5: Co-localization analysis (T1 vs all other modalities)
    log_and_print(f'  Step 5/5: Co-localization analysis...')
    if 'T1' in n4_corrected_paths:
        t1_img, t1_data = load_nifti(n4_corrected_paths['T1'])
        for modality, path in n4_corrected_paths.items():
            if modality == 'T1':
                continue
            mod_img, mod_data = load_nifti(path)
            # Flatten and mask out background (zero) voxels for fair comparison
            mask = (t1_data != 0) & (mod_data != 0)
            t1_flat = t1_data[mask].flatten()
            mod_flat = mod_data[mask].flatten()
            # A. Intensity overlap (Pearson correlation)
            if t1_flat.size > 0 and mod_flat.size > 0:
                pearson_corr = np.corrcoef(t1_flat, mod_flat)[0, 1]
            else:
                pearson_corr = np.nan
            # B. Thresholded mask overlap (mean threshold for now, customizable later)
            t1_thresh = t1_flat.mean()
            mod_thresh = mod_flat.mean()
            t1_mask = t1_flat > t1_thresh
            mod_mask = mod_flat > mod_thresh
            intersection = np.logical_and(t1_mask, mod_mask)
            union = np.logical_or(t1_mask, mod_mask)
            dice = 2. * intersection.sum() / (t1_mask.sum() + mod_mask.sum()) if (t1_mask.sum() + mod_mask.sum()) > 0 else np.nan
            jaccard = intersection.sum() / union.sum() if union.sum() > 0 else np.nan
            # Log and save result
            log_and_print(f'Co-localization {patient_name}: T1 vs {modality} | Pearson: {pearson_corr:.4f}, Dice: {dice:.4f}, Jaccard: {jaccard:.4f}')
            results.append([patient_name, modality, pearson_corr, dice, jaccard])
    
    log_and_print(f'  ✓ Completed patient {current_patient}/{total_patients}: {patient_name}')

# Save results to CSV
with open('colocalization_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Patient', 'Modality', 'PearsonCorr', 'Dice', 'Jaccard'])
    writer.writerows(results)

# Final progress summary
log_and_print(f'\n=== Pipeline Complete ===')
log_and_print(f'Successfully processed {total_patients} patients')
log_and_print(f'Co-localization results saved to: colocalization_results.csv')
log_and_print(f'Total co-localization analyses: {len(results)}') 