import os
from pathlib import Path
import re
from main import (
    log_and_print, convert_dicom_to_nifti, standardize_to_t1, n4_bias_correction, generate_mask, load_nifti, plot_slice
)

# --- CONFIG ---
DATA_ROOT = Path('data/2024')
LOG_PATH = 'your_log_file.txt'  # Change to your actual log file
MODALITIES = ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']
BATCH_SIZE = 5

# --- Step 1: Parse log for skipped patients ---
skipped_patients = set()
with open(LOG_PATH, 'r') as f:
    for line in f:
        match = re.search(r'No T1 NIfTI found for ([^\s]+)', line)
        if match:
            skipped_patients.add(match.group(1))
        match2 = re.search(r'T1 folder missing for ([^\s]+)', line)
        if match2:
            skipped_patients.add(match2.group(1))

# --- Step 2: Filter for those that now exist ---
patient_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir() and d.name in skipped_patients and (d / 'T1').exists()]

# --- Step 3: Run pipeline for these patients in batches ---
for batch_start in range(0, len(patient_dirs), BATCH_SIZE):
    batch = patient_dirs[batch_start:batch_start+BATCH_SIZE]
    for patient_dir in batch:
        patient_name = patient_dir.name.replace(',', '').replace(' ', '_')
        log_and_print(f'\n=== Retrying patient: {patient_name} ===')
        nifti_output = patient_dir / 'nifti_output'
        raw_folder = nifti_output / 'Raw'
        final_folder = raw_folder / 'Final'
        correctedn4_folder = patient_dir / 'correctedn4'
        mask_folder = patient_dir / 'mask'
        os.makedirs(final_folder, exist_ok=True)
        os.makedirs(correctedn4_folder, exist_ok=True)
        os.makedirs(mask_folder, exist_ok=True)

        # Step 1: DICOM to NIfTI conversion for each modality
        for modality in MODALITIES:
            dicom_folder = patient_dir / modality
            modality_output = nifti_output / modality
            if dicom_folder.exists():
                convert_dicom_to_nifti(str(dicom_folder), str(modality_output), str(raw_folder), log_fn=log_and_print)
            else:
                log_and_print(f'  {modality} folder missing for {patient_name}', 'warning')

        # Step 2: FreeSurfer standardization to T1 space
        t1_files = list(raw_folder.glob('T1*.nii.gz'))
        if not t1_files:
            log_and_print(f'  No T1 NIfTI found for {patient_name}', 'error')
            continue
        t1_reference = t1_files[0]
        all_nifti_files = list(raw_folder.glob('*.nii.gz'))
        standardize_to_t1(all_nifti_files, t1_reference, final_folder, log_fn=log_and_print)

        # Step 3: N4 bias correction for standardized DWI (if present)
        dwi_std_files = list(final_folder.glob('DWI*_in_T1.nii.gz'))
        for dwi_file in dwi_std_files:
            corrected_name = f'CORRECTDWI_{patient_name}.nii.gz'
            corrected_path_final = final_folder / corrected_name
            corrected_path_new = correctedn4_folder / corrected_name
            n4_bias_correction(dwi_file, corrected_path_final, log_fn=log_and_print)
            n4_bias_correction(dwi_file, corrected_path_new, log_fn=log_and_print)

            # Step 4: Generate mask from corrected DWI
            mask_name = f'MASK_{patient_name}.nii.gz'
            mask_path_final = final_folder / mask_name
            mask_path_new = mask_folder / mask_name
            generate_mask(corrected_path_final, output_path=mask_path_final, log_fn=log_and_print)
            generate_mask(corrected_path_new, output_path=mask_path_new, log_fn=log_and_print)

            # Step 5: Visualization (show a slice)
            img, data = load_nifti(corrected_path_final)
            plot_slice(data, slice_index=data.shape[2]//2, title=f'Corrected DWI {patient_name} (mid-slice)')
            img_mask, mask_data = load_nifti(mask_path_final)
            plot_slice(mask_data, slice_index=mask_data.shape[2]//2, title=f'Generated Mask {patient_name} (mid-slice)')
            break  # Only process the first DWI file for mask/visualization
    input(f"\nBatch {batch_start//BATCH_SIZE + 1} complete. Press Enter to continue to the next batch...") 