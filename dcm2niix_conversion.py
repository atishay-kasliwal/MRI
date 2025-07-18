import os
import subprocess
import glob
import shutil

def convert_dicom_to_nifti(input_folder, output_folder, raw_folder=None, log_fn=print):
    """
    Convert DICOM in input_folder to NIfTI in output_folder using dcm2niix.
    Optionally copy .nii.gz files to raw_folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    result = subprocess.run([
        "dcm2niix",
        "-z", "y",
        "-o", output_folder,
        input_folder
    ], capture_output=True, text=True)
    if result.returncode != 0:
        log_fn(f"dcm2niix failed for {input_folder}: {result.stderr}")
    else:
        log_fn(f"dcm2niix success for {input_folder}")
    if raw_folder:
        os.makedirs(raw_folder, exist_ok=True)
        nii_files = glob.glob(os.path.join(output_folder, "*.nii.gz"))
        for nii_file in nii_files:
            shutil.copy(nii_file, raw_folder) 