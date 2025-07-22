import subprocess
from pathlib import Path

def standardize_to_t1(nifti_files, t1_reference, output_folder, mri_convert_path='mri_convert', log_fn=print):
    """
    Standardize NIfTI files to T1 space using mri_convert --like.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for nifti_file in nifti_files:
        nifti_file = Path(nifti_file)
        out_name = nifti_file.stem + '_in_T1.nii.gz'
        out_path = output_folder / out_name
        cmd = [
            mri_convert_path,
            '--like', str(t1_reference),
            str(nifti_file),
            str(out_path)
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if out_path.exists():
                log_fn(f"Standardized: {nifti_file.name} -> {out_path.name}")
            else:
                log_fn(f"Output not created: {out_path}", 'error')
        except subprocess.CalledProcessError as e:
            log_fn(f"Failed: {nifti_file.name}")
            log_fn(f"    Error: {e.stderr}", 'error') 