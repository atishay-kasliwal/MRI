from pathlib import Path
from n_largest_masking import manual_generate_mask_n_largest

DATA_ROOT = Path("/Users/atishaykasliwal/Desktop/mri/data/2021")  # Change year as needed
PERCENTILES = [99, 99.5]
N_VALUES = [1, 2, 3]  # You can add more values if needed

for patient_dir in DATA_ROOT.iterdir():
    if not patient_dir.is_dir():
        continue
    nii_files = list((patient_dir / "nifti_output" / "Raw" / "Final").glob("CORRECTDWI_*.nii.gz"))
    if not nii_files:
        print(f"No N4-corrected DWI found for {patient_dir.name}")
        continue
    nii_path = nii_files[0]
    mask_dir = patient_dir / "mask"
    mask_dir.mkdir(exist_ok=True)
    for n in N_VALUES:
        for percentile in PERCENTILES:
            output_path = mask_dir / f"{n}largest_p{percentile}_mask_{patient_dir.name}.nii.gz"
            manual_generate_mask_n_largest(nii_path, percentile, n, output_path) 