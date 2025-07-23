from pathlib import Path
from manual_masking import manual_generate_mask_brightest

# Google Drive source
SRC_ROOT = Path("/Users/atishaykasliwal/Library/CloudStorage/GoogleDrive-atishay.kasliwal@stonybrook.edu/My Drive/MRI MASK DATA/2020")
# Local destination
DST_ROOT = Path("/Users/atishaykasliwal/Desktop/local_mri_masks/2020")
PERCENTILE = 98.5

for patient_dir in SRC_ROOT.iterdir():
    if not patient_dir.is_dir():
        continue
    nii_files = list((patient_dir / "nifti_output" / "Raw" / "Final").glob("CORRECTDWI_*.nii.gz"))
    if not nii_files:
        print(f"No N4-corrected DWI found for {patient_dir.name}")
        continue
    nii_path = nii_files[0]
    # Create corresponding local patient folder
    local_patient_dir = DST_ROOT / patient_dir.name / "mask"
    local_patient_dir.mkdir(parents=True, exist_ok=True)
    output_path = local_patient_dir / f"brightest_p{PERCENTILE}_mask_{patient_dir.name}.nii.gz"
    manual_generate_mask_brightest(nii_path, PERCENTILE, output_path)
    print(f"Mask saved locally: {output_path}") 