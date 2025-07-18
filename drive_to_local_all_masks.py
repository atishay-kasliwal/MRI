import time
from pathlib import Path
from manual_masking import manual_generate_mask_brightest
from n_largest_masking import manual_generate_mask_n_largest
from allregions_masking import manual_generate_mask_all_regions

# Google Drive source
SRC_ROOT = Path("/Users/atishaykasliwal/Library/CloudStorage/GoogleDrive-atishay.kasliwal@stonybrook.edu/My Drive/MRI MASK DATA/2020")
# Local destination
DST_ROOT = Path("/Users/atishaykasliwal/Desktop/local_mri_masks/2020")

BRIGHTEST_PERCENTILES = [75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97, 97.5, 98, 98.5, 99, 99.5, 99.9]
N_LARGEST_VALUES = [1, 2, 3]
N_LARGEST_PERCENTILES = [75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97, 97.5, 98, 98.5, 99, 99.5, 99.9]
ALL_REGIONS_PERCENTILES = N_LARGEST_PERCENTILES

for patient_dir in SRC_ROOT.iterdir():
    if not patient_dir.is_dir():
        continue
    nii_files = list((patient_dir / "nifti_output" / "Raw" / "Final").glob("CORRECTDWI_*.nii.gz"))
    if not nii_files:
        print(f"No N4-corrected DWI found for {patient_dir.name}")
        continue
    nii_path = nii_files[0]
    local_patient_dir = DST_ROOT / patient_dir.name / "mask"
    local_patient_dir.mkdir(parents=True, exist_ok=True)

    # Brightest masks
    for percentile in BRIGHTEST_PERCENTILES:
        output_path = local_patient_dir / f"brightest_p{percentile}_mask_{patient_dir.name}.nii.gz"
        manual_generate_mask_brightest(nii_path, percentile, output_path)
        time.sleep(2)

    # N-largest masks
    for n in N_LARGEST_VALUES:
        for percentile in N_LARGEST_PERCENTILES:
            output_path = local_patient_dir / f"{n}largest_p{percentile}_mask_{patient_dir.name}.nii.gz"
            manual_generate_mask_n_largest(nii_path, percentile, n, output_path)
            time.sleep(2)

    # All-regions masks
    for percentile in ALL_REGIONS_PERCENTILES:
        output_path = local_patient_dir / f"all_p{percentile}_mask_{patient_dir.name}.nii.gz"
        manual_generate_mask_all_regions(nii_path, percentile, output_path)
        time.sleep(2)

    print(f"All masks saved locally for {patient_dir.name}")

print("All mask generation completed and saved locally.") 