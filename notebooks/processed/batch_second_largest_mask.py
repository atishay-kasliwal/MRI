import sys
from pathlib import Path
import argparse
sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts'))  # Ensure scripts/ is in path
from masking import generate_mask

parser = argparse.ArgumentParser(description='Batch generate second largest masks.')
parser.add_argument('--data_root', type=str, required=True, help='Root directory for patient data')
parser.add_argument('--percentiles', type=float, nargs='+', required=True, help='List of percentiles')
parser.add_argument('--n_values', type=int, nargs='+', required=False, help='List of N values (unused, for compatibility)')
args = parser.parse_args()

DATA_ROOT = Path(args.data_root)
PERCENTILES = args.percentiles

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
    for percentile in PERCENTILES:
        output_path = mask_dir / f"second_largest_p{percentile}_mask_{patient_dir.name}.nii.gz"
        generate_mask(nii_path, output_path=output_path, component=2, threshold_percentile=percentile) 