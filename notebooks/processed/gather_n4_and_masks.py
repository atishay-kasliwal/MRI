import os
import shutil
from pathlib import Path

# Source and destination root folders
DATA_ROOT = Path('data')
DEST_ROOT = Path('n4_and_masks')

# Patterns for N4-corrected and mask files
N4_PATTERN = 'CORRECT*.nii.gz'
MASK_PATTERN = '*mask*.nii.gz'

for year_dir in DATA_ROOT.iterdir():
    if not year_dir.is_dir() or not year_dir.name.isdigit():
        continue  # Skip non-year folders
    year = year_dir.name
    for patient_dir in year_dir.iterdir():
        if not patient_dir.is_dir() or not patient_dir.name.startswith('DE-IDENTIFIED'):
            continue
        dest_patient_dir = DEST_ROOT / year / patient_dir.name
        dest_patient_dir.mkdir(parents=True, exist_ok=True)
        # Find N4-corrected files
        n4_files = list(patient_dir.rglob(N4_PATTERN))
        # Find mask files
        mask_files = list(patient_dir.rglob(MASK_PATTERN))
        for f in n4_files + mask_files:
            dest_path = dest_patient_dir / f.name
            shutil.copy2(f, dest_path)
            print(f"Copied {f} -> {dest_path}") 