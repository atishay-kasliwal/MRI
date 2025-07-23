import shutil
from pathlib import Path
import subprocess
import time
import sys

# Set your local data root
DATA_ROOT = Path('/Volumes/Kasliwal V1.1/MRi Data V1.2/2023')  # Change year as needed
PERCENTILES = [80,80.5,81,81.5,82,82.5,83,83.5,84,84.5,85,85.5,86,86.5,87,87.5,88,88.5,89,89.5,90,90.5,91,91.5,92,92.5,93,93.5,94,94.5,95,95.5,96,96.5,97,97.5,98,98.5,99,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9]
N_VALUES = [1, 2, 3]  # You can add more values if needed

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent
BATCH_DIR = SCRIPT_DIR  # All batch scripts are in notebooks/processed

# Delete all mask folders for a clean start
print("Deleting all previous mask files...")
for patient_dir in DATA_ROOT.iterdir():
    mask_dir = patient_dir / "mask"
    if mask_dir.exists() and mask_dir.is_dir():
        shutil.rmtree(mask_dir)
        print(f"Deleted {mask_dir}")

# Now run the batch scripts as before
print("Running batch_n_largest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_n_largest_mask.py"),
    "--data_root", str(DATA_ROOT),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(5)

print("Running batch_second_largest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_second_largest_mask.py"),
    "--data_root", str(DATA_ROOT),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(5)

print("Running batch_third_largest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_third_largest_mask.py"),
    "--data_root", str(DATA_ROOT),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(5)

print("Running batch_brightest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_brightest_mask.py"),
    "--data_root", str(DATA_ROOT),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(5)

print("Running batch_allregions_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_allregions_mask.py"),
    "--data_root", str(DATA_ROOT),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(5)

print("All batch mask generation scripts completed.") 