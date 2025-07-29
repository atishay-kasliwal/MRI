import shutil
from pathlib import Path
import subprocess
import time
import sys

# Set your local data root
DATA_ROOT = Path('/Volumes/Kasliwal V1.1/MRi Data V1.2/2024')  # Change year as needed
PERCENTILES = [94,94.5,95,95.5,96,96.5,97,97.5,98,98.5,99,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9]
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