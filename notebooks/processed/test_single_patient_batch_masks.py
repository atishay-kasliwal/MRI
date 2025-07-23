import shutil
from pathlib import Path
import subprocess
import time
import sys

# Set your local data root
DATA_ROOT = Path('/Volumes/Kasliwal V1.1/MRi Data V1.2/2023')  # Change as needed
PERCENTILES = [98.9]  # Use a single percentile for quick test, or expand as needed
N_VALUES = [1, 2, 3]

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent
BATCH_DIR = SCRIPT_DIR  # All batch scripts are in notebooks/processed

# Find the first patient directory
patient_dirs = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
if not patient_dirs:
    print("No patient directories found.")
    exit(1)
patient_dir = patient_dirs[0]
print(f"Testing batch mask scripts for patient: {patient_dir.name}")

# Clean up mask folder for this patient
mask_dir = patient_dir / "mask"
if mask_dir.exists() and mask_dir.is_dir():
    shutil.rmtree(mask_dir)
    print(f"Deleted {mask_dir}")

# Run all batch scripts for this patient
print("Running batch_n_largest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_n_largest_mask.py"),
    "--data_root", str(patient_dir),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(1)

print("Running batch_second_largest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_second_largest_mask.py"),
    "--data_root", str(patient_dir),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(1)

print("Running batch_third_largest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_third_largest_mask.py"),
    "--data_root", str(patient_dir),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(1)

print("Running batch_brightest_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_brightest_mask.py"),
    "--data_root", str(patient_dir),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(1)

print("Running batch_allregions_mask.py...")
subprocess.run([
    sys.executable, str(BATCH_DIR / "batch_allregions_mask.py"),
    "--data_root", str(patient_dir),
    "--percentiles", *(str(p) for p in PERCENTILES),
    "--n_values", *(str(n) for n in N_VALUES)
], check=True)
time.sleep(1)

print("All batch mask generation scripts completed for one patient.") 