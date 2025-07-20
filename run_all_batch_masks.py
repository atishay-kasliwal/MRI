import shutil
from pathlib import Path
import subprocess
import time

# Set your local data root
DATA_ROOT = Path('/Volumes/Kasliwal V1.1/MRi Data V1.2/2024')

# Delete all mask folders for a clean start
print("Deleting all previous mask files...")
for patient_dir in DATA_ROOT.iterdir():
    mask_dir = patient_dir / "mask"
    if mask_dir.exists() and mask_dir.is_dir():
        shutil.rmtree(mask_dir)
        print(f"Deleted {mask_dir}")

# Now run the batch scripts as before
print("Running batch_n_largest_mask.py...")
subprocess.run(["python", "batch_n_largest_mask.py"], check=True)
time.sleep(5)

print("Running batch_brightest_mask.py...")
subprocess.run(["python", "batch_brightest_mask.py"], check=True)
time.sleep(5)

print("Running batch_allregions_mask.py...")
subprocess.run(["python", "batch_allregions_mask.py"], check=True)
time.sleep(5)

print("All batch mask generation scripts completed.") 