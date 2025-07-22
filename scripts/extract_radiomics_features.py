import os
from pathlib import Path
import csv
from radiomics import featureextractor
import SimpleITK as sitk

# Configuration
PATIENT_DIR = Path('data/DE-IDENTIFIED, 6112052.brainlab')
MODALITIES = ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']
CORRECTEDN4_DIR = PATIENT_DIR / 'correctedn4'
MASK_DIR = PATIENT_DIR / 'mask'
PATIENT_NAME = PATIENT_DIR.name.replace(',', '').replace(' ', '_')
MASK_PATH = MASK_DIR / f'MASK_{PATIENT_NAME}.nii.gz'

# Output CSV
OUTPUT_CSV = PATIENT_DIR / f'radiomics_features_{PATIENT_NAME}.csv'

# Initialize feature extractor (default settings)
extractor = featureextractor.RadiomicsFeatureExtractor()

# Check mask exists
if not MASK_PATH.exists():
    raise FileNotFoundError(f"Mask file not found: {MASK_PATH}")

results = []
header = ['Modality']

for modality in MODALITIES:
    img_path = CORRECTEDN4_DIR / f'CORRECT{modality}_{PATIENT_NAME}.nii.gz'
    if not img_path.exists():
        print(f"Warning: {img_path} not found, skipping.")
        continue
    # Extract features
    feature_vector = extractor.execute(str(img_path), str(MASK_PATH))
    # On first run, set header
    if len(header) == 1:
        header += [k for k in feature_vector.keys() if not k.startswith('diagnostics_')]
    row = [modality] + [feature_vector[k] for k in header[1:]]
    results.append(row)
    print(f"Extracted features for {modality}")

# Write to CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results)

print(f"Radiomics features saved to {OUTPUT_CSV}") 