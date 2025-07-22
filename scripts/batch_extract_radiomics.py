import os
from pathlib import Path
import csv
from radiomics import featureextractor
import SimpleITK as sitk
import sys

# Accept root directory as argument
if len(sys.argv) > 1:
    ROOT = Path(sys.argv[1])
else:
    ROOT = Path('/Volumes/Kasliwal V1.1/downloads/2020')

MODALITIES = ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']

# Output CSV
year = ROOT.name if ROOT.name.isdigit() else ROOT.parts[-1]
OUTPUT_CSV = ROOT / f'all_patients_radiomics_{year}.csv'

extractor = featureextractor.RadiomicsFeatureExtractor()

results = []
header = ['PatientID', 'Modality']

for patient_dir in ROOT.iterdir():
    if not patient_dir.is_dir():
        continue
    outcome_dir = patient_dir / 'Outcome'
    if not outcome_dir.exists():
        print(f"No Outcome folder for {patient_dir.name}, skipping.")
        continue
    # Find mask (assume only one .nii.gz mask file)
    mask_files = list(outcome_dir.glob('*mask*.nii.gz'))
    if not mask_files:
        print(f"No mask found in {outcome_dir}, skipping.")
        continue
    mask_path = mask_files[0]
    for modality in MODALITIES:
        img_files = list(outcome_dir.glob(f'CORRECT{modality}_*.nii.gz'))
        if not img_files:
            print(f"No N4-corrected {modality} for {patient_dir.name}, skipping.")
            continue
        img_path = img_files[0]
        try:
            feature_vector = extractor.execute(str(img_path), str(mask_path))
        except Exception as e:
            if 'geometry mismatch' in str(e) or 'Inputs do not occupy the same physical space' in str(e):
                print(f"SKIPPED (geometry mismatch): {patient_dir.name} - {modality}")
                continue
            else:
                print(f"ERROR for {patient_dir.name} - {modality}: {e}")
                continue
        if len(header) == 2:
            header += [k for k in feature_vector.keys() if not k.startswith('diagnostics_')]
        row = [patient_dir.name, modality] + [feature_vector[k] for k in header[2:]]
        results.append(row)
        print(f"Extracted features for {patient_dir.name} - {modality}")

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results)

print(f"All radiomics features saved to {OUTPUT_CSV}") 