import pandas as pd
from pathlib import Path

# Paths
RADIOMICS_2020 = Path('/Volumes/Kasliwal V1.1/downloads/2020/all_patients_radiomics_2020.csv')
RADIOMICS_2021 = Path('/Volumes/Kasliwal V1.1/downloads/2021/all_patients_radiomics_2021.csv')
CLINICAL_2020 = Path('data/radiomics/MRI SCAN- MRN NUMBER.xlsx - 2020_Patients.csv')
CLINICAL_2021 = Path('data/radiomics/MRI SCAN- MRN NUMBER.xlsx - 2021_Patients.csv')
OUTPUT = Path('data/radiomics/merged_radiomics_clinical.csv')

# Load radiomics
r2020 = pd.read_csv(RADIOMICS_2020)
r2021 = pd.read_csv(RADIOMICS_2021)
print(f"Radiomics 2020: {len(r2020)} rows | 2021: {len(r2021)} rows")
radiomics = pd.concat([r2020, r2021], ignore_index=True)
print(f"Total radiomics rows: {len(radiomics)}")

# Extract MRN ANON
radiomics['MRN ANON'] = radiomics['PatientID'].str.extract(r'(\d+)')

# Load clinical
c2020 = pd.read_csv(CLINICAL_2020)
c2021 = pd.read_csv(CLINICAL_2021)

# Standardize MRN column name in both clinical files
if 'MRN ANON' not in c2020.columns:
    if 'ANON MRN ' in c2020.columns:
        c2020 = c2020.rename(columns={'ANON MRN ': 'MRN ANON'})
if 'MRN ANON' not in c2021.columns:
    if 'ANON MRN ' in c2021.columns:
        c2021 = c2021.rename(columns={'ANON MRN ': 'MRN ANON'})

# Strip 'ANON' prefix from clinical MRN ANON column
c2020['MRN ANON'] = c2020['MRN ANON'].astype(str).str.replace('ANON', '').str.strip()
c2021['MRN ANON'] = c2021['MRN ANON'].astype(str).str.replace('ANON', '').str.strip()

print(f"Clinical 2020: {len(c2020)} rows | 2021: {len(c2021)} rows")
clinical = pd.concat([c2020, c2021], ignore_index=True)
print(f"Total clinical rows: {len(clinical)}")

# Merge
merged = pd.merge(radiomics, clinical, on='MRN ANON')
print(f"Rows after merge: {len(merged)}")
if len(merged) == 0:
    print("Sample radiomics MRN ANON:", radiomics['MRN ANON'].unique()[:5])
    print("Sample clinical MRN ANON:", clinical['MRN ANON'].unique()[:5])
    print("Check for leading/trailing spaces or type mismatches.")

# Save
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT, index=False)
print(f"Merged file saved to {OUTPUT}") 