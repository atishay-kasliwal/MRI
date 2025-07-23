import pandas as pd
import os

# Paths
input_csv = os.path.join('data', 'radiomics', 'merged_radiomics_clinical.csv')
output_csv = os.path.join('results', 'radiomics_2020_only.csv')

# Load data
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv)

# Filter for 2020
if 'Year' not in df.columns:
    raise ValueError("Column 'Year' not found in the input file.")
df_2020 = df[df['Year'] == 2020].copy()

# Select radiomic features and identifiers
radiomic_cols = [col for col in df_2020.columns if col.startswith('original_')]
id_cols = ['PatientID', 'Modality']
selected_cols = id_cols + radiomic_cols

# Save to CSV
print(f"Saving 2020 radiomics to {output_csv}...")
df_2020[selected_cols].to_csv(output_csv, index=False)
print("Done.") 