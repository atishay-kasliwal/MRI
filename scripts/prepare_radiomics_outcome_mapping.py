import pandas as pd
import os

# Paths
input_csv = os.path.join('data', 'radiomics', 'merged_radiomics_clinical.csv')
output_csv = os.path.join('results', 'radiomics_lastmrs_mapping.csv')

# Load data
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv)

# Identify radiomic feature columns (those starting with 'original_')
radiomic_cols = [col for col in df.columns if col.startswith('original_')]

# Always keep identifiers
id_cols = ['PatientID', 'Modality']

# Outcome column
outcome_col = 'Last mRS'

# Check if outcome column exists
if outcome_col not in df.columns:
    raise ValueError(f"Column '{outcome_col}' not found in input CSV.")

# Select relevant columns
selected_cols = id_cols + radiomic_cols + [outcome_col]
subset = df[selected_cols]

# Filter out rows with missing Last mRS
subset = subset.dropna(subset=[outcome_col])

# Save to output
print(f"Saving mapping to {output_csv}...")
subset.to_csv(output_csv, index=False)
print("Done.") 