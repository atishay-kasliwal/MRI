import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
import re

# Output directory
output_dir = os.path.join('results', 'survival_analysis')
os.makedirs(output_dir, exist_ok=True)
input_csv = os.path.join('data', 'radiomics', 'merged_radiomics_clinical.csv')

# Load data
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv)

# --- Use 'Days f/u' as follow-up time, create Event=1 for all (no censoring info) ---
if 'Days f/u' not in df.columns:
    raise ValueError("Column 'Days f/u' (follow-up time) is required.")
df['Event'] = 1  # Assume all patients had the event (uncensored)

# Clean 'Days f/u': remove non-numeric chars, convert to float, drop rows where conversion fails
km_df = df.copy()
km_df['Days f/u'] = km_df['Days f/u'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
km_df['Days f/u'] = pd.to_numeric(km_df['Days f/u'], errors='coerce')
km_df = km_df.dropna(subset=['Days f/u'])

# --- Kaplan-Meier for whole cohort ---
kmf = KaplanMeierFitter()
T = km_df['Days f/u']
E = km_df['Event']
kmf.fit(T, event_observed=E, label='All Patients')

plt.figure(figsize=(8,6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve (All Patients)')
plt.xlabel('Days of Follow-up')
plt.ylabel('Survival Probability')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'kaplan_meier.pdf'))
plt.close()

# --- Stratified KM by median of a top radiomic feature ---
radiomic_cols = [col for col in df.columns if col.startswith('original_')]
if radiomic_cols:
    top_feature = radiomic_cols[0]
    median_val = km_df[top_feature].median()
    km_df['High_'+top_feature] = km_df[top_feature] > median_val
    kmf_high = KaplanMeierFitter()
    kmf_low = KaplanMeierFitter()
    plt.figure(figsize=(8,6))
    for group, label in zip([True, False], ['High', 'Low']):
        mask = km_df['High_'+top_feature] == group
        kmf.fit(km_df.loc[mask, 'Days f/u'], event_observed=km_df.loc[mask, 'Event'], label=f'{label} {top_feature}')
        kmf.plot_survival_function()
    plt.title(f'Kaplan-Meier by {top_feature} (median split)')
    plt.xlabel('Days of Follow-up')
    plt.ylabel('Survival Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kaplan_meier_{top_feature}_stratified.pdf'))
    plt.close()

# --- Cox Proportional Hazards Model ---
# Use radiomic + clinical features (drop rows with missing)
clinical_cols = [col for col in df.columns if col not in radiomic_cols and col not in ['PatientID', 'Modality', 'Last_mRS', 'Year', 'Comments', 'Unnamed: 2', 'Unnamed: 77', 'comment', 'NEW MRN', 'Days f/u', 'Event'] and not col.startswith('original_') and df[col].dtype in [np.float64, np.int64]]
features = radiomic_cols[:5] + clinical_cols[:5]  # Use top 5 of each for demo
cox_df = df[['Days f/u', 'Event'] + features].dropna()
if len(cox_df) > 10:
    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col='Days f/u', event_col='Event')
    with open(os.path.join(output_dir, 'cox_summary.txt'), 'w') as f:
        f.write(cph.summary.to_string())
    print('Cox model summary saved.')
else:
    print('Not enough data for Cox model.') 