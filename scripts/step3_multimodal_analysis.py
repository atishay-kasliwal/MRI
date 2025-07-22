import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Paths
MERGED_CSV = 'data/radiomics/merged_radiomics_clinical.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Modalities
MODALITIES = ['T1', 'DWI', 'ADC', 'FLAIR', 'T2']

# Load data
data = pd.read_csv(MERGED_CSV)

def get_modality_features(X, modality):
    # Select columns that contain the modality name (case-insensitive)
    return [col for col in X.columns if modality.lower() in col.lower()]

# Prepare features and target
drop_cols = ['PatientID', 'MRN ANON', 'Last mRS', 'Modality']
X_all = data.drop(columns=[col for col in drop_cols if col in data.columns])
y = data['Last mRS']

# Convert all columns to numeric where possible, coerce errors to NaN
X_all = X_all.apply(pd.to_numeric, errors='coerce')
X_all = X_all.select_dtypes(include=[np.number])

# Drop columns with >50% missing values
min_non_missing = int(0.5 * len(X_all))
X_all = X_all.loc[:, X_all.notnull().sum() >= min_non_missing]
X_all = X_all.fillna(X_all.mean())

# Drop any rows where X or y has a missing value
mask = X_all.notnull().all(axis=1) & y.notnull()
X_all = X_all[mask]
y = y[mask]

results = []

for modality in MODALITIES + ['ALL']:
    if modality == 'ALL':
        X = X_all.copy()
        feat_set = 'ALL'
    else:
        feat_cols = get_modality_features(X_all, modality)
        if not feat_cols:
            print(f"No features found for {modality}, skipping.")
            continue
        X = X_all[feat_cols]
        feat_set = modality
    if X.shape[1] == 0:
        print(f"No usable features for {feat_set}, skipping.")
        continue
    # Drop any rows with missing values (should be none after fillna, but just in case)
    mask = X.notnull().all(axis=1) & y.notnull()
    X_mod = X[mask]
    y_mod = y[mask]
    if len(X_mod) < 10:
        print(f"Not enough samples for {feat_set}, skipping.")
        continue
    X_train, X_test, y_train, y_test = train_test_split(X_mod, y_mod, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'modality': feat_set, 'n_features': X.shape[1], 'MAE': mae, 'R2': r2})
    print(f"{feat_set}: n_features={X.shape[1]}, MAE={mae:.4f}, R2={r2:.4f}")

# Save results
results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, 'step3_multimodal_performance.csv')
results_df.to_csv(results_path, index=False)
print(f"Step 3 complete: Multimodal performance saved to {results_path}") 