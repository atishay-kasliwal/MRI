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

# Load data
data = pd.read_csv(MERGED_CSV)

def get_feature_sets(X):
    radiomics_cols = [col for col in X.columns if col.startswith('original_')]
    clinical_cols = [col for col in X.columns if col not in radiomics_cols]
    return clinical_cols, radiomics_cols

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

# Get feature sets
df_cols = X_all.columns
clinical_cols, radiomics_cols = get_feature_sets(X_all)

results = []

for feat_set, feat_cols in zip(['Clinical', 'Radiomics', 'Combined'], [clinical_cols, radiomics_cols, df_cols]):
    X = X_all[feat_cols]
    if X.shape[1] == 0:
        print(f"No usable features for {feat_set}, skipping.")
        continue
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
    results.append({'feature_set': feat_set, 'n_features': X.shape[1], 'MAE': mae, 'R2': r2})
    print(f"{feat_set}: n_features={X.shape[1]}, MAE={mae:.4f}, R2={r2:.4f}")

# Save results
results_df = pd.DataFrame(results)
results_path = os.path.join(RESULTS_DIR, 'step4_clinical_vs_radiomics_performance.csv')
results_df.to_csv(results_path, index=False)
print(f"Step 4 complete: Clinical vs. Radiomics performance saved to {results_path}") 