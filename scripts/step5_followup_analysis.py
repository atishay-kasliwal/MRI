import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

MERGED_CSV = 'data/radiomics/merged_radiomics_clinical.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
data = pd.read_csv(MERGED_CSV)

# Analyze and plot distribution of Days f/u
days_fu = pd.to_numeric(data['Days f/u'], errors='coerce')
plt.figure(figsize=(7,4))
plt.hist(days_fu.dropna(), bins=20, color='skyblue', edgecolor='k')
plt.xlabel('Days f/u')
plt.ylabel('Count')
plt.title('Distribution of Follow-up Days')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step5_days_fu_distribution.png'))
plt.close()

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

# 1. Model WITHOUT Days f/u
if 'Days f/u' in X_all.columns:
    X_wo_fu = X_all.drop(columns=['Days f/u'])
else:
    X_wo_fu = X_all.copy()
X_train, X_test, y_train, y_test = train_test_split(X_wo_fu, y, test_size=0.2, random_state=42)
model_wo = RandomForestRegressor(n_estimators=100, random_state=42)
model_wo.fit(X_train, y_train)
y_pred_wo = model_wo.predict(X_test)
mae_wo = mean_absolute_error(y_test, y_pred_wo)
r2_wo = r2_score(y_test, y_pred_wo)

# 2. Model WITH Days f/u
if 'Days f/u' in X_all.columns:
    X_w_fu = X_all.copy()
    X_train, X_test, y_train, y_test = train_test_split(X_w_fu, y, test_size=0.2, random_state=42)
    model_w = RandomForestRegressor(n_estimators=100, random_state=42)
    model_w.fit(X_train, y_train)
    y_pred_w = model_w.predict(X_test)
    mae_w = mean_absolute_error(y_test, y_pred_w)
    r2_w = r2_score(y_test, y_pred_w)
else:
    mae_w = np.nan
    r2_w = np.nan

# 3. Stratify by follow-up duration bins
bins = [0, 30, 90, 180, 365, np.inf]
labels = ['<30d', '30-90d', '90-180d', '180-365d', '>365d']
data['fu_bin'] = pd.cut(days_fu, bins=bins, labels=labels)
strat_results = []
for label in labels:
    idx = data['fu_bin'] == label
    if idx.sum() < 10:
        continue
    X_bin = X_all[idx]
    y_bin = y[idx]
    if X_bin.shape[0] < 10:
        continue
    X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    strat_results.append({'fu_bin': label, 'n_samples': X_bin.shape[0], 'MAE': mae, 'R2': r2})

strat_df = pd.DataFrame(strat_results)
strat_path = os.path.join(RESULTS_DIR, 'step5_stratified_followup_performance.csv')
strat_df.to_csv(strat_path, index=False)

# Plot stratified results
plt.figure(figsize=(8,5))
plt.bar(strat_df['fu_bin'], strat_df['MAE'], color='tab:blue', alpha=0.7, label='MAE')
plt.ylabel('MAE')
plt.xlabel('Follow-up Bin')
plt.title('Model MAE by Follow-up Duration Bin')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step5_stratified_mae.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.bar(strat_df['fu_bin'], strat_df['R2'], color='tab:green', alpha=0.7, label='R2')
plt.ylabel('R2')
plt.xlabel('Follow-up Bin')
plt.title('Model R2 by Follow-up Duration Bin')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step5_stratified_r2.png'))
plt.close()

# Save summary results
summary = pd.DataFrame({
    'Model': ['Without Days f/u', 'With Days f/u'],
    'MAE': [mae_wo, mae_w],
    'R2': [r2_wo, r2_w]
})
summary_path = os.path.join(RESULTS_DIR, 'step5_followup_model_comparison.csv')
summary.to_csv(summary_path, index=False)

print("Step 5 complete: Follow-up analysis results saved to results/.") 