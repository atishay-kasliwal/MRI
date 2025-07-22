import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import os

# Paths
MERGED_CSV = 'data/radiomics/merged_radiomics_clinical.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
data = pd.read_csv(MERGED_CSV)

# Prepare features and target
drop_cols = ['PatientID', 'MRN ANON', 'Last mRS', 'Modality']
X = data.drop(columns=[col for col in drop_cols if col in data.columns])
y = data['Last mRS']

# Convert all columns to numeric where possible, coerce errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')
# Keep only numeric columns
X = X.select_dtypes(include=[np.number])

# Drop columns with more than 50% missing values
min_non_missing = int(0.5 * len(X))
X = X.loc[:, X.notnull().sum() >= min_non_missing]

# Fill remaining missing values
X = X.fillna(X.mean())

# Drop any rows where X or y has a missing value
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]

# Retrain model on filtered data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 1. Feature Importances (from model)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns
fi_df = pd.DataFrame({'feature': feature_names[indices], 'importance': importances[indices]})
fi_path = os.path.join(RESULTS_DIR, 'step2_feature_importances.csv')
fi_df.to_csv(fi_path, index=False)

plt.figure(figsize=(10,6))
plt.title('Model Feature Importances (Top 20)')
plt.bar(range(20), importances[indices[:20]], align='center')
plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step2_feature_importances.png'))
plt.close()

# 2. Permutation Importances
perm = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
perm_indices = np.argsort(perm.importances_mean)[::-1]
perm_df = pd.DataFrame({'feature': feature_names[perm_indices], 'perm_importance_mean': perm.importances_mean[perm_indices]})
perm_path = os.path.join(RESULTS_DIR, 'step2_permutation_importances.csv')
perm_df.to_csv(perm_path, index=False)

plt.figure(figsize=(10,6))
plt.title('Permutation Importances (Top 20)')
plt.bar(range(20), perm.importances_mean[perm_indices[:20]], align='center')
plt.xticks(range(20), [feature_names[i] for i in perm_indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step2_permutation_importances.png'))
plt.close()

# 3. SHAP Values (TreeExplainer)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step2_shap_summary_bar.png'))
plt.close()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step2_shap_summary_dot.png'))
plt.close()

# Save mean absolute SHAP values
shap_df = pd.DataFrame({'feature': X.columns, 'mean_abs_shap': np.abs(shap_values).mean(axis=0)})
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
shap_path = os.path.join(RESULTS_DIR, 'step2_shap_importances.csv')
shap_df.to_csv(shap_path, index=False)

print("Step 2 complete: Feature importances, permutation importances, and SHAP values saved to results/.") 