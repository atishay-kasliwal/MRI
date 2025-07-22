import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import re

# Output directory
output_dir = os.path.join('results', 'lastmrs_model_comparison')
os.makedirs(output_dir, exist_ok=True)
input_csv = os.path.join('data', 'radiomics', 'merged_radiomics_clinical.csv')

# Load data
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv)

# --- Sanitize all DataFrame column names for LightGBM compatibility ---
def sanitize_feature_names_regex(cols):
    return [re.sub(r'[^0-9a-zA-Z_]', '_', col) for col in cols]
df.columns = sanitize_feature_names_regex(df.columns)

# Rebuild feature lists after sanitization
radiomic_cols = [col for col in df.columns if col.startswith('original_')]
clinical_cols = [col for col in df.columns if col not in radiomic_cols and col not in ['PatientID', 'Modality', 'Last_mRS', 'Year', 'Comments', 'Unnamed_2', 'Unnamed_77', 'comment', 'NEW_MRN'] and not col.startswith('original_') and df[col].dtype in [np.float64, np.int64]]

# --- Drop rows with missing or non-finite Last_mRS ---
# This is required for both regression and classification
# Use sanitized column name
mask = df['Last_mRS'].notna() & np.isfinite(df['Last_mRS'])
df = df[mask].copy()

# Identify features
# radiomic_cols = [col for col in df.columns if col.startswith('original_')]
# clinical_cols = [col for col in df.columns if col not in radiomic_cols and col not in ['PatientID', 'Modality', 'Last_mRS', 'Year', 'Comments', 'Unnamed: 2', 'Unnamed: 77', 'comment', 'NEW MRN'] and not col.startswith('original_') and df[col].dtype in [np.float64, np.int64]]

# Prepare targets
y_reg = df['Last_mRS']
y_clf = np.round(df['Last_mRS']).astype(int)

# Model configs
regressors = {
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(verbosity=0, random_state=42),
    'LightGBM': LGBMRegressor(random_state=42),
    'Linear': LinearRegression()
}
classifiers = {
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(verbosity=0, random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'Logistic': LogisticRegression(max_iter=1000, random_state=42)
}

results = []

# --- Helper functions for cross-validation ---
def cross_val_regression(model, X, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mae_scores = []
    r2_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))
    return np.mean(mae_scores), np.std(mae_scores), np.mean(r2_scores), np.std(r2_scores)

def cross_val_classification(model, X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    acc_scores, f1_scores, prec_scores, rec_scores = [], [], [], []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred, average='macro', zero_division=0))
        prec_scores.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        rec_scores.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    return (
        np.mean(acc_scores), np.std(acc_scores),
        np.mean(f1_scores), np.std(f1_scores),
        np.mean(prec_scores), np.std(prec_scores),
        np.mean(rec_scores), np.std(rec_scores)
    )

for feature_set, feature_cols in [('radiomics', radiomic_cols), ('radiomics+clinical', radiomic_cols + clinical_cols)]:
    X = df[feature_cols].fillna(0)
    # --- 80/20 Split: Regression ---
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    for name, model in regressors.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        y_test_1d = np.ravel(y_test)
        y_pred_1d = np.ravel(y_pred)
        pred_df = pd.DataFrame({'y_true': y_test_1d, 'y_pred': y_pred_1d})
        pred_df.to_csv(os.path.join(output_dir, f'{feature_set}_{name}_regression_predictions.csv'), index=False)
        results.append({'FeatureSet': feature_set, 'Task': 'regression', 'Model': name, 'Split': '80/20', 'MAE': mae, 'MAE_std': np.nan, 'R2': r2, 'R2_std': np.nan})
        # --- 5-fold CV ---
        mae_cv5, mae_cv5_std, r2_cv5, r2_cv5_std = cross_val_regression(model, X, y_reg, 5)
        results.append({'FeatureSet': feature_set, 'Task': 'regression', 'Model': name, 'Split': 'cv5', 'MAE': mae_cv5, 'MAE_std': mae_cv5_std, 'R2': r2_cv5, 'R2_std': r2_cv5_std})
        # --- 10-fold CV ---
        mae_cv10, mae_cv10_std, r2_cv10, r2_cv10_std = cross_val_regression(model, X, y_reg, 10)
        results.append({'FeatureSet': feature_set, 'Task': 'regression', 'Model': name, 'Split': 'cv10', 'MAE': mae_cv10, 'MAE_std': mae_cv10_std, 'R2': r2_cv10, 'R2_std': r2_cv10_std})
    # --- 80/20 Split: Classification ---
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        y_test_1d = np.ravel(y_test)
        y_pred_1d = np.ravel(y_pred)
        pred_df = pd.DataFrame({'y_true': y_test_1d, 'y_pred': y_pred_1d})
        pred_df.to_csv(os.path.join(output_dir, f'{feature_set}_{name}_classification_predictions.csv'), index=False)
        results.append({'FeatureSet': feature_set, 'Task': 'classification', 'Model': name, 'Split': '80/20', 'Accuracy': acc, 'Accuracy_std': np.nan, 'F1': f1, 'F1_std': np.nan, 'Precision': prec, 'Precision_std': np.nan, 'Recall': rec, 'Recall_std': np.nan})
        # --- 5-fold CV ---
        acc_cv5, acc_cv5_std, f1_cv5, f1_cv5_std, prec_cv5, prec_cv5_std, rec_cv5, rec_cv5_std = cross_val_classification(model, X, y_clf, 5)
        results.append({'FeatureSet': feature_set, 'Task': 'classification', 'Model': name, 'Split': 'cv5', 'Accuracy': acc_cv5, 'Accuracy_std': acc_cv5_std, 'F1': f1_cv5, 'F1_std': f1_cv5_std, 'Precision': prec_cv5, 'Precision_std': prec_cv5_std, 'Recall': rec_cv5, 'Recall_std': rec_cv5_std})
        # --- 10-fold CV ---
        acc_cv10, acc_cv10_std, f1_cv10, f1_cv10_std, prec_cv10, prec_cv10_std, rec_cv10, rec_cv10_std = cross_val_classification(model, X, y_clf, 10)
        results.append({'FeatureSet': feature_set, 'Task': 'classification', 'Model': name, 'Split': 'cv10', 'Accuracy': acc_cv10, 'Accuracy_std': acc_cv10_std, 'F1': f1_cv10, 'F1_std': f1_cv10_std, 'Precision': prec_cv10, 'Precision_std': prec_cv10_std, 'Recall': rec_cv10, 'Recall_std': rec_cv10_std})

# Save summary metrics
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'model_metrics_summary.csv'), index=False)
print('All models trained and predictions/metrics saved to', output_dir) 