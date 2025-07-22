import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
import shap
import lime
import lime.lime_tabular
from sklearn.tree import plot_tree
from mapie.classification import MapieClassifier

# Output directory
output_dir = os.path.join('results', 'model_interpretability_and_uncertainty')
os.makedirs(output_dir, exist_ok=True)

# --- Load data and model ---
# For demo, use radiomics+clinical RandomForest classification (80/20 split)
pred_csv = os.path.join('results', 'lastmrs_model_comparison', 'radiomics+clinical_RandomForest_classification_predictions.csv')
metrics_csv = os.path.join('results', 'lastmrs_model_comparison', 'model_metrics_summary.csv')
merged_csv = os.path.join('data', 'radiomics', 'merged_radiomics_clinical.csv')

df = pd.read_csv(merged_csv)
# Drop rows with missing or non-finite Last mRS before creating X and y
mask = df['Last mRS'].notna() & np.isfinite(df['Last mRS'])
df = df[mask].copy()
# Use same feature selection as training script
radiomic_cols = [col for col in df.columns if col.startswith('original_')]
clinical_cols = [col for col in df.columns if col not in radiomic_cols and col not in ['PatientID', 'Modality', 'Last_mRS', 'Year', 'Comments', 'Unnamed: 2', 'Unnamed: 77', 'comment', 'NEW MRN'] and not col.startswith('original_') and df[col].dtype in [np.float64, np.int64]]
X = df[radiomic_cols + clinical_cols].fillna(0)
y = np.round(df['Last mRS']).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train RandomForestClassifier (for demo) ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- SHAP values ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
plt.close()

# Dependence plot for top feature
# For multiclass, use the most common class in y_test
most_common_class = np.bincount(y_test).argmax()
shap_vals_for_class = shap_values[most_common_class]
if shap_vals_for_class.shape[0] == X_test.shape[0]:
    top_feat = X_test.columns[np.abs(shap_vals_for_class).mean(axis=0).argmax()]
    plt.figure()
    shap.dependence_plot(top_feat, shap_vals_for_class, X_test, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_dependence_{top_feat}.png'))
    plt.close()
else:
    print('Skipping SHAP dependence plot due to shape mismatch.')

# --- LIME explanations for a few test samples ---
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=[str(i) for i in np.unique(y)], discretize_continuous=True)
for i in range(3):
    exp = lime_explainer.explain_instance(X_test.values[i], rf.predict_proba, num_features=10)
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'lime_explanation_sample_{i}.png'))
    plt.close()

# --- Uncertainty quantification: posterior distribution (predict_proba) ---
probs = rf.predict_proba(X_test)
plt.figure(figsize=(8,6))
for i, cls in enumerate(rf.classes_):
    sns.histplot(probs[:, i], kde=True, label=f'Class {cls}', bins=20, alpha=0.5)
plt.title('Posterior Distribution of Predicted Probabilities (Test Set)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'posterior_distribution.png'))
plt.close()

# --- Conformal prediction (MAPIE) ---
try:
    mapie = MapieClassifier(rf, method="score")
    mapie.fit(X_train, y_train)
    y_pred, y_ps = mapie.predict(X_test, alpha=0.1)
    plt.figure(figsize=(8,6))
    plt.hist([len(set_) for set_ in y_ps], bins=range(1, len(rf.classes_)+2), align='left', rwidth=0.8)
    plt.xlabel('Number of Classes in Prediction Set')
    plt.ylabel('Frequency')
    plt.title('Conformal Prediction Set Sizes (alpha=0.1)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conformal_prediction_set_sizes.png'))
    plt.close()
except Exception as e:
    print('Conformal prediction failed:', e)

# --- Classification threshold vs metrics ---
probs = rf.predict_proba(X_test)
thresh_range = np.linspace(0, 1, 101)
metrics = {'threshold': [], 'f1': [], 'accuracy': [], 'precision': [], 'recall': []}
for thresh in thresh_range:
    y_pred_thresh = (probs[:, 1] >= thresh).astype(int) if probs.shape[1] > 1 else (probs[:, 0] >= thresh).astype(int)
    metrics['threshold'].append(thresh)
    metrics['f1'].append(f1_score(y_test, y_pred_thresh, average='macro', zero_division=0))
    metrics['accuracy'].append(accuracy_score(y_test, y_pred_thresh))
    metrics['precision'].append(precision_score(y_test, y_pred_thresh, average='macro', zero_division=0))
    metrics['recall'].append(recall_score(y_test, y_pred_thresh, average='macro', zero_division=0))
metrics_df = pd.DataFrame(metrics)
# Individual plots
for metric in ['f1', 'accuracy', 'precision', 'recall']:
    plt.figure()
    plt.plot(metrics_df['threshold'], metrics_df[metric], label=metric)
    plt.xlabel('Threshold')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} vs Threshold (Test Set)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'threshold_vs_{metric}.png'))
    plt.close()
# Combined plot
plt.figure(figsize=(8,6))
for metric in ['f1', 'accuracy', 'precision', 'recall']:
    plt.plot(metrics_df['threshold'], metrics_df[metric], label=metric)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Classification Metrics vs Threshold (Test Set)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'threshold_vs_metrics_combined.png'))
plt.close()

# --- Tree plot for a single tree ---
plt.figure(figsize=(20,10))
plot_tree(rf.estimators_[0], feature_names=X_train.columns, class_names=[str(i) for i in rf.classes_], filled=True, rounded=True, max_depth=3)
plt.title('Decision Tree Plot (First Tree in RandomForest)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tree_plot.png'))
plt.close()

print('All interpretability and uncertainty plots saved to', output_dir) 