import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Output directory
output_dir = os.path.join('results', 'visualize_lastmrs_predictions')
os.makedirs(output_dir, exist_ok=True)
input_csv = os.path.join('results', 'last_mrs_predictions.csv')

# Load data
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv)

# Calculate residuals and metrics
df['residual'] = df['y_true'] - df['y_pred']
mae = mean_absolute_error(df['y_true'], df['y_pred'])
r2 = r2_score(df['y_true'], df['y_pred'])

# --- Accuracy, F1, Precision, Recall calculation (treat as classification) ---
df['y_pred_rounded'] = np.round(df['y_pred']).astype(int)
df['y_true_int'] = df['y_true'].astype(int)
accuracy = accuracy_score(df['y_true_int'], df['y_pred_rounded'])
cm = confusion_matrix(df['y_true_int'], df['y_pred_rounded'])
f1_macro = f1_score(df['y_true_int'], df['y_pred_rounded'], average='macro')
precision_macro = precision_score(df['y_true_int'], df['y_pred_rounded'], average='macro', zero_division=0)
recall_macro = recall_score(df['y_true_int'], df['y_pred_rounded'], average='macro', zero_division=0)

# Class-wise metrics
labels = np.unique(np.concatenate([df['y_true_int'], df['y_pred_rounded']]))
f1_class = f1_score(df['y_true_int'], df['y_pred_rounded'], average=None, labels=labels, zero_division=0)
precision_class = precision_score(df['y_true_int'], df['y_pred_rounded'], average=None, labels=labels, zero_division=0)
recall_class = recall_score(df['y_true_int'], df['y_pred_rounded'], average=None, labels=labels, zero_division=0)

# Prepare PDF
pdf_path = os.path.join(output_dir, 'lastmrs_prediction_plots.pdf')
pdf = PdfPages(pdf_path)

# --- Summary Page ---
fig, ax = plt.subplots(figsize=(8, 7))
summary_text = (
    f"Last mRS Prediction Visualization\n\n"
    f"Overall MAE: {mae:.3f}\n"
    f"R2 Score: {r2:.3f}\n"
    f"Accuracy (rounded): {accuracy:.3f}\n"
    f"F1 Score (macro): {f1_macro:.3f}\n"
    f"Precision (macro): {precision_macro:.3f}\n"
    f"Recall (macro): {recall_macro:.3f}\n"
    f"N: {len(df)}"
)
ax.text(0.1, 0.7, summary_text, fontsize=14, va='top')
ax.axis('off')
pdf.savefig(fig)
plt.close(fig)

# --- Class-wise metrics table ---
metrics_table = pd.DataFrame({
    'Class': labels,
    'Precision': np.round(precision_class, 3),
    'Recall': np.round(recall_class, 3),
    'F1': np.round(f1_class, 3)
})
fig, ax = plt.subplots(figsize=(8, 0.5 + 0.3 * len(metrics_table)))
ax.axis('off')
table = ax.table(cellText=metrics_table.values,
                 colLabels=metrics_table.columns,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.5)
plt.title('Class-wise Precision, Recall, F1', fontsize=14, pad=20)
pdf.savefig(fig)
plt.close(fig)

# --- Scatter plot: y_true vs y_pred ---
plt.figure(figsize=(7, 7))
plt.scatter(df['y_true'], df['y_pred'], alpha=0.7)
plt.plot([df['y_true'].min(), df['y_true'].max()], [df['y_true'].min(), df['y_true'].max()], 'r--', label='y=x')
plt.xlabel('True Last mRS')
plt.ylabel('Predicted Last mRS')
plt.title('True vs Predicted Last mRS')
plt.legend()
pdf.savefig()
plt.close()

# --- Residual plot: residuals vs y_true ---
plt.figure(figsize=(8, 5))
plt.scatter(df['y_true'], df['residual'], alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('True Last mRS')
plt.ylabel('Residual (y_true - y_pred)')
plt.title('Residuals vs True Last mRS')
pdf.savefig()
plt.close()

# --- Histogram of residuals ---
plt.figure(figsize=(8, 5))
sns.histplot(df['residual'], bins=15, kde=True)
plt.xlabel('Residual (y_true - y_pred)')
plt.title('Histogram of Residuals')
pdf.savefig()
plt.close()

# --- Boxplot of residuals ---
plt.figure(figsize=(6, 5))
sns.boxplot(y=df['residual'])
plt.ylabel('Residual (y_true - y_pred)')
plt.title('Boxplot of Residuals')
pdf.savefig()
plt.close()

# --- Bar plot of mean absolute error by true Last mRS ---
df['abs_error'] = np.abs(df['residual'])
mae_by_true = df.groupby('y_true')['abs_error'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='y_true', y='abs_error', data=mae_by_true, palette='viridis')
plt.xlabel('True Last mRS')
plt.ylabel('Mean Absolute Error')
plt.title('Mean Absolute Error by True Last mRS')
pdf.savefig()
plt.close()

# --- Confusion matrix plot ---
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted (rounded)')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix (Last mRS, rounded)')
pdf.savefig(fig)
plt.close(fig)

pdf.close()
print(f'All plots saved to {pdf_path}') 