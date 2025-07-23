import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import os

# Paths
merged_file = 'data/radiomics/merged_radiomics_clinical.csv'
pred_file = 'results/lastmrs_model_comparison/radiomics+clinical_RandomForest_classification_predictions.csv'
output_dir = 'results/lastmrs_model_comparison/'
pdf_path = os.path.join(output_dir, 'lastmrs_predictions_2021_report.pdf')

# Load merged data
print(f"Loading merged data from {merged_file}...")
df = pd.read_csv(merged_file)

# Filter for 2021
if 'Year' not in df.columns:
    raise ValueError('Year column not found in merged data!')
df_2021 = df[df['Year'] == 2021].reset_index(drop=True)

# Load predictions
print(f"Loading predictions from {pred_file}...")
preds = pd.read_csv(pred_file)

# Check length match
if len(df_2021) != len(preds):
    raise ValueError(f"Length mismatch: 2021 data has {len(df_2021)} rows, predictions file has {len(preds)} rows.")

# Get true and predicted values
if 'y_true' in preds.columns and 'y_pred' in preds.columns:
    y_true = preds['y_true']
    y_pred = preds['y_pred']
else:
    raise ValueError('Prediction file must have y_true and y_pred columns!')

# Compute metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
report = classification_report(y_true, y_pred, digits=3)
cm = confusion_matrix(y_true, y_pred)

# Prepare PDF
with PdfPages(pdf_path) as pdf:
    # Summary page
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    summary = f"Last mRS Prediction Report (2021)\n\n"
    summary += f"Samples: {len(y_true)}\n"
    summary += f"Accuracy: {acc:.3f}\nF1 Score: {f1:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\n"
    ax.text(0.1, 0.8, summary, fontsize=14, va='top')
    pdf.savefig(fig)
    plt.close(fig)

    # Classification report page
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    ax.text(0.01, 0.99, report, fontsize=12, va='top', family='monospace')
    plt.title('Classification Report', fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    pdf.savefig(fig)
    plt.close(fig)

    # True vs Predicted scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect Prediction')
    ax.set_xlabel('True mRS')
    ax.set_ylabel('Predicted mRS')
    ax.set_title('True vs Predicted mRS (2021)')
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # Distribution plots
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(y_true, color='blue', label='True', kde=True, stat='density', bins=7, alpha=0.5)
    sns.histplot(y_pred, color='orange', label='Predicted', kde=True, stat='density', bins=7, alpha=0.5)
    ax.set_title('Distribution of True vs Predicted mRS (2021)')
    ax.set_xlabel('mRS')
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

print(f"PDF report saved to {pdf_path}") 