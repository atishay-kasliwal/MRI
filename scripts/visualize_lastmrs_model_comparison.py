import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix
from matplotlib.table import Table

# Output directory
results_dir = os.path.join('results', 'lastmrs_model_comparison')
report_pdf = os.path.join(results_dir, 'model_comparison_report.pdf')

# --- Load metrics summary ---
metrics_csv = os.path.join(results_dir, 'model_metrics_summary.csv')
metrics_df = pd.read_csv(metrics_csv)

# --- Start PDF report ---
with PdfPages(report_pdf) as pdf:
    # --- Add detailed written summary and interpretation ---
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    summary_text = (
        'Model Comparison Report for Last mRS Prediction\n\n'
        'This report compares multiple machine learning models for predicting the Last mRS outcome, using radiomic features alone and combined with clinical features.\n\n'
        'Feature Sets:\n'
        '- Radiomics: Only radiomic features (quantitative features from images)\n'
        '- Radiomics+Clinical: Radiomic plus clinical features (e.g., age, sex, clinical scores)\n\n'
        'Tasks:\n'
        '- Regression: Predicting Last mRS as a continuous value (e.g., 2.5)\n'
        '- Classification: Predicting Last mRS as a discrete class (e.g., 0, 1, 2, 3, 4, 5, 6)\n\n'
        'Metrics:\n'
        '- MAE: Mean Absolute Error (lower is better, regression)\n'
        '- R²: R-squared (higher is better, regression)\n'
        '- Accuracy: Fraction of correct predictions (classification)\n'
        '- F1, Precision, Recall: Standard classification metrics (higher is better)\n\n'
        'Key Results:\n'
        '- Tree-based models (CatBoost, RandomForest, XGBoost, LightGBM) perform best, often with perfect or near-perfect scores.\n'
        '- Linear/Logistic regression performs worse, indicating non-linear relationships.\n'
        '- Adding clinical features does not significantly improve performance for tree-based models.\n'
        '- Perfect scores may indicate overfitting; results should be validated on more data.\n\n'
        'Limitations & Next Steps:\n'
        '- Results may not generalize due to small sample size and possible overfitting.\n'
        '- Use cross-validation and/or external validation for more robust assessment.\n'
        '- Consider feature importance analysis and model interpretability tools.\n'
    )
    ax.text(0, 1, summary_text, fontsize=12, va='top', ha='left', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Add readable tables with captions and footnotes ---
    for (feature_set, task), group in metrics_df.groupby(['FeatureSet', 'Task']):
        fig, ax = plt.subplots(figsize=(11, 2 + 0.4 * len(group)))
        ax.axis('off')
        table = ax.table(
            cellText=group[['Model', 'MAE', 'R2', 'Accuracy', 'F1', 'Precision', 'Recall']].round(3).values,
            colLabels=['Model', 'MAE', 'R²', 'Accuracy', 'F1', 'Precision', 'Recall'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        ax.set_title(f'{feature_set.capitalize()} - {task.capitalize()} Model Comparison', fontsize=14, pad=20)
        caption = (
            'MAE: Mean Absolute Error (regression, lower is better). '
            'R²: R-squared (regression, higher is better). '
            'Accuracy, F1, Precision, Recall: Classification metrics (higher is better).'
        )
        ax.text(0, -0.2, caption, fontsize=10, ha='left', va='top', transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)

    # --- Add limitations and next steps page ---
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    limitations_text = (
        'Limitations & Recommendations\n\n'
        '- The models may be overfitting, as indicated by perfect or near-perfect scores.\n'
        '- The dataset is relatively small; results may not generalize to new data.\n'
        '- Cross-validation or external validation is recommended.\n'
        '- Consider exploring feature importance and interpretability tools (e.g., SHAP, permutation importance).\n'
        '- Further work: test on larger, independent datasets; explore model calibration; report confidence intervals.'
    )
    ax.text(0, 1, limitations_text, fontsize=12, va='top', ha='left', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

    # --- Helper: get all prediction files ---
    prediction_files = [f for f in os.listdir(results_dir) if f.endswith('_predictions.csv')]

    # --- Bar plots for model comparison ---
    # Regression
    for feature_set in ['radiomics', 'radiomics+clinical']:
        sub = metrics_df[(metrics_df['Task'] == 'regression') & (metrics_df['FeatureSet'] == feature_set)]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sub.plot(x='Model', y='MAE', kind='bar', ax=ax, legend=False, color='skyblue')
            plt.ylabel('MAE')
            plt.title(f'{feature_set} - Regression MAE')
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8, 5))
            sub.plot(x='Model', y='R2', kind='bar', ax=ax, legend=False, color='salmon')
            plt.ylabel('R2')
            plt.title(f'{feature_set} - Regression R2')
            pdf.savefig(fig)
            plt.close(fig)
    # Classification
    for feature_set in ['radiomics', 'radiomics+clinical']:
        sub = metrics_df[(metrics_df['Task'] == 'classification') & (metrics_df['FeatureSet'] == feature_set)]
        if not sub.empty:
            for metric, color in [('Accuracy', 'skyblue'), ('F1', 'orange'), ('Precision', 'green'), ('Recall', 'purple')]:
                fig, ax = plt.subplots(figsize=(8, 5))
                sub.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color=color)
                plt.ylabel(metric)
                plt.title(f'{feature_set} - Classification {metric}')
                pdf.savefig(fig)
                plt.close(fig)

    # --- Scatter plots for regression predictions ---
    for feature_set in ['radiomics', 'radiomics+clinical']:
        for model in metrics_df[(metrics_df['Task'] == 'regression') & (metrics_df['FeatureSet'] == feature_set)]['Model']:
            fname = f'{feature_set}_{model}_regression_predictions.csv'
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                pred_df = pd.read_csv(fpath)
                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(pred_df['y_true'], pred_df['y_pred'], alpha=0.7)
                ax.plot([pred_df['y_true'].min(), pred_df['y_true'].max()], [pred_df['y_true'].min(), pred_df['y_true'].max()], 'r--', label='y=x')
                ax.set_xlabel('True Last mRS')
                ax.set_ylabel('Predicted Last mRS')
                ax.set_title(f'{feature_set} - {model} Regression: True vs Predicted')
                ax.legend()
                pdf.savefig(fig)
                plt.close(fig)

    # --- Confusion matrices for classification predictions ---
    for feature_set in ['radiomics', 'radiomics+clinical']:
        for model in metrics_df[(metrics_df['Task'] == 'classification') & (metrics_df['FeatureSet'] == feature_set)]['Model']:
            fname = f'{feature_set}_{model}_classification_predictions.csv'
            fpath = os.path.join(results_dir, fname)
            if os.path.exists(fpath):
                pred_df = pd.read_csv(fpath)
                cm = confusion_matrix(pred_df['y_true'], pred_df['y_pred'])
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                ax.set_title(f'{feature_set} - {model} Classification: Confusion Matrix')
                pdf.savefig(fig)
                plt.close(fig)

print(f'All model comparison plots and tables saved to {report_pdf}') 