import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def run_pipeline(year_filter, label):
    # Path to merged data
    MERGED_CSV = 'data/radiomics/merged_radiomics_clinical.csv'
    RESULTS_DIR = 'results'
    MODELS_DIR = 'models'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load merged data
    data = pd.read_csv(MERGED_CSV)
    print(f"Loaded merged data: {len(data)} rows")

    # Filter by year if specified
    if year_filter is not None:
        if 'Year' not in data.columns:
            raise ValueError('Year column not found in merged data!')
        data = data[data['Year'].isin(year_filter)].reset_index(drop=True)
        print(f"Rows for {label}: {len(data)}")
    else:
        print(f"Rows for all years: {len(data)}")

    # Drop rows with missing Last mRS
    data = data.dropna(subset=['Last mRS'])
    print(f"Rows after dropping missing Last mRS: {len(data)}")

    # Prepare features and target
    drop_cols = ['PatientID', 'MRN ANON', 'Last mRS', 'Modality']
    X = data.drop(columns=[col for col in drop_cols if col in data.columns])
    y = data['Last mRS']

    # Convert all columns to numeric where possible, coerce errors to NaN
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill missing values (if any)
    X = X.fillna(X.mean())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(MODELS_DIR, f'random_forest_last_mrs_{label}.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Predict and evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # For classification-style metrics, round predictions to nearest integer (0-6)
    def round_and_clip(arr):
        return np.clip(np.round(arr), 0, 6).astype(int)
    train_pred_cls = round_and_clip(train_pred)
    test_pred_cls = round_and_clip(test_pred)
    y_train_cls = round_and_clip(y_train)
    y_test_cls = round_and_clip(y_test)

    # Classification metrics (test)
    test_acc = accuracy_score(y_test_cls, test_pred_cls)
    test_f1 = f1_score(y_test_cls, test_pred_cls, average='weighted')
    test_precision = precision_score(y_test_cls, test_pred_cls, average='weighted')
    test_recall = recall_score(y_test_cls, test_pred_cls, average='weighted')
    test_report = classification_report(y_test_cls, test_pred_cls, digits=3)
    test_cm = confusion_matrix(y_test_cls, test_pred_cls)

    # Save predictions
    test_pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': test_pred})
    test_pred_path = os.path.join(RESULTS_DIR, f'last_mrs_predictions_{label}.csv')
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Test predictions saved to {test_pred_path}")

    # Save evaluation metrics
    metrics_path = os.path.join(RESULTS_DIR, f'last_mrs_metrics_{label}.csv')
    with open(metrics_path, 'w') as f:
        f.write('split,MAE,R2,Accuracy,F1,Precision,Recall\n')
        f.write(f'train,{train_mae},{train_r2},,,,' + '\n')
        f.write(f'test,{test_mae},{test_r2},{test_acc},{test_f1},{test_precision},{test_recall}\n')
    print(f"Metrics saved to {metrics_path}")

    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns

    # Save feature importances
    fi_df = pd.DataFrame({'feature': feature_names[indices], 'importance': importances[indices]})
    fi_path = os.path.join(RESULTS_DIR, f'last_mrs_feature_importances_{label}.csv')
    fi_df.to_csv(fi_path, index=False)
    print(f"Feature importances saved to {fi_path}")

    # PDF Report
    pdf_path = os.path.join(RESULTS_DIR, f'lastmrs_predictions_{label}_report.pdf')
    with PdfPages(pdf_path) as pdf:
        # Summary page
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        summary = f"Last mRS Prediction Report ({label})\n\n"
        summary += f"Train samples: {len(y_train)}\nTest samples: {len(y_test)}\n"
        summary += f"Train MAE: {train_mae:.3f}\nTrain R2: {train_r2:.3f}\n"
        summary += f"Test MAE: {test_mae:.3f}\nTest R2: {test_r2:.3f}\n"
        summary += f"Test Accuracy: {test_acc:.3f}\nTest F1: {test_f1:.3f}\nTest Precision: {test_precision:.3f}\nTest Recall: {test_recall:.3f}\n"
        ax.text(0.1, 0.8, summary, fontsize=14, va='top')
        pdf.savefig(fig)
        plt.close(fig)

        # Classification report page (test)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        ax.text(0.01, 0.99, test_report, fontsize=12, va='top', family='monospace')
        plt.title('Test Classification Report', fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

        # Confusion matrix (test)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Test Confusion Matrix (rounded mRS)')
        pdf.savefig(fig)
        plt.close(fig)

        # True vs Predicted scatter (test)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, test_pred, alpha=0.7)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction')
        ax.set_xlabel('True mRS')
        ax.set_ylabel('Predicted mRS')
        ax.set_title(f'Test: True vs Predicted mRS ({label})')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # Distribution plots (test)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(y_test, color='blue', label='True', kde=True, stat='density', bins=7, alpha=0.5)
        sns.histplot(test_pred, color='orange', label='Predicted', kde=True, stat='density', bins=7, alpha=0.5)
        ax.set_title(f'Test: Distribution of True vs Predicted mRS ({label})')
        ax.set_xlabel('mRS')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # Feature importance plot
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(range(20), importances[indices[:20]], align='center')
        ax.set_xticks(range(20))
        ax.set_xticklabels([feature_names[i] for i in indices[:20]], rotation=90)
        ax.set_title('Feature Importances (Top 20)')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF report saved to {pdf_path}\n")

if __name__ == '__main__':
    # 2020 only
    run_pipeline([2020], '2020')
    # 2021 only
    run_pipeline([2021], '2021')
    # Combined 2020+2021
    run_pipeline([2020, 2021], '2020_2021') 