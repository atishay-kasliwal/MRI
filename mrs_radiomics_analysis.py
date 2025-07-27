#!/usr/bin/env python3
"""
mRS (Modified Rankin Scale) Radiomics Analysis
Focused analysis on mRS outcomes using real clinical data and radiomics features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_mrs_data():
    """Load radiomics and clinical data, focus on mRS outcomes"""
    
    print("Loading and preparing mRS-focused data...")
    
    # Load patient-level radiomics data
    radiomics_df = pd.read_csv('combined_patient_level_radiomics_data.csv')
    
    # Load clinical data
    clinical_2020 = pd.read_csv('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/MRI SCAN- MRN NUMBER.xlsx - Copy of 2020_Patients.csv')
    clinical_2021 = pd.read_csv('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/MRI SCAN- MRN NUMBER.xlsx - Copy of 2021_Patients.csv')
    clinical_2022 = pd.read_csv('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/MRI SCAN- MRN NUMBER.xlsx - Copy of 2022_Patients.csv')
    
    # Extract ANON IDs
    radiomics_df['ANON_ID'] = radiomics_df['PatientID'].str.extract(r'ANON(\d+)')
    clinical_2020['ANON_ID'] = clinical_2020['MRN ANON'].str.extract(r'ANON(\d+)')
    clinical_2021['ANON_ID'] = clinical_2021['ANON MRN '].str.extract(r'ANON(\d+)')
    clinical_2022['ANON_ID'] = clinical_2022['MRN ANON'].str.extract(r'ANON(\d+)')
    
    # Combine clinical data
    clinical_combined = pd.concat([clinical_2020, clinical_2021, clinical_2022], ignore_index=True)
    
    # Match radiomics with clinical data
    matched_data = pd.merge(radiomics_df, clinical_combined, on='ANON_ID', how='inner')
    
    print(f"Matched patients: {len(matched_data)}")
    
    return matched_data

def prepare_mrs_targets(matched_data):
    """Prepare different mRS-based target variables"""
    
    print("\nPreparing mRS-based target variables...")
    
    # Clean mRS data
    mrs_columns = ['Baseline mRS', 'Discharge mRS', '90 days mRS', 'Last mRS']
    
    for col in mrs_columns:
        if col in matched_data.columns:
            # Convert to numeric, handling non-numeric values
            matched_data[col] = pd.to_numeric(matched_data[col], errors='coerce')
    
    # Create different mRS-based outcomes
    targets = {}
    
    # 1. Good vs Poor outcome (mRS 0-2 vs 3-6) - Last mRS
    if 'Last mRS' in matched_data.columns:
        last_mrs_clean = matched_data['Last mRS'].dropna()
        if len(last_mrs_clean) > 0:
            targets['Last_mRS_Good_vs_Poor'] = (last_mrs_clean >= 3).astype(int)
            print(f"Last mRS Good vs Poor: {targets['Last_mRS_Good_vs_Poor'].value_counts().to_dict()}")
    
    # 2. Good vs Poor outcome - 90 days mRS
    if '90 days mRS' in matched_data.columns:
        mrs_90_clean = matched_data['90 days mRS'].dropna()
        if len(mrs_90_clean) > 0:
            targets['90_days_mRS_Good_vs_Poor'] = (mrs_90_clean >= 3).astype(int)
            print(f"90 days mRS Good vs Poor: {targets['90_days_mRS_Good_vs_Poor'].value_counts().to_dict()}")
    
    # 3. Discharge mRS Good vs Poor
    if 'Discharge mRS' in matched_data.columns:
        discharge_mrs_clean = matched_data['Discharge mRS'].dropna()
        if len(discharge_mrs_clean) > 0:
            targets['Discharge_mRS_Good_vs_Poor'] = (discharge_mrs_clean >= 3).astype(int)
            print(f"Discharge mRS Good vs Poor: {targets['Discharge_mRS_Good_vs_Poor'].value_counts().to_dict()}")
    
    # 4. mRS improvement (Baseline to Last mRS)
    if 'Baseline mRS' in matched_data.columns and 'Last mRS' in matched_data.columns:
        baseline_last = matched_data[['Baseline mRS', 'Last mRS']].dropna()
        if len(baseline_last) > 0:
            mrs_improvement = baseline_last['Last mRS'] < baseline_last['Baseline mRS']
            targets['mRS_Improvement'] = mrs_improvement.astype(int)
            print(f"mRS Improvement: {targets['mRS_Improvement'].value_counts().to_dict()}")
    
    # 5. Severe disability (mRS 4-6)
    if 'Last mRS' in matched_data.columns:
        last_mrs_clean = matched_data['Last mRS'].dropna()
        if len(last_mrs_clean) > 0:
            targets['Last_mRS_Severe_Disability'] = (last_mrs_clean >= 4).astype(int)
            print(f"Last mRS Severe Disability: {targets['Last_mRS_Severe_Disability'].value_counts().to_dict()}")
    
    return targets, matched_data

def analyze_mrs_data(matched_data, targets):
    """Analyze mRS data and create visualizations"""
    
    print("\n=== MRS DATA ANALYSIS ===")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 24))
    
    # 1. mRS Distribution Over Time
    ax1 = plt.subplot(3, 4, 1)
    mrs_columns = ['Baseline mRS', 'Discharge mRS', '90 days mRS', 'Last mRS']
    available_mrs = [col for col in mrs_columns if col in matched_data.columns]
    
    if len(available_mrs) > 0:
        mrs_data = []
        labels = []
        for col in available_mrs:
            clean_data = matched_data[col].dropna()
            if len(clean_data) > 0:
                mrs_data.append(clean_data)
                labels.append(col.replace(' mRS', ''))
        
        if mrs_data:
            plt.boxplot(mrs_data, labels=labels)
            plt.title('mRS Distribution Over Time')
            plt.ylabel('mRS Score')
            plt.grid(True, alpha=0.3)
    
    # 2. Last mRS Distribution
    ax2 = plt.subplot(3, 4, 2)
    if 'Last mRS' in matched_data.columns:
        last_mrs = matched_data['Last mRS'].dropna()
        if len(last_mrs) > 0:
            plt.hist(last_mrs, bins=range(8), alpha=0.7, edgecolor='black')
            plt.axvline(x=2.5, color='red', linestyle='--', label='mRS = 2.5 threshold')
            plt.xlabel('Last mRS Score')
            plt.ylabel('Frequency')
            plt.title('Last mRS Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 3. 90 days mRS Distribution
    ax3 = plt.subplot(3, 4, 3)
    if '90 days mRS' in matched_data.columns:
        mrs_90 = matched_data['90 days mRS'].dropna()
        if len(mrs_90) > 0:
            plt.hist(mrs_90, bins=range(8), alpha=0.7, edgecolor='black')
            plt.axvline(x=2.5, color='red', linestyle='--', label='mRS = 2.5 threshold')
            plt.xlabel('90 days mRS Score')
            plt.ylabel('Frequency')
            plt.title('90 Days mRS Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 4. Baseline vs Last mRS
    ax4 = plt.subplot(3, 4, 4)
    if 'Baseline mRS' in matched_data.columns and 'Last mRS' in matched_data.columns:
        baseline_last = matched_data[['Baseline mRS', 'Last mRS']].dropna()
        if len(baseline_last) > 0:
            plt.scatter(baseline_last['Baseline mRS'], baseline_last['Last mRS'], alpha=0.6)
            plt.plot([0, 6], [0, 6], 'r--', alpha=0.5, label='No change')
            plt.xlabel('Baseline mRS')
            plt.ylabel('Last mRS')
            plt.title('Baseline vs Last mRS')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 5. Age vs Last mRS
    ax5 = plt.subplot(3, 4, 5)
    if 'Age' in matched_data.columns and 'Last mRS' in matched_data.columns:
        age_mrs = matched_data[['Age', 'Last mRS']].dropna()
        if len(age_mrs) > 0:
            plt.scatter(age_mrs['Age'], age_mrs['Last mRS'], alpha=0.6)
            plt.xlabel('Age')
            plt.ylabel('Last mRS')
            plt.title('Age vs Last mRS')
            plt.grid(True, alpha=0.3)
    
    # 6. Sex vs Last mRS
    ax6 = plt.subplot(3, 4, 6)
    if 'Sex' in matched_data.columns and 'Last mRS' in matched_data.columns:
        sex_mrs = matched_data[['Sex', 'Last mRS']].dropna()
        if len(sex_mrs) > 0:
            sex_mrs.boxplot(column='Last mRS', by='Sex', ax=ax6)
            plt.title('Last mRS by Sex')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 7. NIHSS vs Last mRS
    ax7 = plt.subplot(3, 4, 7)
    if 'ADMIT NIH' in matched_data.columns and 'Last mRS' in matched_data.columns:
        nihss_mrs = matched_data[['ADMIT NIH', 'Last mRS']].dropna()
        if len(nihss_mrs) > 0:
            plt.scatter(nihss_mrs['ADMIT NIH'], nihss_mrs['Last mRS'], alpha=0.6)
            plt.xlabel('Admission NIHSS')
            plt.ylabel('Last mRS')
            plt.title('NIHSS vs Last mRS')
            plt.grid(True, alpha=0.3)
    
    # 8. Outcome Distribution
    ax8 = plt.subplot(3, 4, 8)
    if targets:
        outcome_names = list(targets.keys())
        outcome_counts = [len(target) for target in targets.values()]
        plt.bar(range(len(outcome_names)), outcome_counts)
        plt.xticks(range(len(outcome_names)), [name.replace('_', '\n') for name in outcome_names], rotation=45)
        plt.title('Available Outcomes')
        plt.ylabel('Number of Patients')
        plt.grid(True, alpha=0.3)
    
    # 9. mRS Change Over Time
    ax9 = plt.subplot(3, 4, 9)
    if 'Baseline mRS' in matched_data.columns and 'Last mRS' in matched_data.columns:
        baseline_last = matched_data[['Baseline mRS', 'Last mRS']].dropna()
        if len(baseline_last) > 0:
            mrs_change = baseline_last['Last mRS'] - baseline_last['Baseline mRS']
            plt.hist(mrs_change, bins=range(-6, 7), alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='No change')
            plt.xlabel('mRS Change (Last - Baseline)')
            plt.ylabel('Frequency')
            plt.title('mRS Change Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 10. Year vs Last mRS
    ax10 = plt.subplot(3, 4, 10)
    if 'Year' in matched_data.columns and 'Last mRS' in matched_data.columns:
        year_mrs = matched_data[['Year', 'Last mRS']].dropna()
        if len(year_mrs) > 0:
            year_mrs.boxplot(column='Last mRS', by='Year', ax=ax10)
            plt.title('Last mRS by Year')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 11. Treatment vs Last mRS
    ax11 = plt.subplot(3, 4, 11)
    if 'IVTPA' in matched_data.columns and 'Last mRS' in matched_data.columns:
        treatment_mrs = matched_data[['IVTPA', 'Last mRS']].dropna()
        if len(treatment_mrs) > 0:
            treatment_mrs.boxplot(column='Last mRS', by='IVTPA', ax=ax11)
            plt.title('Last mRS by IVTPA Treatment')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 12. TICI vs Last mRS
    ax12 = plt.subplot(3, 4, 12)
    if 'Final TICI' in matched_data.columns and 'Last mRS' in matched_data.columns:
        tici_mrs = matched_data[['Final TICI', 'Last mRS']].dropna()
        if len(tici_mrs) > 0:
            tici_mrs.boxplot(column='Last mRS', by='Final TICI', ax=ax12)
            plt.title('Last mRS by Final TICI')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def implement_mrs_prediction_model(matched_data, target_name, target_values):
    """Implement prediction model for mRS outcomes"""
    
    print(f"\n=== MRS PREDICTION MODEL: {target_name} ===")
    
    # Select radiomics features
    radiomics_features = []
    for col in matched_data.columns:
        if any(modality in col for modality in ['T1_', 'DWI_', 'ADC_', 'FLAIR_', 'T2_', 'cross_modality_']):
            radiomics_features.append(col)
    
    # Select clinical features
    clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Hyper-lipidemia', 
                        'CHF', 'CAD', 'Hemoglobin A1c', 'Prior Stroke', 'Smoking hx', 
                        'Baseline mRS', 'ADMIT NIH', 'IVTPA']
    
    # Filter available features
    available_radiomics = [f for f in radiomics_features if f in matched_data.columns]
    available_clinical = [f for f in clinical_features if f in matched_data.columns]
    
    # Prepare data
    X_radiomics = matched_data[available_radiomics]
    X_clinical = matched_data[available_clinical]
    
    # Combine features
    X_combined = pd.concat([X_radiomics, X_clinical], axis=1)
    
    # Remove rows with missing values
    complete_data = pd.concat([X_combined, target_values], axis=1).dropna()
    
    if len(complete_data) == 0:
        print(f"No complete data available for {target_name}")
        return None
    
    X = complete_data.iloc[:, :-1]
    y = complete_data.iloc[:, -1]
    
    print(f"Radiomics features: {len(available_radiomics)}")
    print(f"Clinical features: {len(available_clinical)}")
    print(f"Complete cases: {len(y)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Check if we have enough variation in target
    if len(y.unique()) < 2:
        print(f"Insufficient variation in target variable: {target_name}")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection with LASSO
    print("\n1. Feature Selection using LASSO...")
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
    lasso.fit(X_train_scaled, y_train)
    
    selected_features_mask = lasso.coef_[0] != 0
    selected_features = X.columns[selected_features_mask]
    
    print(f"Selected features: {sum(selected_features_mask)} out of {len(X.columns)}")
    
    # Train SVM on selected features
    print("\n2. Training SVM classifier...")
    X_train_selected = X_train_scaled[:, selected_features_mask]
    X_test_selected = X_test_scaled[:, selected_features_mask]
    
    svm = SVC(kernel='linear', probability=True, random_state=42)
    
    # Cross-validation for hyperparameter tuning
    C_values = [0.1, 1, 10, 100]
    cv_scores = []
    
    for C in C_values:
        svm.C = C
        scores = cross_val_score(svm, X_train_selected, y_train, cv=5, scoring='roc_auc')
        cv_scores.append(scores.mean())
    
    best_C = C_values[np.argmax(cv_scores)]
    svm.C = best_C
    svm.fit(X_train_selected, y_train)
    
    # Evaluate model
    y_pred = svm.predict(X_test_selected)
    y_pred_proba = svm.predict_proba(X_test_selected)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nModel Performance:")
    print(f"AUC: {auc_score:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    
    return {
        'model': svm,
        'scaler': scaler,
        'selected_features': selected_features,
        'auc': auc_score,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'tpr': tpr,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'n_patients': len(y)
    }

def main():
    """Main function for mRS radiomics analysis"""
    
    print("=== MRS (MODIFIED RANKIN SCALE) RADIOMICS ANALYSIS ===\n")
    
    # Load and prepare data
    matched_data = load_and_prepare_mrs_data()
    
    # Prepare mRS targets
    targets, matched_data = prepare_mrs_targets(matched_data)
    
    if not targets:
        print("No valid mRS targets found. Exiting.")
        return
    
    # Analyze mRS data
    fig1 = analyze_mrs_data(matched_data, targets)
    fig1.savefig('mrs_data_analysis.png', dpi=300, bbox_inches='tight')
    
    # Implement prediction models for different mRS outcomes
    results = {}
    
    for target_name, target_values in targets.items():
        print(f"\n{'='*60}")
        result = implement_mrs_prediction_model(matched_data, target_name, target_values)
        if result is not None:
            results[target_name] = result
    
    if not results:
        print("No valid prediction models could be trained.")
        return
    
    # Create comparison plot
    n_models = len(results)
    fig2, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for i, (target_name, result) in enumerate(results.items()):
        axes[i].plot(result['fpr'], result['tpr'], 
                    label=f'{target_name} (AUC = {result["auc"]:.2f})', linewidth=2)
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[i].set_xlabel('1 - Specificity')
        axes[i].set_ylabel('Sensitivity')
        axes[i].set_title(f'{target_name}\nPrediction')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('mrs_prediction_results.png', dpi=300, bbox_inches='tight')
    
    # Save results
    with open('mrs_analysis_results.txt', 'w') as f:
        f.write("=== MRS RADIOMICS ANALYSIS RESULTS ===\n\n")
        f.write(f"Total patients: {len(matched_data)}\n")
        f.write(f"Available mRS outcomes: {len(targets)}\n\n")
        
        for target_name, result in results.items():
            f.write(f"{target_name}:\n")
            f.write(f"  Patients: {result['n_patients']}\n")
            f.write(f"  AUC: {result['auc']:.3f}\n")
            f.write(f"  Sensitivity: {result['sensitivity']:.3f}\n")
            f.write(f"  Specificity: {result['specificity']:.3f}\n")
            f.write(f"  Selected features: {len(result['selected_features'])}\n\n")
    
    # Save mRS data
    mrs_data = matched_data[['PatientID', 'ANON_ID', 'Baseline mRS', 'Discharge mRS', '90 days mRS', 'Last mRS', 'Age', 'Sex']].copy()
    mrs_data.to_csv('mrs_clinical_data.csv', index=False)
    
    print("\n=== MRS ANALYSIS COMPLETED SUCCESSFULLY ===")
    print("Files generated:")
    print("  - mrs_data_analysis.png")
    print("  - mrs_prediction_results.png")
    print("  - mrs_analysis_results.txt")
    print("  - mrs_clinical_data.csv")
    
    return matched_data, results

if __name__ == "__main__":
    main() 