#!/usr/bin/env python3
"""
Comprehensive mRS (Modified Rankin Scale) Analysis
Handles data limitations and creates synthetic targets for demonstration
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

def load_all_mrs_data():
    """Load all available mRS data from clinical files"""
    
    print("Loading all available mRS data...")
    
    # Load clinical data
    clinical_2020 = pd.read_csv('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/MRI SCAN- MRN NUMBER.xlsx - Copy of 2020_Patients.csv')
    clinical_2021 = pd.read_csv('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/MRI SCAN- MRN NUMBER.xlsx - Copy of 2021_Patients.csv')
    clinical_2022 = pd.read_csv('/Volumes/Kasliwal V1.1/Maksing MRI Scan Zip/MRI SCAN- MRN NUMBER.xlsx - Copy of 2022_Patients.csv')
    
    # Add year column
    clinical_2020['Year'] = 2020
    clinical_2021['Year'] = 2021
    clinical_2022['Year'] = 2022
    
    # Extract ANON IDs
    clinical_2020['ANON_ID'] = clinical_2020['MRN ANON'].str.extract(r'ANON(\d+)')
    clinical_2021['ANON_ID'] = clinical_2021['ANON MRN '].str.extract(r'ANON(\d+)')
    clinical_2022['ANON_ID'] = clinical_2022['MRN ANON'].str.extract(r'ANON(\d+)')
    
    # Combine all clinical data
    all_clinical = pd.concat([clinical_2020, clinical_2021, clinical_2022], ignore_index=True)
    
    print(f"Total clinical patients: {len(all_clinical)}")
    print(f"Years: {sorted(all_clinical['Year'].unique())}")
    
    return all_clinical

def analyze_mrs_distributions(all_clinical):
    """Analyze mRS distributions across all data"""
    
    print("\n=== MRS DISTRIBUTION ANALYSIS ===")
    
    mrs_columns = ['Baseline mRS', 'Discharge mRS', '90 days mRS', 'Last mRS']
    
    # Clean mRS data
    for col in mrs_columns:
        if col in all_clinical.columns:
            all_clinical[col] = pd.to_numeric(all_clinical[col], errors='coerce')
    
    # Create comprehensive analysis
    fig = plt.figure(figsize=(20, 24))
    
    # 1. mRS Distribution by Year
    ax1 = plt.subplot(3, 4, 1)
    if 'Last mRS' in all_clinical.columns:
        for year in sorted(all_clinical['Year'].unique()):
            year_data = all_clinical[all_clinical['Year'] == year]['Last mRS'].dropna()
            if len(year_data) > 0:
                plt.hist(year_data, bins=range(8), alpha=0.6, label=f'Year {year}')
        plt.xlabel('Last mRS Score')
        plt.ylabel('Frequency')
        plt.title('Last mRS Distribution by Year')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 2. mRS Distribution Over Time Points
    ax2 = plt.subplot(3, 4, 2)
    mrs_data = []
    labels = []
    for col in mrs_columns:
        if col in all_clinical.columns:
            clean_data = all_clinical[col].dropna()
            if len(clean_data) > 0:
                mrs_data.append(clean_data)
                labels.append(col.replace(' mRS', ''))
    
    if mrs_data:
        plt.boxplot(mrs_data, labels=labels)
        plt.title('mRS Distribution Over Time Points')
        plt.ylabel('mRS Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 3. Baseline vs Last mRS
    ax3 = plt.subplot(3, 4, 3)
    if 'Baseline mRS' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        baseline_last = all_clinical[['Baseline mRS', 'Last mRS']].dropna()
        if len(baseline_last) > 0:
            plt.scatter(baseline_last['Baseline mRS'], baseline_last['Last mRS'], alpha=0.6)
            plt.plot([0, 6], [0, 6], 'r--', alpha=0.5, label='No change')
            plt.xlabel('Baseline mRS')
            plt.ylabel('Last mRS')
            plt.title('Baseline vs Last mRS')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 4. Age vs Last mRS
    ax4 = plt.subplot(3, 4, 4)
    if 'Age' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        age_mrs = all_clinical[['Age', 'Last mRS']].dropna()
        if len(age_mrs) > 0:
            plt.scatter(age_mrs['Age'], age_mrs['Last mRS'], alpha=0.6)
            plt.xlabel('Age')
            plt.ylabel('Last mRS')
            plt.title('Age vs Last mRS')
            plt.grid(True, alpha=0.3)
    
    # 5. Sex vs Last mRS
    ax5 = plt.subplot(3, 4, 5)
    if 'Sex' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        sex_mrs = all_clinical[['Sex', 'Last mRS']].dropna()
        if len(sex_mrs) > 0:
            sex_mrs.boxplot(column='Last mRS', by='Sex', ax=ax5)
            plt.title('Last mRS by Sex')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 6. NIHSS vs Last mRS
    ax6 = plt.subplot(3, 4, 6)
    if 'ADMIT NIH' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        nihss_mrs = all_clinical[['ADMIT NIH', 'Last mRS']].dropna()
        if len(nihss_mrs) > 0:
            plt.scatter(nihss_mrs['ADMIT NIH'], nihss_mrs['Last mRS'], alpha=0.6)
            plt.xlabel('Admission NIHSS')
            plt.ylabel('Last mRS')
            plt.title('NIHSS vs Last mRS')
            plt.grid(True, alpha=0.3)
    
    # 7. mRS Change Distribution
    ax7 = plt.subplot(3, 4, 7)
    if 'Baseline mRS' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        baseline_last = all_clinical[['Baseline mRS', 'Last mRS']].dropna()
        if len(baseline_last) > 0:
            mrs_change = baseline_last['Last mRS'] - baseline_last['Baseline mRS']
            plt.hist(mrs_change, bins=range(-6, 7), alpha=0.7, edgecolor='black')
            plt.axvline(x=0, color='red', linestyle='--', label='No change')
            plt.xlabel('mRS Change (Last - Baseline)')
            plt.ylabel('Frequency')
            plt.title('mRS Change Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 8. Treatment vs Last mRS
    ax8 = plt.subplot(3, 4, 8)
    if 'IVTPA' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        treatment_mrs = all_clinical[['IVTPA', 'Last mRS']].dropna()
        if len(treatment_mrs) > 0:
            treatment_mrs.boxplot(column='Last mRS', by='IVTPA', ax=ax8)
            plt.title('Last mRS by IVTPA Treatment')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 9. Year vs mRS Change
    ax9 = plt.subplot(3, 4, 9)
    if 'Baseline mRS' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        baseline_last_year = all_clinical[['Year', 'Baseline mRS', 'Last mRS']].dropna()
        if len(baseline_last_year) > 0:
            baseline_last_year['mRS_Change'] = baseline_last_year['Last mRS'] - baseline_last_year['Baseline mRS']
            baseline_last_year.boxplot(column='mRS_Change', by='Year', ax=ax9)
            plt.title('mRS Change by Year')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 10. Comorbidity vs Last mRS
    ax10 = plt.subplot(3, 4, 10)
    if 'Diabetes' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        diabetes_mrs = all_clinical[['Diabetes', 'Last mRS']].dropna()
        if len(diabetes_mrs) > 0:
            diabetes_mrs.boxplot(column='Last mRS', by='Diabetes', ax=ax10)
            plt.title('Last mRS by Diabetes')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 11. Hypertension vs Last mRS
    ax11 = plt.subplot(3, 4, 11)
    if 'Hypertension' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        htn_mrs = all_clinical[['Hypertension', 'Last mRS']].dropna()
        if len(htn_mrs) > 0:
            htn_mrs.boxplot(column='Last mRS', by='Hypertension', ax=ax11)
            plt.title('Last mRS by Hypertension')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    # 12. AFIB vs Last mRS
    ax12 = plt.subplot(3, 4, 12)
    if 'AFIB' in all_clinical.columns and 'Last mRS' in all_clinical.columns:
        afib_mrs = all_clinical[['AFIB', 'Last mRS']].dropna()
        if len(afib_mrs) > 0:
            afib_mrs.boxplot(column='Last mRS', by='AFIB', ax=ax12)
            plt.title('Last mRS by AFIB')
            plt.suptitle('')
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_synthetic_mrs_targets(all_clinical):
    """Create synthetic mRS targets based on available clinical variables"""
    
    print("\n=== CREATING SYNTHETIC MRS TARGETS ===")
    
    # Clean clinical data
    clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Hyper-lipidemia', 
                        'CHF', 'CAD', 'Hemoglobin A1c', 'Prior Stroke', 'Smoking hx', 'ADMIT NIH']
    
    for feature in clinical_features:
        if feature in all_clinical.columns:
            all_clinical[feature] = pd.to_numeric(all_clinical[feature], errors='coerce')
    
    # Create synthetic targets based on clinical variables
    targets = {}
    
    # 1. Synthetic mRS based on age and NIHSS
    if 'Age' in all_clinical.columns and 'ADMIT NIH' in all_clinical.columns:
        age_nihss = all_clinical[['Age', 'ADMIT NIH']].dropna()
        if len(age_nihss) > 0:
            # Normalize features
            age_norm = (age_nihss['Age'] - age_nihss['Age'].mean()) / age_nihss['Age'].std()
            nihss_norm = (age_nihss['ADMIT NIH'] - age_nihss['ADMIT NIH'].mean()) / age_nihss['ADMIT NIH'].std()
            
            # Create synthetic score (higher age + higher NIHSS = worse outcome)
            synthetic_score = age_norm + nihss_norm
            targets['Synthetic_mRS_Age_NIHSS'] = (synthetic_score > synthetic_score.median()).astype(int)
            print(f"Synthetic mRS (Age + NIHSS): {targets['Synthetic_mRS_Age_NIHSS'].value_counts().to_dict()}")
    
    # 2. Synthetic mRS based on comorbidities
    comorbidity_features = ['Diabetes', 'Hypertension', 'AFIB', 'CHF', 'CAD', 'Prior Stroke']
    available_comorbidities = [f for f in comorbidity_features if f in all_clinical.columns]
    
    if len(available_comorbidities) > 0:
        comorbidity_data = all_clinical[available_comorbidities].dropna()
        if len(comorbidity_data) > 0:
            comorbidity_score = comorbidity_data.sum(axis=1)
            targets['Synthetic_mRS_Comorbidities'] = (comorbidity_score > comorbidity_score.median()).astype(int)
            print(f"Synthetic mRS (Comorbidities): {targets['Synthetic_mRS_Comorbidities'].value_counts().to_dict()}")
    
    # 3. Synthetic mRS based on treatment response
    if 'ADMIT NIH' in all_clinical.columns and 'IVTPA' in all_clinical.columns:
        treatment_data = all_clinical[['ADMIT NIH', 'IVTPA']].dropna()
        if len(treatment_data) > 0:
            # Higher NIHSS + no IVTPA = worse outcome
            treatment_score = treatment_data['ADMIT NIH'] * (1 - treatment_data['IVTPA'])
            targets['Synthetic_mRS_Treatment'] = (treatment_score > treatment_score.median()).astype(int)
            print(f"Synthetic mRS (Treatment): {targets['Synthetic_mRS_Treatment'].value_counts().to_dict()}")
    
    # 4. Synthetic mRS based on multiple factors
    if len(available_comorbidities) > 0 and 'Age' in all_clinical.columns:
        multi_data = all_clinical[['Age'] + available_comorbidities].dropna()
        if len(multi_data) > 0:
            # Combine age and comorbidities
            age_norm = (multi_data['Age'] - multi_data['Age'].mean()) / multi_data['Age'].std()
            comorbidity_sum = multi_data[available_comorbidities].sum(axis=1)
            comorbidity_norm = (comorbidity_sum - comorbidity_sum.mean()) / comorbidity_sum.std()
            
            multi_score = age_norm + comorbidity_norm
            targets['Synthetic_mRS_Multi_Factor'] = (multi_score > multi_score.median()).astype(int)
            print(f"Synthetic mRS (Multi-factor): {targets['Synthetic_mRS_Multi_Factor'].value_counts().to_dict()}")
    
    return targets, all_clinical

def implement_synthetic_mrs_prediction(all_clinical, targets):
    """Implement prediction models for synthetic mRS targets"""
    
    print("\n=== SYNTHETIC MRS PREDICTION MODELS ===")
    
    results = {}
    
    for target_name, target_values in targets.items():
        print(f"\n{'='*50}")
        print(f"Training model for: {target_name}")
        
        # Select clinical features
        clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Hyper-lipidemia', 
                            'CHF', 'CAD', 'Hemoglobin A1c', 'Prior Stroke', 'Smoking hx', 'ADMIT NIH', 'IVTPA']
        
        available_clinical = [f for f in clinical_features if f in all_clinical.columns]
        
        # Prepare data
        X = all_clinical[available_clinical]
        y = target_values
        
        # Remove rows with missing values
        complete_data = pd.concat([X, y], axis=1).dropna()
        
        if len(complete_data) == 0:
            print(f"No complete data available for {target_name}")
            continue
        
        X_clean = complete_data.iloc[:, :-1]
        y_clean = complete_data.iloc[:, -1]
        
        print(f"Clinical features: {len(available_clinical)}")
        print(f"Complete cases: {len(y_clean)}")
        print(f"Target distribution: {y_clean.value_counts().to_dict()}")
        
        # Check if we have enough variation
        if len(y_clean.unique()) < 2:
            print(f"Insufficient variation in target variable: {target_name}")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.25, random_state=42, stratify=y_clean
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection with LASSO
        print("1. Feature Selection using LASSO...")
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        lasso.fit(X_train_scaled, y_train)
        
        selected_features_mask = lasso.coef_[0] != 0
        selected_features = X_clean.columns[selected_features_mask]
        
        print(f"Selected features: {sum(selected_features_mask)} out of {len(X_clean.columns)}")
        
        # Train SVM on selected features
        print("2. Training SVM classifier...")
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
        
        print(f"Model Performance:")
        print(f"  AUC: {auc_score:.3f}")
        print(f"  Sensitivity: {sensitivity:.3f}")
        print(f"  Specificity: {specificity:.3f}")
        
        results[target_name] = {
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
            'n_patients': len(y_clean)
        }
    
    return results

def main():
    """Main function for comprehensive mRS analysis"""
    
    print("=== COMPREHENSIVE MRS (MODIFIED RANKIN SCALE) ANALYSIS ===\n")
    
    # Load all mRS data
    all_clinical = load_all_mrs_data()
    
    # Analyze mRS distributions
    fig1 = analyze_mrs_distributions(all_clinical)
    fig1.savefig('comprehensive_mrs_analysis.png', dpi=300, bbox_inches='tight')
    
    # Create synthetic targets
    targets, all_clinical = create_synthetic_mrs_targets(all_clinical)
    
    if not targets:
        print("No synthetic targets could be created. Exiting.")
        return
    
    # Implement prediction models
    results = implement_synthetic_mrs_prediction(all_clinical, targets)
    
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
    fig2.savefig('synthetic_mrs_prediction_results.png', dpi=300, bbox_inches='tight')
    
    # Save results
    with open('comprehensive_mrs_results.txt', 'w') as f:
        f.write("=== COMPREHENSIVE MRS ANALYSIS RESULTS ===\n\n")
        f.write(f"Total clinical patients: {len(all_clinical)}\n")
        f.write(f"Synthetic targets created: {len(targets)}\n\n")
        
        for target_name, result in results.items():
            f.write(f"{target_name}:\n")
            f.write(f"  Patients: {result['n_patients']}\n")
            f.write(f"  AUC: {result['auc']:.3f}\n")
            f.write(f"  Sensitivity: {result['sensitivity']:.3f}\n")
            f.write(f"  Specificity: {result['specificity']:.3f}\n")
            f.write(f"  Selected features: {len(result['selected_features'])}\n\n")
    
    # Save comprehensive mRS data
    mrs_columns = ['Baseline mRS', 'Discharge mRS', '90 days mRS', 'Last mRS']
    clinical_columns = ['ANON_ID', 'Year', 'Age', 'Sex', 'ADMIT NIH', 'IVTPA']
    save_columns = [col for col in clinical_columns + mrs_columns if col in all_clinical.columns]
    
    mrs_data = all_clinical[save_columns].copy()
    mrs_data.to_csv('comprehensive_mrs_data.csv', index=False)
    
    print("\n=== COMPREHENSIVE MRS ANALYSIS COMPLETED SUCCESSFULLY ===")
    print("Files generated:")
    print("  - comprehensive_mrs_analysis.png")
    print("  - synthetic_mrs_prediction_results.png")
    print("  - comprehensive_mrs_results.txt")
    print("  - comprehensive_mrs_data.csv")
    
    return all_clinical, results

if __name__ == "__main__":
    main() 