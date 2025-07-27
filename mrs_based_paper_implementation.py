#!/usr/bin/env python3
"""
mRS-Based Paper Implementation
Following the original meningioma paper methodology but using mRS 0-2 vs 3-5 as target
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
    """Load radiomics and clinical data, prepare mRS-based target"""
    
    print("Loading and preparing mRS-based data...")
    
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

def prepare_mrs_target(matched_data):
    """Prepare mRS 0-2 vs 3-5 target variable (equivalent to Ki-67 < 5% vs ≥ 5%)"""
    
    print("\nPreparing mRS-based target variable (0-2 vs 3-5)...")
    
    # Clean mRS data
    mrs_columns = ['Baseline mRS', 'Discharge mRS', '90 days mRS', 'Last mRS']
    
    for col in mrs_columns:
        if col in matched_data.columns:
            matched_data[col] = pd.to_numeric(matched_data[col], errors='coerce')
    
    # Create mRS 0-2 vs 3-5 target (equivalent to Ki-67 < 5% vs ≥ 5%)
    # mRS 0-2 = Good outcome (independent), mRS 3-5 = Poor outcome (dependent)
    
    targets = {}
    
    # 1. Last mRS (primary target - equivalent to Ki-67 in original paper)
    if 'Last mRS' in matched_data.columns:
        last_mrs_clean = matched_data['Last mRS'].dropna()
        if len(last_mrs_clean) > 0:
            # mRS 0-2 = 0 (good outcome), mRS 3-5 = 1 (poor outcome)
            targets['Last_mRS_0_2_vs_3_5'] = (last_mrs_clean >= 3).astype(int)
            print(f"Last mRS 0-2 vs 3-5: {targets['Last_mRS_0_2_vs_3_5'].value_counts().to_dict()}")
            print(f"  Good outcome (mRS 0-2): {sum(targets['Last_mRS_0_2_vs_3_5'] == 0)} patients")
            print(f"  Poor outcome (mRS 3-5): {sum(targets['Last_mRS_0_2_vs_3_5'] == 1)} patients")
    
    # 2. 90 days mRS (secondary target)
    if '90 days mRS' in matched_data.columns:
        mrs_90_clean = matched_data['90 days mRS'].dropna()
        if len(mrs_90_clean) > 0:
            targets['90_days_mRS_0_2_vs_3_5'] = (mrs_90_clean >= 3).astype(int)
            print(f"90 days mRS 0-2 vs 3-5: {targets['90_days_mRS_0_2_vs_3_5'].value_counts().to_dict()}")
    
    # 3. Discharge mRS (tertiary target)
    if 'Discharge mRS' in matched_data.columns:
        discharge_mrs_clean = matched_data['Discharge mRS'].dropna()
        if len(discharge_mrs_clean) > 0:
            targets['Discharge_mRS_0_2_vs_3_5'] = (discharge_mrs_clean >= 3).astype(int)
            print(f"Discharge mRS 0-2 vs 3-5: {targets['Discharge_mRS_0_2_vs_3_5'].value_counts().to_dict()}")
    
    return targets, matched_data

def implement_paper_methodology_with_mrs(matched_data, target_name, target_values):
    """Implement the original paper methodology using mRS target"""
    
    print(f"\n=== IMPLEMENTING PAPER METHODOLOGY: {target_name} ===")
    
    # Select radiomics features (equivalent to original paper's 2520 features)
    radiomics_features = []
    for col in matched_data.columns:
        if any(modality in col for modality in ['T1_', 'DWI_', 'ADC_', 'FLAIR_', 'T2_', 'cross_modality_']):
            radiomics_features.append(col)
    
    # Select clinical features (equivalent to original paper's clinical variables)
    clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Hyper-lipidemia', 
                        'CHF', 'CAD', 'Hemoglobin A1c', 'Prior Stroke', 'Smoking hx', 
                        'Baseline mRS', 'ADMIT NIH', 'IVTPA']
    
    # Filter available features
    available_radiomics = [f for f in radiomics_features if f in matched_data.columns]
    available_clinical = [f for f in clinical_features if f in matched_data.columns]
    
    # Prepare data
    X_radiomics = matched_data[available_radiomics]
    X_clinical = matched_data[available_clinical]
    
    # Combine features (equivalent to original paper's feature matrix)
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
    print(f"Total features: {len(X.columns)}")
    print(f"Complete cases: {len(y)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Check if we have enough variation
    if len(y.unique()) < 2:
        print(f"Insufficient variation in target variable: {target_name}")
        return None
    
    # Split data into discovery and replication cohorts (75%/25% - same as original paper)
    X_discovery, X_replication, y_discovery, y_replication = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Discovery cohort: {len(y_discovery)} patients")
    print(f"Replication cohort: {len(y_replication)} patients")
    
    # Scale features (equivalent to original paper's z-scoring)
    scaler = StandardScaler()
    X_discovery_scaled = scaler.fit_transform(X_discovery)
    X_replication_scaled = scaler.transform(X_replication)
    
    # Step 1: Feature Selection using LASSO (same as original paper)
    print("\n1. Feature Selection using LASSO...")
    lasso = LogisticRegression(penalty='l1', solver='liblinear', 
                              random_state=42, max_iter=1000)
    lasso.fit(X_discovery_scaled, y_discovery)
    
    selected_features_mask = lasso.coef_[0] != 0
    selected_features = X.columns[selected_features_mask]
    
    print(f"Selected features: {sum(selected_features_mask)} out of {len(X.columns)}")
    print(f"Feature selection rate: {sum(selected_features_mask)/len(X.columns)*100:.1f}%")
    
    # Step 2: Train SVM on selected features (same as original paper)
    print("\n2. Training SVM classifier...")
    X_discovery_selected = X_discovery_scaled[:, selected_features_mask]
    X_replication_selected = X_replication_scaled[:, selected_features_mask]
    
    # Cross-validation for hyperparameter tuning (simplified version of nested CV)
    svm = SVC(kernel='linear', probability=True, random_state=42)
    
    # Grid search for C parameter (equivalent to Bayesian optimization)
    C_values = [0.1, 1, 10, 100]
    cv_scores = []
    
    for C in C_values:
        svm.C = C
        scores = cross_val_score(svm, X_discovery_selected, y_discovery, cv=5, scoring='roc_auc')
        cv_scores.append(scores.mean())
    
    best_C = C_values[np.argmax(cv_scores)]
    svm.C = best_C
    
    # Final training
    svm.fit(X_discovery_selected, y_discovery)
    
    print(f"Best C parameter: {best_C}")
    
    # Step 3: Evaluate model performance (same metrics as original paper)
    print("\n3. Evaluating model performance...")
    
    # Discovery cohort
    y_pred_discovery = svm.predict(X_discovery_selected)
    y_pred_proba_discovery = svm.predict_proba(X_discovery_selected)[:, 1]
    
    fpr_discovery, tpr_discovery, _ = roc_curve(y_discovery, y_pred_proba_discovery)
    auc_discovery = auc(fpr_discovery, tpr_discovery)
    
    tn, fp, fn, tp = confusion_matrix(y_discovery, y_pred_discovery).ravel()
    sensitivity_discovery = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_discovery = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Replication cohort
    y_pred_replication = svm.predict(X_replication_selected)
    y_pred_proba_replication = svm.predict_proba(X_replication_selected)[:, 1]
    
    fpr_replication, tpr_replication, _ = roc_curve(y_replication, y_pred_proba_replication)
    auc_replication = auc(fpr_replication, tpr_replication)
    
    tn, fp, fn, tp = confusion_matrix(y_replication, y_pred_replication).ravel()
    sensitivity_replication = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity_replication = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results (same format as original paper)
    print(f"\nDiscovery Cohort Results:")
    print(f"  AUC: {auc_discovery:.3f}")
    print(f"  Sensitivity: {sensitivity_discovery:.3f}")
    print(f"  Specificity: {specificity_discovery:.3f}")
    
    print(f"\nReplication Cohort Results:")
    print(f"  AUC: {auc_replication:.3f}")
    print(f"  Sensitivity: {sensitivity_replication:.3f}")
    print(f"  Specificity: {specificity_replication:.3f}")
    
    return {
        'discovery': {
            'fpr': fpr_discovery, 'tpr': tpr_discovery, 'auc': auc_discovery,
            'sensitivity': sensitivity_discovery, 'specificity': specificity_discovery,
            'y_pred': y_pred_discovery, 'y_pred_proba': y_pred_proba_discovery
        },
        'replication': {
            'fpr': fpr_replication, 'tpr': tpr_replication, 'auc': auc_replication,
            'sensitivity': sensitivity_replication, 'specificity': specificity_replication,
            'y_pred': y_pred_replication, 'y_pred_proba': y_pred_proba_replication
        },
        'selected_features': selected_features,
        'feature_importance': np.abs(svm.coef_[0]) if hasattr(svm, 'coef_') else None,
        'model': svm,
        'scaler': scaler,
        'n_patients': len(y)
    }

def create_paper_style_visualizations(results, matched_data, target_name):
    """Create visualizations in the style of the original paper"""
    
    print(f"\n4. Creating paper-style visualizations for {target_name}...")
    
    # Set style similar to original paper
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with multiple subplots (similar to original paper)
    fig = plt.figure(figsize=(20, 24))
    
    # 1. ROC Curves (Figure 4 from original paper)
    ax1 = plt.subplot(3, 4, 1)
    plt.plot(results['discovery']['fpr'], results['discovery']['tpr'],
             label=f'Discovery (AUC = {results["discovery"]["auc"]:.2f})', linewidth=2)
    plt.plot(results['replication']['fpr'], results['replication']['tpr'],
             label=f'Replication (AUC = {results["replication"]["auc"]:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(f'ROC Curves - {target_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Target Variable Distribution (equivalent to Ki-67 distribution)
    ax2 = plt.subplot(3, 4, 2)
    if 'Last mRS' in matched_data.columns:
        last_mrs = matched_data['Last mRS'].dropna()
        if len(last_mrs) > 0:
            plt.hist(last_mrs, bins=range(8), alpha=0.7, edgecolor='black')
            plt.axvline(x=2.5, color='red', linestyle='--', label='mRS = 2.5 threshold')
            plt.xlabel('Last mRS Score')
            plt.ylabel('Frequency')
            plt.title('Last mRS Distribution (Target Variable)')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    # 3. Age Distribution by Outcome (equivalent to age vs Ki-67)
    ax3 = plt.subplot(3, 4, 3)
    if 'Age' in matched_data.columns and 'Last mRS' in matched_data.columns:
        age_mrs = matched_data[['Age', 'Last mRS']].dropna()
        if len(age_mrs) > 0:
            good_outcome = age_mrs[age_mrs['Last mRS'] < 3]['Age']
            poor_outcome = age_mrs[age_mrs['Last mRS'] >= 3]['Age']
            
            plt.boxplot([good_outcome, poor_outcome], labels=['Good Outcome\n(mRS 0-2)', 'Poor Outcome\n(mRS 3-5)'])
            plt.ylabel('Age (years)')
            plt.title('Age Distribution by Outcome')
            plt.grid(True, alpha=0.3)
    
    # 4. NIHSS vs Outcome (equivalent to clinical variables vs Ki-67)
    ax4 = plt.subplot(3, 4, 4)
    if 'ADMIT NIH' in matched_data.columns and 'Last mRS' in matched_data.columns:
        nihss_mrs = matched_data[['ADMIT NIH', 'Last mRS']].dropna()
        if len(nihss_mrs) > 0:
            good_outcome = nihss_mrs[nihss_mrs['Last mRS'] < 3]['ADMIT NIH']
            poor_outcome = nihss_mrs[nihss_mrs['Last mRS'] >= 3]['ADMIT NIH']
            
            plt.boxplot([good_outcome, poor_outcome], labels=['Good Outcome\n(mRS 0-2)', 'Poor Outcome\n(mRS 3-5)'])
            plt.ylabel('Admission NIHSS')
            plt.title('NIHSS Distribution by Outcome')
            plt.grid(True, alpha=0.3)
    
    # 5. Feature Importance (equivalent to original paper's feature analysis)
    ax5 = plt.subplot(3, 4, 5)
    if results['feature_importance'] is not None and len(results['selected_features']) > 0:
        importance = results['feature_importance']
        selected_features_names = results['selected_features']
        
        if len(importance) == len(selected_features_names):
            # Sort by importance
            sorted_indices = np.argsort(importance)[-15:]  # Top 15
            top_features = [selected_features_names[i] for i in sorted_indices]
            top_importance = importance[sorted_indices]
            
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), [f.replace('_', ' ').replace('cross modality', 'CM') for f in top_features])
            plt.xlabel('Feature Importance (Absolute Coefficient)')
            plt.title('Top 15 Most Important Features')
            plt.grid(True, alpha=0.3)
    
    # 6. Confusion Matrix - Discovery
    ax6 = plt.subplot(3, 4, 6)
    cm_discovery = confusion_matrix(results['discovery']['y_pred'], 
                                   [1 if x >= 3 else 0 for x in matched_data['Last mRS'].dropna()[:len(results['discovery']['y_pred'])]])
    sns.heatmap(cm_discovery, annot=True, fmt='d', cmap='Blues', ax=ax6)
    ax6.set_title('Confusion Matrix - Discovery Cohort')
    ax6.set_xlabel('Predicted')
    ax6.set_ylabel('Actual')
    
    # 7. Confusion Matrix - Replication
    ax7 = plt.subplot(3, 4, 7)
    cm_replication = confusion_matrix(results['replication']['y_pred'], 
                                     [1 if x >= 3 else 0 for x in matched_data['Last mRS'].dropna()[len(results['discovery']['y_pred']):]])
    sns.heatmap(cm_replication, annot=True, fmt='d', cmap='Blues', ax=ax7)
    ax7.set_title('Confusion Matrix - Replication Cohort')
    ax7.set_xlabel('Predicted')
    ax7.set_ylabel('Actual')
    
    # 8. Sex Distribution by Outcome
    ax8 = plt.subplot(3, 4, 8)
    if 'Sex' in matched_data.columns and 'Last mRS' in matched_data.columns:
        sex_outcome = pd.crosstab(matched_data['Sex'], 
                                 (matched_data['Last mRS'] >= 3).astype(int))
        sex_outcome.plot(kind='bar', ax=ax8)
        plt.title('Sex Distribution by Outcome')
        plt.xlabel('Sex')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(['Good Outcome (mRS 0-2)', 'Poor Outcome (mRS 3-5)'])
        plt.grid(True, alpha=0.3)
    
    # 9. Treatment vs Outcome
    ax9 = plt.subplot(3, 4, 9)
    if 'IVTPA' in matched_data.columns and 'Last mRS' in matched_data.columns:
        treatment_outcome = pd.crosstab(matched_data['IVTPA'], 
                                       (matched_data['Last mRS'] >= 3).astype(int))
        treatment_outcome.plot(kind='bar', ax=ax9)
        plt.title('IVTPA Treatment by Outcome')
        plt.xlabel('IVTPA Treatment')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(['Good Outcome (mRS 0-2)', 'Poor Outcome (mRS 3-5)'])
        plt.grid(True, alpha=0.3)
    
    # 10. Feature Categories (equivalent to original paper's feature distribution)
    ax10 = plt.subplot(3, 4, 10)
    if results['selected_features'] is not None:
        categories = {}
        for feature in results['selected_features']:
            if feature.startswith('cross_modality_'):
                category = 'cross_modality'
            elif any(modality in feature for modality in ['T1_', 'DWI_', 'ADC_', 'FLAIR_', 'T2_']):
                category = feature.split('_')[0]
            else:
                category = 'clinical'
            categories[category] = categories.get(category, 0) + 1
        
        if categories:
            plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
            plt.title('Selected Features by Category')
    
    # 11. Year vs Outcome
    ax11 = plt.subplot(3, 4, 11)
    if 'Year' in matched_data.columns and 'Last mRS' in matched_data.columns:
        year_outcome = pd.crosstab(matched_data['Year'], 
                                  (matched_data['Last mRS'] >= 3).astype(int))
        year_outcome.plot(kind='bar', ax=ax11)
        plt.title('Outcome Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(['Good Outcome (mRS 0-2)', 'Poor Outcome (mRS 3-5)'])
        plt.grid(True, alpha=0.3)
    
    # 12. Baseline mRS vs Last mRS
    ax12 = plt.subplot(3, 4, 12)
    if 'Baseline mRS' in matched_data.columns and 'Last mRS' in matched_data.columns:
        baseline_last = matched_data[['Baseline mRS', 'Last mRS']].dropna()
        if len(baseline_last) > 0:
            plt.scatter(baseline_last['Baseline mRS'], baseline_last['Last mRS'], alpha=0.6)
            plt.plot([0, 6], [0, 6], 'r--', alpha=0.5, label='No change')
            plt.axhline(y=2.5, color='green', linestyle='--', alpha=0.7, label='mRS = 2.5 threshold')
            plt.xlabel('Baseline mRS')
            plt.ylabel('Last mRS')
            plt.title('Baseline vs Last mRS')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def save_paper_style_results(results, matched_data, target_name):
    """Save results in the style of the original paper"""
    
    print(f"\n5. Saving paper-style results for {target_name}...")
    
    # Create results summary (similar to original paper's Table 2)
    results_summary = {
        'discovery_auc': results['discovery']['auc'],
        'discovery_sensitivity': results['discovery']['sensitivity'],
        'discovery_specificity': results['discovery']['specificity'],
        'replication_auc': results['replication']['auc'],
        'replication_sensitivity': results['replication']['sensitivity'],
        'replication_specificity': results['replication']['specificity'],
        'total_patients': results['n_patients'],
        'selected_features_count': len(results['selected_features']),
        'good_outcome_count': sum(matched_data['Last mRS'].dropna() < 3),
        'poor_outcome_count': sum(matched_data['Last mRS'].dropna() >= 3)
    }
    
    # Save results summary (equivalent to original paper's results)
    with open(f'mrs_paper_implementation_results_{target_name}.txt', 'w') as f:
        f.write(f"=== MRS-BASED PAPER IMPLEMENTATION RESULTS: {target_name} ===\n\n")
        f.write("This implementation follows the original meningioma paper methodology\n")
        f.write("but uses mRS 0-2 vs 3-5 as the target variable instead of Ki-67.\n\n")
        
        f.write(f"Total patients: {results_summary['total_patients']}\n")
        f.write(f"Good outcome (mRS 0-2): {results_summary['good_outcome_count']}\n")
        f.write(f"Poor outcome (mRS 3-5): {results_summary['poor_outcome_count']}\n")
        f.write(f"Selected features: {results_summary['selected_features_count']}\n\n")
        
        f.write("DISCOVERY COHORT:\n")
        f.write(f"AUC: {results_summary['discovery_auc']:.3f}\n")
        f.write(f"Sensitivity: {results_summary['discovery_sensitivity']:.3f}\n")
        f.write(f"Specificity: {results_summary['discovery_specificity']:.3f}\n\n")
        
        f.write("REPLICATION COHORT:\n")
        f.write(f"AUC: {results_summary['replication_auc']:.3f}\n")
        f.write(f"Sensitivity: {results_summary['replication_sensitivity']:.3f}\n")
        f.write(f"Specificity: {results_summary['replication_specificity']:.3f}\n\n")
        
        f.write("SELECTED FEATURES:\n")
        for i, feature in enumerate(results['selected_features']):
            f.write(f"{i+1}. {feature}\n")
    
    # Save predictions (equivalent to original paper's predictions)
    predictions_df = matched_data[['PatientID', 'ANON_ID', 'Age', 'Sex', 'Last mRS']].copy()
    predictions_df['mRS_0_2_vs_3_5'] = (predictions_df['Last mRS'] >= 3).astype(int)
    
    # Add predictions (simplified approach)
    all_predictions = np.concatenate([results['discovery']['y_pred'], results['replication']['y_pred']])
    all_probabilities = np.concatenate([results['discovery']['y_pred_proba'], results['replication']['y_pred_proba']])
    
    predictions_df['Predicted_Outcome'] = all_predictions
    predictions_df['Predicted_Probability'] = all_probabilities
    
    predictions_df.to_csv(f'mrs_paper_implementation_predictions_{target_name}.csv', index=False)
    
    print(f"Results saved to:")
    print(f"  - mrs_paper_implementation_results_{target_name}.txt")
    print(f"  - mrs_paper_implementation_predictions_{target_name}.csv")

def main():
    """Main function for mRS-based paper implementation"""
    
    print("=== MRS-BASED PAPER IMPLEMENTATION ===\n")
    print("Following original meningioma paper methodology")
    print("Target: mRS 0-2 vs 3-5 (equivalent to Ki-67 < 5% vs ≥ 5%)\n")
    
    # Load and prepare data
    matched_data = load_and_prepare_mrs_data()
    
    # Prepare mRS targets
    targets, matched_data = prepare_mrs_target(matched_data)
    
    if not targets:
        print("No valid mRS targets found. Exiting.")
        return
    
    # Implement paper methodology for each target
    results = {}
    
    for target_name, target_values in targets.items():
        print(f"\n{'='*60}")
        result = implement_paper_methodology_with_mrs(matched_data, target_name, target_values)
        if result is not None:
            results[target_name] = result
            
            # Create visualizations
            fig = create_paper_style_visualizations(result, matched_data, target_name)
            fig.savefig(f'mrs_paper_implementation_{target_name}.png', dpi=300, bbox_inches='tight')
            
            # Save results
            save_paper_style_results(result, matched_data, target_name)
    
    if not results:
        print("No valid prediction models could be trained.")
        return
    
    # Create comparison plot (similar to original paper's Figure 4)
    n_models = len(results)
    fig2, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for i, (target_name, result) in enumerate(results.items()):
        axes[i].plot(result['discovery']['fpr'], result['discovery']['tpr'], 
                    label=f'Discovery (AUC = {result["discovery"]["auc"]:.2f})', linewidth=2)
        axes[i].plot(result['replication']['fpr'], result['replication']['tpr'], 
                    label=f'Replication (AUC = {result["replication"]["auc"]:.2f})', linewidth=2)
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[i].set_xlabel('1 - Specificity')
        axes[i].set_ylabel('Sensitivity')
        axes[i].set_title(f'{target_name}\nROC Curves')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('mrs_paper_implementation_roc_comparison.png', dpi=300, bbox_inches='tight')
    
    print("\n=== MRS-BASED PAPER IMPLEMENTATION COMPLETED SUCCESSFULLY ===")
    print("This implementation follows the original meningioma paper methodology")
    print("but uses mRS 0-2 vs 3-5 as the clinically relevant target variable.")
    print("\nFiles generated:")
    for target_name in results.keys():
        print(f"  - mrs_paper_implementation_{target_name}.png")
        print(f"  - mrs_paper_implementation_results_{target_name}.txt")
        print(f"  - mrs_paper_implementation_predictions_{target_name}.csv")
    print("  - mrs_paper_implementation_roc_comparison.png")
    
    return matched_data, results

if __name__ == "__main__":
    main() 