#!/usr/bin/env python3
"""
Multiple Sclerosis (MS) Radiomics Analysis
Specialized radiomics analysis for MS patients with MS-specific features and outcomes
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

def create_ms_synthetic_data(n_patients=100):
    """
    Create synthetic MS data for demonstration purposes
    This simulates what MS patient data would look like
    """
    print("Creating synthetic MS patient data for demonstration...")
    
    np.random.seed(42)
    
    # Generate synthetic patient data
    data = {
        'PatientID': [f'MS_Patient_{i:03d}' for i in range(n_patients)],
        'Age': np.random.normal(45, 12, n_patients).astype(int),
        'Sex': np.random.choice([1, 2], n_patients),  # 1=Male, 2=Female
        'MS_Type': np.random.choice(['RRMS', 'SPMS', 'PPMS', 'PRMS'], n_patients, p=[0.7, 0.15, 0.1, 0.05]),
        'Disease_Duration': np.random.exponential(8, n_patients).astype(int),
        'EDSS_Baseline': np.random.uniform(1.0, 6.5, n_patients),
        'EDSS_Followup': np.random.uniform(1.0, 7.0, n_patients),
        'Relapses_1_Year': np.random.poisson(1.5, n_patients),
        'T2_Lesion_Count': np.random.poisson(15, n_patients),
        'T1_Lesion_Count': np.random.poisson(8, n_patients),
        'Gadolinium_Enhancing': np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
        'Brain_Atrophy_Rate': np.random.normal(-0.5, 0.3, n_patients),
        'Treatment_Type': np.random.choice(['DMT', 'No_Treatment', 'Steroids'], n_patients, p=[0.6, 0.2, 0.2])
    }
    
    # Create MS-specific radiomics features
    ms_radiomics = {}
    
    # T2-FLAIR features (most important for MS)
    ms_radiomics.update({
        'T2FLAIR_Lesion_Volume': np.random.gamma(2, 5000, n_patients),
        'T2FLAIR_Lesion_Density': np.random.normal(0.3, 0.1, n_patients),
        'T2FLAIR_Mean_Intensity': np.random.normal(120, 30, n_patients),
        'T2FLAIR_Entropy': np.random.normal(6.5, 0.5, n_patients),
        'T2FLAIR_Texture_Energy': np.random.normal(0.15, 0.05, n_patients),
        'T2FLAIR_Texture_Contrast': np.random.normal(0.8, 0.2, n_patients),
        'T2FLAIR_Texture_Homogeneity': np.random.normal(0.7, 0.1, n_patients),
        'T2FLAIR_Texture_Correlation': np.random.normal(0.6, 0.15, n_patients),
        'T2FLAIR_Shape_Sphericity': np.random.normal(0.4, 0.1, n_patients),
        'T2FLAIR_Shape_Compactness': np.random.normal(0.3, 0.08, n_patients)
    })
    
    # T1 features
    ms_radiomics.update({
        'T1_Mean_Intensity': np.random.normal(80, 20, n_patients),
        'T1_Entropy': np.random.normal(5.8, 0.4, n_patients),
        'T1_Texture_Energy': np.random.normal(0.12, 0.04, n_patients),
        'T1_Texture_Contrast': np.random.normal(0.6, 0.15, n_patients),
        'T1_Texture_Homogeneity': np.random.normal(0.75, 0.08, n_patients),
        'T1_Texture_Correlation': np.random.normal(0.65, 0.12, n_patients)
    })
    
    # DTI features (important for MS)
    ms_radiomics.update({
        'DTI_FA_Mean': np.random.normal(0.4, 0.08, n_patients),
        'DTI_FA_Std': np.random.normal(0.15, 0.03, n_patients),
        'DTI_MD_Mean': np.random.normal(0.8, 0.12, n_patients),
        'DTI_MD_Std': np.random.normal(0.2, 0.04, n_patients),
        'DTI_RD_Mean': np.random.normal(0.6, 0.1, n_patients),
        'DTI_AD_Mean': np.random.normal(1.2, 0.18, n_patients)
    })
    
    # Cross-modality features
    ms_radiomics.update({
        'Cross_Modality_Intensity_Ratio': np.random.normal(1.5, 0.3, n_patients),
        'Cross_Modality_Texture_Correlation': np.random.normal(0.7, 0.15, n_patients),
        'Cross_Modality_Entropy_Difference': np.random.normal(0.7, 0.2, n_patients)
    })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    for key, values in ms_radiomics.items():
        df[key] = values
    
    # Create MS-specific outcomes
    # 1. Disease progression (EDSS increase > 1.0)
    df['Disease_Progression'] = (df['EDSS_Followup'] - df['EDSS_Baseline'] > 1.0).astype(int)
    
    # 2. High relapse rate (> 2 relapses in 1 year)
    df['High_Relapse_Rate'] = (df['Relapses_1_Year'] > 2).astype(int)
    
    # 3. Treatment response (synthetic based on features)
    treatment_score = (df['T2FLAIR_Lesion_Volume'] / 10000 + 
                      df['Relapses_1_Year'] * 0.3 + 
                      df['EDSS_Baseline'] * 0.2)
    df['Treatment_Response'] = (treatment_score < treatment_score.median()).astype(int)
    
    print(f"Created synthetic MS dataset with {n_patients} patients")
    print(f"Features: {len(df.columns)} total features")
    print(f"MS-specific radiomics: {len(ms_radiomics)} features")
    
    return df

def analyze_ms_data(df):
    """Analyze MS patient data and create visualizations"""
    
    print("\n=== MS PATIENT DATA ANALYSIS ===")
    
    # Basic demographics
    print(f"\nDemographics:")
    print(f"Total patients: {len(df)}")
    print(f"Age: {df['Age'].mean():.1f} ± {df['Age'].std():.1f} years")
    print(f"Sex distribution: {df['Sex'].value_counts().to_dict()}")
    
    print(f"\nMS Types:")
    print(df['MS_Type'].value_counts())
    
    print(f"\nDisease characteristics:")
    print(f"Disease duration: {df['Disease_Duration'].mean():.1f} ± {df['Disease_Duration'].std():.1f} years")
    print(f"Baseline EDSS: {df['EDSS_Baseline'].mean():.2f} ± {df['EDSS_Baseline'].std():.2f}")
    print(f"T2 lesion count: {df['T2_Lesion_Count'].mean():.1f} ± {df['T2_Lesion_Count'].std():.1f}")
    
    # Create visualizations
    fig = plt.figure(figsize=(20, 24))
    
    # 1. MS Type Distribution
    ax1 = plt.subplot(3, 4, 1)
    df['MS_Type'].value_counts().plot(kind='bar', ax=ax1)
    plt.title('MS Type Distribution')
    plt.xlabel('MS Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 2. Age Distribution by MS Type
    ax2 = plt.subplot(3, 4, 2)
    df.boxplot(column='Age', by='MS_Type', ax=ax2)
    plt.title('Age Distribution by MS Type')
    plt.suptitle('')
    plt.grid(True, alpha=0.3)
    
    # 3. EDSS Distribution
    ax3 = plt.subplot(3, 4, 3)
    plt.hist(df['EDSS_Baseline'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Baseline EDSS')
    plt.ylabel('Frequency')
    plt.title('Baseline EDSS Distribution')
    plt.grid(True, alpha=0.3)
    
    # 4. T2 Lesion Count Distribution
    ax4 = plt.subplot(3, 4, 4)
    plt.hist(df['T2_Lesion_Count'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('T2 Lesion Count')
    plt.ylabel('Frequency')
    plt.title('T2 Lesion Count Distribution')
    plt.grid(True, alpha=0.3)
    
    # 5. Disease Progression by MS Type
    ax5 = plt.subplot(3, 4, 5)
    progression_by_type = df.groupby('MS_Type')['Disease_Progression'].mean()
    progression_by_type.plot(kind='bar', ax=ax5)
    plt.title('Disease Progression Rate by MS Type')
    plt.xlabel('MS Type')
    plt.ylabel('Progression Rate')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. T2FLAIR Lesion Volume vs EDSS
    ax6 = plt.subplot(3, 4, 6)
    plt.scatter(df['T2FLAIR_Lesion_Volume'], df['EDSS_Baseline'], alpha=0.6)
    plt.xlabel('T2FLAIR Lesion Volume')
    plt.ylabel('Baseline EDSS')
    plt.title('Lesion Volume vs EDSS')
    plt.grid(True, alpha=0.3)
    
    # 7. DTI FA vs Disease Duration
    ax7 = plt.subplot(3, 4, 7)
    plt.scatter(df['Disease_Duration'], df['DTI_FA_Mean'], alpha=0.6)
    plt.xlabel('Disease Duration (years)')
    plt.ylabel('DTI FA Mean')
    plt.title('Disease Duration vs DTI FA')
    plt.grid(True, alpha=0.3)
    
    # 8. Treatment Response by MS Type
    ax8 = plt.subplot(3, 4, 8)
    response_by_type = df.groupby('MS_Type')['Treatment_Response'].mean()
    response_by_type.plot(kind='bar', ax=ax8)
    plt.title('Treatment Response Rate by MS Type')
    plt.xlabel('MS Type')
    plt.ylabel('Response Rate')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 9. Radiomics Feature Correlation
    ax9 = plt.subplot(3, 4, 9)
    radiomics_features = [col for col in df.columns if any(modality in col for modality in ['T2FLAIR', 'T1', 'DTI'])]
    correlation_matrix = df[radiomics_features[:8]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=ax9)
    plt.title('Radiomics Feature Correlation')
    
    # 10. Outcome Distribution
    ax10 = plt.subplot(3, 4, 10)
    outcomes = ['Disease_Progression', 'High_Relapse_Rate', 'Treatment_Response']
    outcome_counts = [df[outcome].sum() for outcome in outcomes]
    plt.bar(outcomes, outcome_counts)
    plt.title('Outcome Distribution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 11. Age vs Disease Progression
    ax11 = plt.subplot(3, 4, 11)
    df.boxplot(column='Age', by='Disease_Progression', ax=ax11)
    plt.title('Age by Disease Progression')
    plt.suptitle('')
    plt.grid(True, alpha=0.3)
    
    # 12. Treatment Type Distribution
    ax12 = plt.subplot(3, 4, 12)
    df['Treatment_Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax12)
    plt.title('Treatment Type Distribution')
    
    plt.tight_layout()
    return fig

def implement_ms_prediction_model(df, target_variable='Disease_Progression'):
    """Implement prediction model for MS outcomes"""
    
    print(f"\n=== MS PREDICTION MODEL: {target_variable} ===")
    
    # Select features
    radiomics_features = [col for col in df.columns if any(modality in col for modality in ['T2FLAIR', 'T1', 'DTI', 'Cross_Modality'])]
    clinical_features = ['Age', 'Sex', 'Disease_Duration', 'EDSS_Baseline', 'T2_Lesion_Count', 'T1_Lesion_Count', 'Gadolinium_Enhancing', 'Relapses_1_Year']
    
    # Filter available features
    available_radiomics = [f for f in radiomics_features if f in df.columns]
    available_clinical = [f for f in clinical_features if f in df.columns]
    
    # Prepare data
    X_radiomics = df[available_radiomics]
    X_clinical = df[available_clinical]
    y = df[target_variable]
    
    # Combine features
    X_combined = pd.concat([X_radiomics, X_clinical], axis=1)
    
    print(f"Radiomics features: {len(available_radiomics)}")
    print(f"Clinical features: {len(available_clinical)}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.25, random_state=42, stratify=y
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
    selected_features = X_combined.columns[selected_features_mask]
    
    print(f"Selected features: {sum(selected_features_mask)} out of {len(X_combined.columns)}")
    
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
        'y_pred_proba': y_pred_proba
    }

def main():
    """Main function for MS radiomics analysis"""
    
    print("=== MULTIPLE SCLEROSIS RADIOMICS ANALYSIS ===\n")
    
    # Create synthetic MS data
    ms_data = create_ms_synthetic_data(n_patients=100)
    
    # Analyze MS data
    fig1 = analyze_ms_data(ms_data)
    fig1.savefig('ms_data_analysis.png', dpi=300, bbox_inches='tight')
    
    # Implement prediction models for different outcomes
    outcomes = ['Disease_Progression', 'High_Relapse_Rate', 'Treatment_Response']
    results = {}
    
    for outcome in outcomes:
        print(f"\n{'='*50}")
        results[outcome] = implement_ms_prediction_model(ms_data, outcome)
    
    # Create comparison plot
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, outcome in enumerate(outcomes):
        result = results[outcome]
        axes[i].plot(result['fpr'], result['tpr'], 
                    label=f'{outcome} (AUC = {result["auc"]:.2f})', linewidth=2)
        axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[i].set_xlabel('1 - Specificity')
        axes[i].set_ylabel('Sensitivity')
        axes[i].set_title(f'{outcome} Prediction')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig('ms_prediction_results.png', dpi=300, bbox_inches='tight')
    
    # Save results
    with open('ms_analysis_results.txt', 'w') as f:
        f.write("=== MS RADIOMICS ANALYSIS RESULTS ===\n\n")
        f.write(f"Total patients: {len(ms_data)}\n")
        f.write(f"MS types: {ms_data['MS_Type'].value_counts().to_dict()}\n\n")
        
        for outcome in outcomes:
            result = results[outcome]
            f.write(f"{outcome} Prediction:\n")
            f.write(f"  AUC: {result['auc']:.3f}\n")
            f.write(f"  Sensitivity: {result['sensitivity']:.3f}\n")
            f.write(f"  Specificity: {result['specificity']:.3f}\n")
            f.write(f"  Selected features: {len(result['selected_features'])}\n\n")
    
    # Save MS data
    ms_data.to_csv('ms_synthetic_data.csv', index=False)
    
    print("\n=== MS ANALYSIS COMPLETED SUCCESSFULLY ===")
    print("Files generated:")
    print("  - ms_data_analysis.png")
    print("  - ms_prediction_results.png")
    print("  - ms_analysis_results.txt")
    print("  - ms_synthetic_data.csv")
    
    return ms_data, results

if __name__ == "__main__":
    main() 