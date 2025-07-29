#!/usr/bin/env python3
"""
Data Quality Analysis for mRS Prediction
Analyze missing values, data distribution, and identify issues affecting model performance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_data_quality():
    print("=== DATA QUALITY ANALYSIS FOR MRS PREDICTION ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Check mRS columns
    mrs_columns = [col for col in df.columns if 'mrs' in col.lower()]
    print(f"\n🎯 MRS COLUMNS FOUND:")
    for col in mrs_columns:
        print(f"   {col}")
    
    # Analyze each mRS column
    print(f"\n📋 MRS DATA QUALITY ANALYSIS:")
    for col in mrs_columns:
        total_patients = len(df)
        valid_patients = df[col].notna().sum()
        missing_patients = df[col].isna().sum()
        missing_percent = (missing_patients / total_patients) * 100
        
        print(f"\n   {col}:")
        print(f"     Total patients: {total_patients}")
        print(f"     Valid patients: {valid_patients}")
        print(f"     Missing patients: {missing_patients} ({missing_percent:.1f}%)")
        
        if valid_patients > 0:
            # Check value distribution
            valid_values = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(valid_values) > 0:
                print(f"     Value range: {valid_values.min()}-{valid_values.max()}")
                print(f"     Value distribution: {valid_values.value_counts().sort_index().to_dict()}")
                
                # Check for binary classification
                good_outcome = len(valid_values[valid_values <= 2])
                poor_outcome = len(valid_values[valid_values >= 3])
                print(f"     Good outcome (0-2): {good_outcome} ({good_outcome/len(valid_values)*100:.1f}%)")
                print(f"     Poor outcome (3-5): {poor_outcome} ({poor_outcome/len(valid_values)*100:.1f}%)")
    
    # Focus on 90-day mRS
    target_col = '90 days mRS'
    print(f"\n🎯 DETAILED ANALYSIS: {target_col}")
    
    # Check data types and unique values
    print(f"   Data type: {df[target_col].dtype}")
    print(f"   Unique values: {df[target_col].unique()}")
    
    # Check for non-numeric values
    non_numeric = df[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    if non_numeric.sum() > 0:
        print(f"   Non-numeric values found: {df[target_col][non_numeric].unique()}")
    
    # Analyze radiomics features
    radiomics_cols = [col for col in df.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    print(f"\n🔬 RADIOMICS FEATURES ANALYSIS:")
    print(f"   Total radiomics features: {len(radiomics_cols)}")
    
    # Check missing values in radiomics
    radiomics_missing = df[radiomics_cols].isnull().sum().sum()
    total_radiomics_cells = len(df) * len(radiomics_cols)
    radiomics_missing_percent = (radiomics_missing / total_radiomics_cells) * 100
    print(f"   Missing radiomics values: {radiomics_missing} ({radiomics_missing_percent:.2f}%)")
    
    # Check for infinite values
    radiomics_infinite = np.isinf(df[radiomics_cols].select_dtypes(include=[np.number])).sum().sum()
    print(f"   Infinite values: {radiomics_infinite}")
    
    # Check for zero variance features
    zero_var_features = []
    for col in radiomics_cols:
        if df[col].var() == 0:
            zero_var_features.append(col)
    print(f"   Zero variance features: {len(zero_var_features)}")
    
    # Analyze clinical features
    clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    available_clinical = [col for col in clinical_features if col in df.columns]
    
    print(f"\n🏥 CLINICAL FEATURES ANALYSIS:")
    print(f"   Available clinical features: {len(available_clinical)}")
    for col in available_clinical:
        missing_count = df[col].isna().sum()
        missing_percent = (missing_count / len(df)) * 100
        print(f"   {col}: {missing_count} missing ({missing_percent:.1f}%)")
    
    # Check correlation between features and target
    print(f"\n🔗 FEATURE-TARGET CORRELATION ANALYSIS:")
    
    # Create clean dataset for correlation analysis
    df_clean = df[df[target_col].notna()].copy()
    df_clean['mRS_binary'] = (pd.to_numeric(df_clean[target_col], errors='coerce') <= 2).astype(int)
    
    if len(df_clean) > 0:
        # Check clinical feature correlations
        for col in available_clinical:
            if col in df_clean.columns:
                if df_clean[col].dtype in ['object', 'string']:
                    # For categorical variables, check distribution by outcome
                    good_outcome = df_clean[df_clean['mRS_binary'] == 1][col].value_counts()
                    poor_outcome = df_clean[df_clean['mRS_binary'] == 0][col].value_counts()
                    print(f"   {col} distribution by outcome:")
                    print(f"     Good outcome: {good_outcome.to_dict()}")
                    print(f"     Poor outcome: {poor_outcome.to_dict()}")
                else:
                    # For numeric variables, calculate correlation
                    correlation = df_clean[col].corr(df_clean['mRS_binary'])
                    print(f"   {col} correlation with mRS: {correlation:.3f}")
    
    # Identify potential issues
    print(f"\n⚠️  POTENTIAL ISSUES IDENTIFIED:")
    
    issues = []
    
    # Check sample size
    if len(df_clean) < 100:
        issues.append(f"Small sample size: {len(df_clean)} patients")
    
    # Check class balance
    if len(df_clean) > 0:
        good_outcome = df_clean['mRS_binary'].sum()
        poor_outcome = len(df_clean) - good_outcome
        balance_ratio = min(good_outcome, poor_outcome) / max(good_outcome, poor_outcome)
        if balance_ratio < 0.3:
            issues.append(f"Severe class imbalance: {good_outcome} vs {poor_outcome} patients")
    
    # Check feature quality
    if radiomics_missing_percent > 10:
        issues.append(f"High missing radiomics data: {radiomics_missing_percent:.1f}%")
    
    if len(zero_var_features) > 0:
        issues.append(f"Zero variance features: {len(zero_var_features)} features")
    
    # Check for data leakage
    if len(df_clean) > 0:
        # Check if any features are too highly correlated with target
        high_corr_features = []
        for col in radiomics_cols[:10]:  # Check first 10 features
            if df_clean[col].dtype in ['float64', 'int64']:
                corr = abs(df_clean[col].corr(df_clean['mRS_binary']))
                if corr > 0.8:
                    high_corr_features.append(col)
        
        if high_corr_features:
            issues.append(f"Potential data leakage: {len(high_corr_features)} highly correlated features")
    
    if issues:
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"   No major issues identified")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT:")
    
    recommendations = []
    
    if len(df_clean) < 100:
        recommendations.append("Collect more patient data to increase sample size")
    
    if radiomics_missing_percent > 10:
        recommendations.append("Improve radiomics extraction pipeline to reduce missing data")
    
    if len(zero_var_features) > 0:
        recommendations.append("Remove zero variance features before training")
    
    if len(df_clean) > 0:
        good_outcome = df_clean['mRS_binary'].sum()
        poor_outcome = len(df_clean) - good_outcome
        if min(good_outcome, poor_outcome) < 20:
            recommendations.append("Use techniques to handle class imbalance (SMOTE, class weights)")
    
    recommendations.append("Try different feature selection methods")
    recommendations.append("Experiment with different algorithms (XGBoost, CatBoost)")
    recommendations.append("Use ensemble methods to combine multiple models")
    recommendations.append("Implement cross-validation with more folds")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    return df_clean, issues, recommendations

if __name__ == "__main__":
    analyze_data_quality() 