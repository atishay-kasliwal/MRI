#!/usr/bin/env python3
"""
Model Performance Diagnostic Analysis
Comprehensive analysis to identify why models might not be performing optimally
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def model_performance_diagnostic():
    print("=== MODEL PERFORMANCE DIAGNOSTIC ANALYSIS ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Focus on 90-day mRS prediction
    target_col = '90 days mRS'
    
    # Clean the target variable
    df_clean = df.copy()
    non_numeric_mask = df_clean[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean = df_clean[~non_numeric_mask].copy()
    
    # Filter patients with valid mRS data
    df_analysis = df_clean[df_clean[target_col].notna()].copy()
    df_analysis['mRS_binary'] = (pd.to_numeric(df_analysis[target_col], errors='coerce') <= 2).astype(int)
    
    print(f"📋 DATASET OVERVIEW")
    print(f"   Total patients: {len(df_analysis)}")
    print(f"   Good outcome (mRS 0-2): {df_analysis['mRS_binary'].sum()} patients")
    print(f"   Poor outcome (mRS 3-5): {(df_analysis['mRS_binary'] == 0).sum()} patients")
    print(f"   Class balance: {df_analysis['mRS_binary'].mean()*100:.1f}% vs {(1-df_analysis['mRS_binary'].mean())*100:.1f}%")
    
    # Prepare features
    radiomics_cols = [col for col in df_analysis.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    available_clinical = [col for col in clinical_features if col in df_analysis.columns]
    
    print(f"\n🔬 FEATURE ANALYSIS")
    print(f"   Radiomics features: {len(radiomics_cols)}")
    print(f"   Clinical features: {len(available_clinical)}")
    
    feature_cols = radiomics_cols + available_clinical
    X = df_analysis[feature_cols].copy()
    y = df_analysis['mRS_binary']
    
    # Handle missing values and encode
    radiomics_missing = X[radiomics_cols].isnull().sum().sum()
    if radiomics_missing > 0:
        X[radiomics_cols] = X[radiomics_cols].fillna(X[radiomics_cols].median())
    
    clinical_missing = X[available_clinical].isnull().sum().sum()
    if clinical_missing > 0:
        for col in available_clinical:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 0
                X[col] = X[col].fillna(mode_val)
    
    # Encode categorical variables
    if 'Sex' in X.columns:
        X['Sex'] = X['Sex'].map({'M': 1, 'F': 0}).fillna(0)
    
    boolean_cols = ['Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    for col in boolean_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0)
    
    # Remove zero variance features
    zero_var_features = []
    for col in X.columns:
        if X[col].var() == 0:
            zero_var_features.append(col)
    
    if zero_var_features:
        X = X.drop(columns=zero_var_features)
        print(f"   Removed {len(zero_var_features)} zero variance features")
    
    print(f"\n⚠️  POTENTIAL PERFORMANCE ISSUES IDENTIFIED:")
    
    issues = []
    
    # 1. Sample Size Analysis
    print(f"\n📊 1. SAMPLE SIZE ANALYSIS")
    if len(df_analysis) < 100:
        issues.append("Small sample size")
        print(f"   ❌ Small sample size: {len(df_analysis)} patients")
        print(f"   💡 Recommendation: Need at least 100-200 patients for reliable ML")
    else:
        print(f"   ✅ Adequate sample size: {len(df_analysis)} patients")
    
    # 2. Class Imbalance Analysis
    print(f"\n⚖️  2. CLASS IMBALANCE ANALYSIS")
    good_outcome = df_analysis['mRS_binary'].sum()
    poor_outcome = (df_analysis['mRS_binary'] == 0).sum()
    balance_ratio = min(good_outcome, poor_outcome) / max(good_outcome, poor_outcome)
    
    print(f"   Good outcome: {good_outcome} patients")
    print(f"   Poor outcome: {poor_outcome} patients")
    print(f"   Balance ratio: {balance_ratio:.3f}")
    
    if balance_ratio < 0.3:
        issues.append("Severe class imbalance")
        print(f"   ❌ Severe class imbalance (ratio: {balance_ratio:.3f})")
        print(f"   💡 Recommendation: Use SMOTE, class weights, or collect more data")
    elif balance_ratio < 0.5:
        issues.append("Moderate class imbalance")
        print(f"   ⚠️  Moderate class imbalance (ratio: {balance_ratio:.3f})")
        print(f"   💡 Recommendation: Consider balancing techniques")
    else:
        print(f"   ✅ Good class balance (ratio: {balance_ratio:.3f})")
    
    # 3. Feature Quality Analysis
    print(f"\n🔬 3. FEATURE QUALITY ANALYSIS")
    
    # Check feature correlations with target
    feature_correlations = []
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            corr = abs(X[col].corr(y))
            feature_correlations.append((col, corr))
    
    feature_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Top 10 feature correlations with target:")
    for i, (feature, corr) in enumerate(feature_correlations[:10]):
        print(f"     {i+1:2d}. {feature:<40} {corr:.4f}")
    
    # Check if any features have high correlation
    high_corr_features = [f for f, c in feature_correlations if c > 0.3]
    if len(high_corr_features) == 0:
        issues.append("Low feature-target correlations")
        print(f"   ❌ No features with correlation > 0.3")
        print(f"   💡 Recommendation: Feature engineering or different features needed")
    else:
        print(f"   ✅ {len(high_corr_features)} features with good correlation (>0.3)")
    
    # 4. Feature Redundancy Analysis
    print(f"\n🔄 4. FEATURE REDUNDANCY ANALYSIS")
    
    # Check for highly correlated features
    X_numeric = X.select_dtypes(include=[np.number])
    corr_matrix = X_numeric.corr().abs()
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        issues.append("Feature redundancy")
        print(f"   ❌ {len(high_corr_pairs)} highly correlated feature pairs (>0.95)")
        print(f"   💡 Recommendation: Remove redundant features")
    else:
        print(f"   ✅ No highly redundant features found")
    
    # 5. Data Distribution Analysis
    print(f"\n📈 5. DATA DISTRIBUTION ANALYSIS")
    
    # Check for outliers in radiomics features
    outlier_counts = []
    for col in radiomics_cols[:10]:  # Check first 10 features
        if col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_counts.append(outliers)
    
    avg_outliers = np.mean(outlier_counts) if outlier_counts else 0
    outlier_percent = (avg_outliers / len(X)) * 100
    
    print(f"   Average outliers per feature: {avg_outliers:.1f} ({outlier_percent:.1f}%)")
    
    if outlier_percent > 10:
        issues.append("High outlier percentage")
        print(f"   ⚠️  High outlier percentage: {outlier_percent:.1f}%")
        print(f"   💡 Recommendation: Use robust scaling or outlier removal")
    else:
        print(f"   ✅ Acceptable outlier percentage: {outlier_percent:.1f}%")
    
    # 6. Model Complexity vs Data Size
    print(f"\n🤖 6. MODEL COMPLEXITY ANALYSIS")
    
    n_features = len(X.columns)
    n_samples = len(X)
    complexity_ratio = n_features / n_samples
    
    print(f"   Features: {n_features}")
    print(f"   Samples: {n_samples}")
    print(f"   Feature-to-sample ratio: {complexity_ratio:.3f}")
    
    if complexity_ratio > 1:
        issues.append("High feature-to-sample ratio")
        print(f"   ❌ High feature-to-sample ratio: {complexity_ratio:.3f}")
        print(f"   💡 Recommendation: Feature selection or more data needed")
    elif complexity_ratio > 0.5:
        issues.append("Moderate feature-to-sample ratio")
        print(f"   ⚠️  Moderate feature-to-sample ratio: {complexity_ratio:.3f}")
        print(f"   💡 Recommendation: Consider feature selection")
    else:
        print(f"   ✅ Good feature-to-sample ratio: {complexity_ratio:.3f}")
    
    # 7. Cross-validation Analysis
    print(f"\n🔄 7. CROSS-VALIDATION ANALYSIS")
    
    # Test with different feature selection methods
    selector_50 = SelectKBest(score_func=f_classif, k=50)
    selector_100 = SelectKBest(score_func=f_classif, k=100)
    selector_150 = SelectKBest(score_func=f_classif, k=150)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Test with different feature sets
    feature_sets = [
        ("50 features", selector_50),
        ("100 features", selector_100),
        ("150 features", selector_150),
        ("All features", None)
    ]
    
    print(f"   Cross-validation results with different feature sets:")
    for name, selector in feature_sets:
        if selector:
            X_selected = selector.fit_transform(X, y)
        else:
            X_selected = X
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, X_selected, y, cv=cv, scoring='f1')
        
        print(f"     {name}: F1 = {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # 8. Clinical Feature Analysis
    print(f"\n🏥 8. CLINICAL FEATURE ANALYSIS")
    
    for col in available_clinical:
        if col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                corr = abs(X[col].corr(y))
                print(f"   {col}: correlation = {corr:.3f}")
            else:
                # For categorical variables, check distribution
                good_dist = X[col][y == 1].value_counts()
                poor_dist = X[col][y == 0].value_counts()
                print(f"   {col}:")
                print(f"     Good outcome: {good_dist.to_dict()}")
                print(f"     Poor outcome: {poor_dist.to_dict()}")
    
    # 9. Summary and Recommendations
    print(f"\n📋 SUMMARY OF ISSUES FOUND:")
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print(f"   ✅ No major issues identified")
    
    print(f"\n💡 RECOMMENDATIONS FOR IMPROVEMENT:")
    
    recommendations = []
    
    if "Small sample size" in issues:
        recommendations.append("Collect more patient data (aim for 200+ patients)")
    
    if "Severe class imbalance" in issues or "Moderate class imbalance" in issues:
        recommendations.append("Use SMOTE, class weights, or collect more data from minority class")
    
    if "Low feature-target correlations" in issues:
        recommendations.append("Engineer new features or collect different radiomics features")
    
    if "Feature redundancy" in issues:
        recommendations.append("Remove highly correlated features")
    
    if "High outlier percentage" in issues:
        recommendations.append("Use robust scaling or outlier removal techniques")
    
    if "High feature-to-sample ratio" in issues:
        recommendations.append("Use aggressive feature selection or collect more data")
    
    recommendations.append("Try ensemble methods (voting, stacking)")
    recommendations.append("Use hyperparameter tuning with GridSearchCV")
    recommendations.append("Consider different algorithms (SVM, XGBoost, CatBoost)")
    recommendations.append("Implement cross-validation with more folds")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # 10. Expected Performance Limits
    print(f"\n🎯 EXPECTED PERFORMANCE LIMITS:")
    print(f"   Based on your dataset characteristics:")
    print(f"   • Sample size: {len(df_analysis)} patients")
    print(f"   • Feature count: {len(X.columns)} features")
    print(f"   • Class balance: {balance_ratio:.3f}")
    print(f"   • Best feature correlation: {feature_correlations[0][1]:.3f}")
    
    # Estimate expected performance
    expected_accuracy = min(0.85, 0.6 + (balance_ratio * 0.2) + (feature_correlations[0][1] * 0.3))
    expected_f1 = min(0.80, 0.5 + (balance_ratio * 0.2) + (feature_correlations[0][1] * 0.3))
    
    print(f"   • Expected accuracy range: {expected_accuracy:.3f}")
    print(f"   • Expected F1-score range: {expected_f1:.3f}")
    
    current_best_f1 = 0.727  # From our best model
    if current_best_f1 >= expected_f1:
        print(f"   ✅ Current performance ({current_best_f1:.3f}) meets expectations!")
    else:
        print(f"   ⚠️  Current performance ({current_best_f1:.3f}) below expectations ({expected_f1:.3f})")
    
    return issues, recommendations

if __name__ == "__main__":
    model_performance_diagnostic() 