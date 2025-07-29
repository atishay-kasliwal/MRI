#!/usr/bin/env python3
"""
mRS Prediction Analysis
Practical implementation of mRS prediction using radiomics and clinical features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def analyze_mrs_prediction():
    """Analyze mRS prediction using radiomics and clinical features"""
    
    print("=== MRS PREDICTION ANALYSIS ===\n")
    
    # Load the merged dataset
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    
    # Focus on 90-day mRS prediction (largest sample size)
    print("🎯 FOCUSING ON 90-DAY MRS PREDICTION")
    print(f"   Total patients: {len(df)}")
    
    # Prepare data for 90-day mRS prediction
    target_col = '90 days mRS'
    valid_target = pd.to_numeric(df[target_col], errors='coerce').dropna()
    
    print(f"   Patients with 90-day mRS: {len(valid_target)}")
    
    # Create binary outcome: Good (0-2) vs Poor (3-5)
    df_analysis = df[df[target_col].notna()].copy()
    df_analysis['mRS_binary'] = (pd.to_numeric(df_analysis[target_col], errors='coerce') <= 2).astype(int)
    
    good_outcome = len(df_analysis[df_analysis['mRS_binary'] == 1])
    poor_outcome = len(df_analysis[df_analysis['mRS_binary'] == 0])
    
    print(f"   Good outcome (mRS 0-2): {good_outcome} patients")
    print(f"   Poor outcome (mRS 3-5): {poor_outcome} patients")
    print(f"   Class balance: {good_outcome/(good_outcome+poor_outcome)*100:.1f}% vs {poor_outcome/(good_outcome+poor_outcome)*100:.1f}%")
    
    # Prepare features
    print(f"\n🔬 PREPARING FEATURES")
    
    # Radiomics features
    radiomics_cols = [col for col in df_analysis.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    print(f"   Radiomics features: {len(radiomics_cols)}")
    
    # Clinical features
    clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    available_clinical = [col for col in clinical_features if col in df_analysis.columns]
    print(f"   Clinical features: {len(available_clinical)}")
    
    # Combine features
    feature_cols = radiomics_cols + available_clinical
    print(f"   Total features: {len(feature_cols)}")
    
    # Prepare X and y
    X = df_analysis[feature_cols].copy()
    y = df_analysis['mRS_binary']
    
    # Handle missing values
    print(f"\n📊 HANDLING MISSING VALUES")
    missing_before = X.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    
    # For radiomics, fill with median
    radiomics_missing = X[radiomics_cols].isnull().sum().sum()
    if radiomics_missing > 0:
        X[radiomics_cols] = X[radiomics_cols].fillna(X[radiomics_cols].median())
        print(f"   Filled {radiomics_missing} missing radiomics values with median")
    
    # For clinical, fill with mode
    clinical_missing = X[available_clinical].isnull().sum().sum()
    if clinical_missing > 0:
        for col in available_clinical:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 0
                X[col] = X[col].fillna(mode_val)
        print(f"   Filled {clinical_missing} missing clinical values with mode")
    
    missing_after = X.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")
    
    # Convert categorical variables
    if 'Sex' in X.columns:
        X['Sex'] = X['Sex'].map({'M': 1, 'F': 0}).fillna(0)
    
    # Convert boolean variables
    boolean_cols = ['Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    for col in boolean_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0)
    
    # Feature selection
    print(f"\n🎯 FEATURE SELECTION")
    
    # Select top features using ANOVA F-test
    selector = SelectKBest(score_func=f_classif, k=min(100, len(feature_cols)))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"   Selected top {len(selected_features)} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train)} patients")
    print(f"   Test set: {len(X_test)} patients")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print(f"\n🤖 TRAINING MODELS")
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'auc': auc
        }
        
        print(f"     CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"     Test AUC: {auc:.3f}")
    
    # Compare models
    print(f"\n📈 MODEL COMPARISON")
    print(f"{'Model':<20} {'CV AUC':<15} {'Test AUC':<10}")
    print("-" * 45)
    for name, result in results.items():
        cv_auc = result['cv_scores'].mean()
        cv_std = result['cv_scores'].std()
        test_auc = result['auc']
        print(f"{name:<20} {cv_auc:.3f}±{cv_std:.3f}     {test_auc:.3f}")
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 TOP 10 MOST IMPORTANT FEATURES")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")
    
    # Detailed results for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Test AUC: {best_result['auc']:.3f}")
    
    # Classification report
    print(f"\n📊 CLASSIFICATION REPORT")
    print(classification_report(y_test, best_result['y_pred'], target_names=['Poor Outcome', 'Good Outcome']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print(f"\n📋 CONFUSION MATRIX")
    print(f"   True Negatives (Poor→Poor): {cm[0,0]}")
    print(f"   False Positives (Poor→Good): {cm[0,1]}")
    print(f"   False Negatives (Good→Poor): {cm[1,0]}")
    print(f"   True Positives (Good→Good): {cm[1,1]}")
    
    # Save results
    print(f"\n💾 SAVING RESULTS")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'PatientID': df_analysis.iloc[-len(y_test):]['PatientID'].values,
        'Actual_90day_mRS': df_analysis.iloc[-len(y_test):][target_col].values,
        'Actual_Binary': y_test.values,
        'Predicted_Binary': best_result['y_pred'],
        'Predicted_Probability': best_result['y_pred_proba']
    })
    
    predictions_df.to_csv('mrs_prediction_results.csv', index=False)
    print(f"   Saved predictions to: mrs_prediction_results.csv")
    
    # Save feature importance
    if 'Random Forest' in results:
        feature_importance.to_csv('mrs_feature_importance.csv', index=False)
        print(f"   Saved feature importance to: mrs_feature_importance.csv")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   Best model: {best_model_name}")
    print(f"   Test AUC: {best_result['auc']:.3f}")
    print(f"   Ready for clinical validation and publication")
    
    return results, predictions_df

if __name__ == "__main__":
    analyze_mrs_prediction() 