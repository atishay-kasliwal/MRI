#!/usr/bin/env python3
"""
Advanced mRS Prediction with Cross-Validation and Hyperparameter Tuning
Enhanced version with comprehensive model evaluation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def advanced_mrs_prediction():
    print("=== ADVANCED MRS PREDICTION WITH CROSS-VALIDATION ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Focus on 90-day mRS prediction
    target_col = '90 days mRS'
    print(f"🎯 Target: {target_col}")
    
    # Filter patients with valid mRS data
    df_analysis = df[df[target_col].notna()].copy()
    print(f"   Patients with {target_col}: {len(df_analysis)}")
    
    # Create binary target (Good: 0-2, Poor: 3-5)
    df_analysis['mRS_binary'] = (pd.to_numeric(df_analysis[target_col], errors='coerce') <= 2).astype(int)
    good_outcome = len(df_analysis[df_analysis['mRS_binary'] == 1])
    poor_outcome = len(df_analysis[df_analysis['mRS_binary'] == 0])
    
    print(f"   Good outcome (mRS 0-2): {good_outcome} patients")
    print(f"   Poor outcome (mRS 3-5): {poor_outcome} patients")
    print(f"   Class balance: {good_outcome/(good_outcome+poor_outcome)*100:.1f}% vs {poor_outcome/(good_outcome+poor_outcome)*100:.1f}%")
    
    # Prepare features
    radiomics_cols = [col for col in df_analysis.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    clinical_features = ['Age', 'Sex', 'Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    available_clinical = [col for col in clinical_features if col in df_analysis.columns]
    
    print(f"\n🔬 Preparing features...")
    print(f"   Radiomics features: {len(radiomics_cols)}")
    print(f"   Clinical features: {len(available_clinical)}")
    
    feature_cols = radiomics_cols + available_clinical
    X = df_analysis[feature_cols].copy()
    y = df_analysis['mRS_binary']
    
    # Handle missing values
    print(f"\n📊 Handling missing values...")
    
    # Fill missing values
    radiomics_missing = X[radiomics_cols].isnull().sum().sum()
    if radiomics_missing > 0:
        X[radiomics_cols] = X[radiomics_cols].fillna(X[radiomics_cols].median())
        print(f"   Filled {radiomics_missing} missing radiomics values with median")
    
    clinical_missing = X[available_clinical].isnull().sum().sum()
    if clinical_missing > 0:
        for col in available_clinical:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 0
                X[col] = X[col].fillna(mode_val)
        print(f"   Filled {clinical_missing} missing clinical values with mode")
    
    # Encode categorical variables
    if 'Sex' in X.columns:
        X['Sex'] = X['Sex'].map({'M': 1, 'F': 0}).fillna(0)
    
    boolean_cols = ['Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    for col in boolean_cols:
        if col in X.columns:
            X[col] = X[col].astype(str).str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0)
    
    # Feature selection
    print(f"\n🎯 Feature selection...")
    selector = SelectKBest(score_func=f_classif, k=min(100, len(feature_cols)))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"   Selected {len(selected_features)} features")
    
    # Split data 80/20 with stratification
    print(f"\n📈 Data splitting (80/20 with stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} patients")
    print(f"   Test set: {len(X_test)} patients")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Hyperparameter tuning for Random Forest
    print(f"\n🔧 Hyperparameter tuning for Random Forest...")
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    rf_grid_search.fit(X_train_scaled, y_train)
    print(f"   Best parameters: {rf_grid_search.best_params_}")
    print(f"   Best CV F1-score: {rf_grid_search.best_score_:.3f}")
    
    # Hyperparameter tuning for Logistic Regression
    print(f"\n🔧 Hyperparameter tuning for Logistic Regression...")
    lr_param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr_grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        lr_param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    lr_grid_search.fit(X_train_scaled, y_train)
    print(f"   Best parameters: {lr_grid_search.best_params_}")
    print(f"   Best CV F1-score: {lr_grid_search.best_score_:.3f}")
    
    # Train final models with best parameters
    print(f"\n🤖 Training final models with best parameters...")
    
    # Random Forest
    best_rf = rf_grid_search.best_estimator_
    rf_cv_scores = cross_val_score(best_rf, X_train_scaled, y_train, cv=cv, scoring='f1')
    best_rf.fit(X_train_scaled, y_train)
    rf_pred = best_rf.predict(X_test_scaled)
    rf_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
    
    # Logistic Regression
    best_lr = lr_grid_search.best_estimator_
    lr_cv_scores = cross_val_score(best_lr, X_train_scaled, y_train, cv=cv, scoring='f1')
    best_lr.fit(X_train_scaled, y_train)
    lr_pred = best_lr.predict(X_test_scaled)
    lr_pred_proba = best_lr.predict_proba(X_test_scaled)[:, 1]
    
    # Results summary
    print(f"\n📊 ADVANCED MODEL PERFORMANCE SUMMARY")
    print(f"{'Model':<25} {'CV F1':<12} {'Test F1':<10} {'Test AUC':<10} {'Test Acc':<10}")
    print("-" * 75)
    
    # Random Forest results
    rf_cv_f1 = rf_cv_scores.mean()
    rf_cv_std = rf_cv_scores.std()
    rf_test_f1 = f1_score(y_test, rf_pred)
    rf_test_auc = roc_auc_score(y_test, rf_pred_proba)
    rf_test_acc = accuracy_score(y_test, rf_pred)
    
    print(f"{'Random Forest (Tuned)':<25} {rf_cv_f1:.3f}±{rf_cv_std:.3f}   {rf_test_f1:<10.3f} {rf_test_auc:<10.3f} {rf_test_acc:<10.3f}")
    
    # Logistic Regression results
    lr_cv_f1 = lr_cv_scores.mean()
    lr_cv_std = lr_cv_scores.std()
    lr_test_f1 = f1_score(y_test, lr_pred)
    lr_test_auc = roc_auc_score(y_test, lr_pred_proba)
    lr_test_acc = accuracy_score(y_test, lr_pred)
    
    print(f"{'Logistic Regression (Tuned)':<25} {lr_cv_f1:.3f}±{lr_cv_std:.3f}   {lr_test_f1:<10.3f} {lr_test_auc:<10.3f} {lr_test_acc:<10.3f}")
    
    # Find best model
    if rf_test_f1 > lr_test_f1:
        best_model = best_rf
        best_model_name = "Random Forest"
        best_pred = rf_pred
        best_pred_proba = rf_pred_proba
        best_cv_f1 = rf_cv_f1
        best_test_f1 = rf_test_f1
        best_test_auc = rf_test_auc
    else:
        best_model = best_lr
        best_model_name = "Logistic Regression"
        best_pred = lr_pred
        best_pred_proba = lr_pred_proba
        best_cv_f1 = lr_cv_f1
        best_test_f1 = lr_test_f1
        best_test_auc = lr_test_auc
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   CV F1-Score: {best_cv_f1:.3f}")
    print(f"   Test F1-Score: {best_test_f1:.3f}")
    print(f"   Test AUC: {best_test_auc:.3f}")
    
    # Detailed classification report
    print(f"\n📊 CLASSIFICATION REPORT (Best Model)")
    print(classification_report(y_test, best_pred, target_names=['Poor Outcome', 'Good Outcome']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    print(f"\n📋 CONFUSION MATRIX (Best Model)")
    print(f"   True Negatives (Poor→Poor): {cm[0,0]}")
    print(f"   False Positives (Poor→Good): {cm[0,1]}")
    print(f"   False Negatives (Good→Poor): {cm[1,0]}")
    print(f"   True Positives (Good→Good): {cm[1,1]}")
    
    # Feature importance for Random Forest
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n🔍 TOP 15 FEATURE IMPORTANCE (Best Model)")
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")
    
    # Save results
    print(f"\n💾 Saving advanced results...")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'PatientID': df_analysis.iloc[-len(y_test):]['PatientID'].values,
        'Actual_90day_mRS': df_analysis.iloc[-len(y_test):][target_col].values,
        'Actual_Binary': y_test.values,
        'Predicted_Binary': best_pred,
        'Predicted_Probability': best_pred_proba
    })
    predictions_df.to_csv('advanced_mrs_predictions.csv', index=False)
    print(f"   Saved predictions to: advanced_mrs_predictions.csv")
    
    # Save model comparison
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest (Tuned)', 'Logistic Regression (Tuned)'],
        'CV_F1_Mean': [rf_cv_f1, lr_cv_f1],
        'CV_F1_Std': [rf_cv_std, lr_cv_std],
        'Test_F1': [rf_test_f1, lr_test_f1],
        'Test_AUC': [rf_test_auc, lr_test_auc],
        'Test_Accuracy': [rf_test_acc, lr_test_acc]
    })
    comparison_df.to_csv('advanced_model_comparison.csv', index=False)
    print(f"   Saved model comparison to: advanced_model_comparison.csv")
    
    # Save feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        feature_importance.to_csv('advanced_feature_importance.csv', index=False)
        print(f"   Saved feature importance to: advanced_feature_importance.csv")
    
    print(f"\n✅ ADVANCED MRS PREDICTION COMPLETE!")
    print(f"   Best model: {best_model_name}")
    print(f"   Best CV F1-Score: {best_cv_f1:.3f}")
    print(f"   Best Test F1-Score: {best_test_f1:.3f}")
    print(f"   Best Test AUC: {best_test_auc:.3f}")
    
    # Return results (handle case where feature_importance might not exist)
    if hasattr(best_model, 'feature_importances_'):
        return best_model, comparison_df, feature_importance
    else:
        return best_model, comparison_df, None

if __name__ == "__main__":
    advanced_mrs_prediction() 