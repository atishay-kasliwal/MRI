#!/usr/bin/env python3
"""
mRS Score Prediction Models
Predicts mRS scores using multiple ML models with 80/20 train/test split
Binary classification: Good outcome (mRS 0-2) vs Poor outcome (mRS 3-5)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def predict_mrs_scores():
    print("=== MRS SCORE PREDICTION MODELS ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients, {len(df.columns)} features")
    
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
    missing_before = X.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    
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
    
    missing_after = X.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")
    
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
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'XGBoost-style (Gradient Boosting)': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    }
    
    # Train and evaluate models
    print(f"\n🤖 Training and evaluating models...")
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"     CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"     Test AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")
    
    # Print results summary
    print(f"\n📊 MODEL PERFORMANCE SUMMARY")
    print(f"{'Model':<25} {'CV AUC':<12} {'Test AUC':<10} {'Accuracy':<10} {'F1':<8}")
    print("-" * 70)
    
    for name, result in results.items():
        cv_auc = result['cv_scores'].mean()
        cv_std = result['cv_scores'].std()
        test_auc = result['auc']
        accuracy = result['accuracy']
        f1 = result['f1']
        print(f"{name:<25} {cv_auc:.3f}±{cv_std:.3f}   {test_auc:<10.3f} {accuracy:<10.3f} {f1:<8.3f}")
    
    # Find best model (by AUC)
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Test AUC: {best_result['auc']:.3f}")
    print(f"   Test Accuracy: {best_result['accuracy']:.3f}")
    print(f"   Test F1-Score: {best_result['f1']:.3f}")
    
    # Detailed classification report for best model
    print(f"\n📊 CLASSIFICATION REPORT (Best Model)")
    print(classification_report(y_test, best_result['y_pred'], target_names=['Poor Outcome', 'Good Outcome']))
    
    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print(f"\n📋 CONFUSION MATRIX (Best Model)")
    print(f"   True Negatives (Poor→Poor): {cm[0,0]}")
    print(f"   False Positives (Poor→Good): {cm[0,1]}")
    print(f"   False Negatives (Good→Poor): {cm[1,0]}")
    print(f"   True Positives (Good→Good): {cm[1,1]}")
    
    # Feature importance for tree-based models
    if hasattr(best_result['model'], 'feature_importances_'):
        print(f"\n🔍 TOP 10 FEATURE IMPORTANCE (Best Model)")
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': best_result['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'PatientID': df_analysis.iloc[-len(y_test):]['PatientID'].values,
        'Actual_90day_mRS': df_analysis.iloc[-len(y_test):][target_col].values,
        'Actual_Binary': y_test.values,
        'Predicted_Binary': best_result['y_pred'],
        'Predicted_Probability': best_result['y_pred_proba']
    })
    predictions_df.to_csv('mrs_predictions.csv', index=False)
    print(f"   Saved predictions to: mrs_predictions.csv")
    
    # Save model comparison
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'CV_AUC_Mean': [results[name]['cv_scores'].mean() for name in results.keys()],
        'CV_AUC_Std': [results[name]['cv_scores'].std() for name in results.keys()],
        'Test_AUC': [results[name]['auc'] for name in results.keys()],
        'Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'Precision': [results[name]['precision'] for name in results.keys()],
        'Recall': [results[name]['recall'] for name in results.keys()],
        'F1_Score': [results[name]['f1'] for name in results.keys()]
    }).sort_values('Test_AUC', ascending=False)
    
    comparison_df.to_csv('mrs_model_comparison.csv', index=False)
    print(f"   Saved model comparison to: mrs_model_comparison.csv")
    
    # Save feature importance if available
    if hasattr(best_result['model'], 'feature_importances_'):
        feature_importance.to_csv('mrs_feature_importance.csv', index=False)
        print(f"   Saved feature importance to: mrs_feature_importance.csv")
    
    print(f"\n✅ MRS PREDICTION ANALYSIS COMPLETE!")
    print(f"   Best model: {best_model_name}")
    print(f"   Best Test AUC: {best_result['auc']:.3f}")
    print(f"   Best Test Accuracy: {best_result['accuracy']:.3f}")
    
    return results, comparison_df

if __name__ == "__main__":
    predict_mrs_scores() 