#!/usr/bin/env python3
"""
Simple mRS Score Prediction
Predicts mRS scores using basic ML models with 80/20 split
Binary classification: Good outcome (mRS 0-2) vs Poor outcome (mRS 3-5)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def simple_mrs_prediction():
    print("=== SIMPLE MRS PREDICTION ===\n")
    
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
    
    # Split data 80/20 with stratification
    print(f"\n📈 Data splitting (80/20 with stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Train and evaluate models
    print(f"\n🤖 Training and evaluating models...")
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'y_pred': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"     Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
    
    # Print results summary
    print(f"\n📊 MODEL PERFORMANCE SUMMARY")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<8}")
    print("-" * 65)
    
    for name, result in results.items():
        accuracy = result['accuracy']
        precision = result['precision']
        recall = result['recall']
        f1 = result['f1']
        print(f"{name:<20} {accuracy:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<8.3f}")
    
    # Find best model (by F1 score)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Test Accuracy: {best_result['accuracy']:.3f}")
    print(f"   Test Precision: {best_result['precision']:.3f}")
    print(f"   Test Recall: {best_result['recall']:.3f}")
    print(f"   Test F1-Score: {best_result['f1']:.3f}")
    
    # Confusion matrix for best model
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print(f"\n📋 CONFUSION MATRIX (Best Model)")
    print(f"   True Negatives (Poor→Poor): {cm[0,0]}")
    print(f"   False Positives (Poor→Good): {cm[0,1]}")
    print(f"   False Negatives (Good→Poor): {cm[1,0]}")
    print(f"   True Positives (Good→Good): {cm[1,1]}")
    
    # Feature importance for Random Forest
    if best_model_name == 'Random Forest':
        print(f"\n🔍 TOP 10 FEATURE IMPORTANCE (Random Forest)")
        # Get the Random Forest model
        rf_model = models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")
    
    print(f"\n✅ MRS PREDICTION COMPLETE!")
    print(f"   Best model: {best_model_name}")
    print(f"   Best Accuracy: {best_result['accuracy']:.3f}")
    print(f"   Best F1-Score: {best_result['f1']:.3f}")
    
    return results

if __name__ == "__main__":
    simple_mrs_prediction() 