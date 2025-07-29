#!/usr/bin/env python3
"""
Improved mRS Prediction Model
Addresses data quality issues and implements advanced techniques for better performance
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

def improved_mrs_prediction():
    print("=== IMPROVED MRS PREDICTION MODEL ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Focus on 90-day mRS prediction
    target_col = '90 days mRS'
    print(f"🎯 Target: {target_col}")
    
    # Clean the target variable - remove non-numeric values
    print(f"\n🧹 CLEANING TARGET VARIABLE...")
    df_clean = df.copy()
    
    # Remove non-numeric values from target
    non_numeric_mask = df_clean[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    non_numeric_values = df_clean[target_col][non_numeric_mask].unique()
    print(f"   Non-numeric values found: {non_numeric_values}")
    
    # Remove rows with non-numeric target values
    df_clean = df_clean[~non_numeric_mask].copy()
    print(f"   Removed {len(df) - len(df_clean)} patients with non-numeric mRS values")
    print(f"   Remaining patients: {len(df_clean)}")
    
    # Filter patients with valid mRS data
    df_analysis = df_clean[df_clean[target_col].notna()].copy()
    print(f"   Patients with valid {target_col}: {len(df_analysis)}")
    
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
    
    # Remove zero variance features
    print(f"\n🎯 Feature preprocessing...")
    zero_var_features = []
    for col in X.columns:
        if X[col].var() == 0:
            zero_var_features.append(col)
    
    if zero_var_features:
        X = X.drop(columns=zero_var_features)
        print(f"   Removed {len(zero_var_features)} zero variance features")
    
    # Advanced feature selection
    print(f"   Performing advanced feature selection...")
    
    # Method 1: Statistical feature selection
    selector1 = SelectKBest(score_func=f_classif, k=min(150, len(X.columns)))
    X_selected1 = selector1.fit_transform(X, y)
    selected_features1 = X.columns[selector1.get_support()].tolist()
    
    # Method 2: Recursive Feature Elimination
    rf_for_rfe = RandomForestClassifier(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf_for_rfe, n_features_to_select=min(100, len(X.columns)), step=10)
    X_selected2 = rfe.fit_transform(X, y)
    selected_features2 = X.columns[rfe.support_].tolist()
    
    # Combine both methods
    combined_features = list(set(selected_features1) & set(selected_features2))
    if len(combined_features) < 50:
        combined_features = selected_features1[:100]  # Fallback to statistical selection
    
    X_selected = X[combined_features]
    print(f"   Selected {len(combined_features)} features using combined methods")
    
    # Split data 80/20 with stratification
    print(f"\n📈 Data splitting (80/20 with stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} patients")
    print(f"   Test set: {len(X_test)} patients")
    
    # Handle class imbalance
    print(f"\n⚖️  Handling class imbalance...")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"   Class weights: {class_weight_dict}")
    
    # Apply SMOTE for oversampling
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE - Good outcome: {sum(y_train_balanced == 1)}, Poor outcome: {sum(y_train_balanced == 0)}")
    
    # Scale features using robust scaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define models with class weights
    models = {
        'Random Forest (Balanced)': RandomForestClassifier(
            n_estimators=200, 
            max_depth=None, 
            min_samples_split=5, 
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting (Balanced)': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'Logistic Regression (Balanced)': LogisticRegression(
            C=1.0,
            penalty='l2',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        'SVM (Balanced)': SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }
    
    # Train and evaluate models
    print(f"\n🤖 Training and evaluating improved models...")
    results = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=cv, scoring='f1')
        
        # Train model
        model.fit(X_train_scaled, y_train_balanced)
        
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
        
        print(f"     CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"     Test F1: {f1:.3f}, AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")
    
    # Print results summary
    print(f"\n📊 IMPROVED MODEL PERFORMANCE SUMMARY")
    print(f"{'Model':<30} {'CV F1':<12} {'Test F1':<10} {'Test AUC':<10} {'Test Acc':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        cv_f1 = result['cv_scores'].mean()
        cv_std = result['cv_scores'].std()
        test_f1 = result['f1']
        test_auc = result['auc']
        test_acc = result['accuracy']
        print(f"{name:<30} {cv_f1:.3f}±{cv_std:.3f}   {test_f1:<10.3f} {test_auc:<10.3f} {test_acc:<10.3f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST IMPROVED MODEL: {best_model_name}")
    print(f"   CV F1-Score: {best_result['cv_scores'].mean():.3f}")
    print(f"   Test F1-Score: {best_result['f1']:.3f}")
    print(f"   Test AUC: {best_result['auc']:.3f}")
    print(f"   Test Accuracy: {best_result['accuracy']:.3f}")
    
    # Detailed classification report
    print(f"\n📊 CLASSIFICATION REPORT (Best Model)")
    print(classification_report(y_test, best_result['y_pred'], target_names=['Poor Outcome', 'Good Outcome']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, best_result['y_pred'])
    print(f"\n📋 CONFUSION MATRIX (Best Model)")
    print(f"   True Negatives (Poor→Poor): {cm[0,0]}")
    print(f"   False Positives (Poor→Good): {cm[0,1]}")
    print(f"   False Negatives (Good→Poor): {cm[1,0]}")
    print(f"   True Positives (Good→Good): {cm[1,1]}")
    
    # Feature importance for tree-based models
    if hasattr(best_result['model'], 'feature_importances_'):
        print(f"\n🔍 TOP 15 FEATURE IMPORTANCE (Best Model)")
        feature_importance = pd.DataFrame({
            'feature': combined_features,
            'importance': best_result['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            print(f"   {i+1:2d}. {row['feature']:<40} {row['importance']:.4f}")
    
    # Save results
    print(f"\n💾 Saving improved results...")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'PatientID': df_analysis.iloc[-len(y_test):]['PatientID'].values,
        'Actual_90day_mRS': df_analysis.iloc[-len(y_test):][target_col].values,
        'Actual_Binary': y_test.values,
        'Predicted_Binary': best_result['y_pred'],
        'Predicted_Probability': best_result['y_pred_proba']
    })
    predictions_df.to_csv('improved_mrs_predictions.csv', index=False)
    print(f"   Saved predictions to: improved_mrs_predictions.csv")
    
    # Save model comparison
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'CV_F1_Mean': [results[name]['cv_scores'].mean() for name in results.keys()],
        'CV_F1_Std': [results[name]['cv_scores'].std() for name in results.keys()],
        'Test_F1': [results[name]['f1'] for name in results.keys()],
        'Test_AUC': [results[name]['auc'] for name in results.keys()],
        'Test_Accuracy': [results[name]['accuracy'] for name in results.keys()]
    }).sort_values('Test_F1', ascending=False)
    
    comparison_df.to_csv('improved_model_comparison.csv', index=False)
    print(f"   Saved model comparison to: improved_model_comparison.csv")
    
    # Save feature importance if available
    if hasattr(best_result['model'], 'feature_importances_'):
        feature_importance.to_csv('improved_feature_importance.csv', index=False)
        print(f"   Saved feature importance to: improved_feature_importance.csv")
    
    print(f"\n✅ IMPROVED MRS PREDICTION COMPLETE!")
    print(f"   Best model: {best_model_name}")
    print(f"   Best Test F1-Score: {best_result['f1']:.3f}")
    print(f"   Best Test AUC: {best_result['auc']:.3f}")
    print(f"   Best Test Accuracy: {best_result['accuracy']:.3f}")
    
    # Improvement summary
    print(f"\n📈 IMPROVEMENTS MADE:")
    print(f"   • Removed non-numeric mRS values")
    print(f"   • Applied SMOTE for class balancing")
    print(f"   • Used robust scaling")
    print(f"   • Advanced feature selection (combined methods)")
    print(f"   • Class weights for imbalanced data")
    print(f"   • Multiple algorithms with optimized parameters")
    
    return best_result['model'], comparison_df, feature_importance if hasattr(best_result['model'], 'feature_importances_') else None

if __name__ == "__main__":
    improved_mrs_prediction() 