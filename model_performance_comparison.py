#!/usr/bin/env python3
"""
Model Performance Comparison
Create horizontal bar graphs showing performance of top 5 models with different feature sets
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available. Install with: pip install lightgbm")

def create_model_performance_comparison():
    print("=== MODEL PERFORMANCE COMPARISON ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Prepare target variable
    target_col = '90 days mRS'
    
    # Clean target variable
    df_clean = df.copy()
    non_numeric_mask = df_clean[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean = df_clean[~non_numeric_mask].copy()
    target_data = pd.to_numeric(df_clean[target_col], errors='coerce').dropna()
    
    # Create binary target (0-2 vs 3-5)
    binary_target = (target_data <= 2).astype(int)
    
    print(f"📋 Valid target data: {len(binary_target)} patients")
    print(f"📋 Binary distribution: Good (0-2): {sum(binary_target == 1)}, Poor (3-5): {sum(binary_target == 0)}")
    
    # Prepare features
    feature_columns = [col for col in df_clean.columns if any(prefix in col for prefix in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    print(f"📋 Radiomics features: {len(feature_columns)}")
    
    # Align features with target
    common_indices = df_clean.index.intersection(target_data.index)
    X = df_clean.loc[common_indices, feature_columns]
    y = binary_target.loc[common_indices]
    
    # Handle missing values
    X = X.fillna(0)
    
    # Remove zero variance features
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold()
    X = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()], index=X.index)
    print(f"📋 Features after variance filtering: {X.shape[1]}")
    
    # Define feature sets
    feature_sets = {
        'All Features': X.shape[1],
        'Top 100': 100,
        'Top 70': 70,
        'Top 50': 50
    }
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'SVM': SVC(random_state=42, class_weight='balanced', probability=True)
    }
    
    # Add advanced models if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = cb.CatBoostClassifier(random_state=42, verbose=False)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
    
    # Store results
    all_results = []
    
    # Test each feature set
    for feature_set_name, n_features in feature_sets.items():
        print(f"\n🔬 Testing {feature_set_name} ({n_features} features)...")
        
        # Feature selection
        if n_features < X.shape[1]:
            # Use SelectKBest for feature selection
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        else:
            X_selected = X
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Test each model
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train_balanced, y_train_balanced)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                all_results.append({
                    'Feature_Set': feature_set_name,
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1_Score': f1,
                    'AUC': auc,
                    'N_Features': n_features
                })
                
                print(f"   {model_name}: F1={f1:.3f}, AUC={auc:.3f}, Acc={accuracy:.3f}")
                
            except Exception as e:
                print(f"   ⚠️  Error with {model_name}: {str(e)}")
                continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Model Performance Comparison by Feature Set', fontsize=16, fontweight='bold')
    
    # Flatten axes
    axes_flat = axes.flatten()
    
    # Define colors for models
    model_colors = {
        'Random Forest': '#FF6B6B',
        'Gradient Boosting': '#4ECDC4',
        'Extra Trees': '#45B7D1',
        'Logistic Regression': '#96CEB4',
        'SVM': '#FFEAA7',
        'XGBoost': '#DDA0DD',
        'CatBoost': '#98D8C8',
        'LightGBM': '#F7DC6F'
    }
    
    # Create plots for each metric
    metrics = ['F1_Score', 'AUC', 'Accuracy', 'Precision']
    metric_names = ['F1 Score', 'AUC', 'Accuracy', 'Precision']
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes_flat[i]
        
        # Get top 5 models for this metric across all feature sets
        top_models = results_df.nlargest(5, metric)
        
        # Create horizontal bar chart
        y_pos = range(len(top_models))
        bars = ax.barh(y_pos, top_models[metric], 
                      color=[model_colors.get(model, '#CCCCCC') for model in top_models['Model']],
                      alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, top_models[metric])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Customize labels
        labels = [f"{row['Model']}\n({row['Feature_Set']})" for _, row in top_models.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        
        # Customize axis
        ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlim(0, top_models[metric].max() * 1.15)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_title(f'Top 5 Models by {metric_name}', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved model performance comparison to: model_performance_comparison.png")
    
    # Create unified top performers graph
    fig2, ax2 = plt.subplots(figsize=(16, 10))
    
    # Get overall top 5 models by F1 score
    top_overall = results_df.nlargest(5, 'F1_Score')
    
    # Create horizontal bar chart
    y_pos = range(len(top_overall))
    bars = ax2.barh(y_pos, top_overall['F1_Score'], 
                   color=[model_colors.get(model, '#CCCCCC') for model in top_overall['Model']],
                   alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add value labels with multiple metrics
    for j, (bar, row) in enumerate(zip(bars, top_overall.iterrows())):
        metrics_text = f"F1: {row[1]['F1_Score']:.3f}\nAUC: {row[1]['AUC']:.3f}\nAcc: {row[1]['Accuracy']:.3f}"
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                metrics_text, ha='left', va='center', fontweight='bold', fontsize=10)
    
    # Customize labels
    labels = [f"{row['Model']}\n({row['Feature_Set']})" for _, row in top_overall.iterrows()]
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=12, fontweight='bold')
    
    # Customize axis
    ax2.set_xlabel('F1 Score', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, top_overall['F1_Score'].max() * 1.2)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.set_title('Top 5 Overall Model Performers\nRanked by F1 Score', fontweight='bold', fontsize=16)
    
    # Add summary statistics
    summary_text = f"Best Model: {top_overall.iloc[0]['Model']} ({top_overall.iloc[0]['Feature_Set']})\n"
    summary_text += f"Best F1 Score: {top_overall.iloc[0]['F1_Score']:.3f}\n"
    summary_text += f"Best AUC: {top_overall.iloc[0]['AUC']:.3f}"
    
    ax2.text(0.02, 0.98, summary_text, transform=ax2.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
             facecolor='white', alpha=0.9, edgecolor='#BDC3C7'))
    
    plt.tight_layout()
    plt.savefig('top_5_overall_models.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved top 5 overall models to: top_5_overall_models.png")
    
    # Print detailed results
    print(f"\n📊 DETAILED RESULTS:")
    print("=" * 80)
    
    for feature_set in feature_sets.keys():
        feature_results = results_df[results_df['Feature_Set'] == feature_set].sort_values('F1_Score', ascending=False)
        print(f"\n🏆 {feature_set.upper()}:")
        print("-" * 50)
        
        for _, row in feature_results.head(5).iterrows():
            print(f"  {row['Model']}: F1={row['F1_Score']:.3f}, AUC={row['AUC']:.3f}, Acc={row['Accuracy']:.3f}")
    
    # Save results
    results_df.to_csv('model_performance_results.csv', index=False)
    print(f"\n💾 Saved detailed results to: model_performance_results.csv")
    
    return results_df

if __name__ == "__main__":
    create_model_performance_comparison() 