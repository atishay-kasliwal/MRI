#!/usr/bin/env python3
"""
Radiomics Feature Prediction Models
Predicts radiomics features using multiple ML models with 80/20 train/test split
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

def predict_radiomics_features():
    print("=== RADIOMICS FEATURE PREDICTION MODELS ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients, {len(df.columns)} features")
    
    # Separate radiomics and clinical features
    radiomics_cols = [col for col in df.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    clinical_cols = [col for col in df.columns if col not in radiomics_cols and col not in ['PatientID', 'Year', 'AvailableModalities']]
    
    print(f"🔬 Radiomics features: {len(radiomics_cols)}")
    print(f"🏥 Clinical features: {len(clinical_cols)}")
    
    # Prepare features and targets
    X = df[clinical_cols].copy()
    y = df[radiomics_cols].copy()
    
    # Handle missing values
    print(f"\n📊 Handling missing values...")
    missing_before = X.isnull().sum().sum() + y.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    
    # Fill missing values
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    missing_after = X.isnull().sum().sum() + y.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes
    
    # Feature selection for clinical features
    print(f"\n🎯 Feature selection...")
    selector = SelectKBest(score_func=f_regression, k=min(50, len(X.columns)))
    X_selected = selector.fit_transform(X, y.iloc[:, 0])  # Use first radiomics feature for selection
    selected_clinical_features = X.columns[selector.get_support()].tolist()
    print(f"   Selected {len(selected_clinical_features)} clinical features")
    
    # Use selected clinical features
    X = X[selected_clinical_features]
    
    # Select a subset of radiomics features for prediction (top 10 most important)
    print(f"\n🎯 Selecting target radiomics features...")
    # Use the top 10 features from our previous analysis
    top_radiomics_features = [
        'T1_original_firstorder_Minimum',
        'ADC_original_firstorder_Minimum', 
        'T1_original_glcm_Idmn',
        'DWI_original_shape_Maximum2DDiameterSlice',
        'T2_original_ngtdm_Busyness',
        'T2_original_glszm_GrayLevelNonUniformity',
        'T1_original_glrlm_LongRunEmphasis',
        'T1_original_shape_Maximum2DDiameterColumn',
        'T2_original_shape_Maximum2DDiameterRow',
        'DWI_original_shape_VoxelVolume'
    ]
    
    # Filter to only features that exist in our dataset
    available_radiomics = [feat for feat in top_radiomics_features if feat in y.columns]
    y = y[available_radiomics]
    print(f"   Predicting {len(available_radiomics)} radiomics features")
    
    # Split data 80/20
    print(f"\n📈 Data splitting (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    print(f"   Training set: {len(X_train)} patients")
    print(f"   Test set: {len(X_test)} patients")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42),
        'Lasso Regression': Lasso(alpha=0.1, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGBoost-style (Gradient Boosting)': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
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
        
        # Calculate metrics for each radiomics feature
        feature_metrics = {}
        for i, feature in enumerate(available_radiomics):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            
            feature_metrics[feature] = {
                'MSE': mse,
                'R2': r2,
                'MAE': mae
            }
        
        # Overall metrics (average across features)
        overall_mse = np.mean([metrics['MSE'] for metrics in feature_metrics.values()])
        overall_r2 = np.mean([metrics['R2'] for metrics in feature_metrics.values()])
        overall_mae = np.mean([metrics['MAE'] for metrics in feature_metrics.values()])
        
        results[name] = {
            'model': model,
            'feature_metrics': feature_metrics,
            'overall_mse': overall_mse,
            'overall_r2': overall_r2,
            'overall_mae': overall_mae,
            'predictions': y_pred
        }
        
        print(f"     Overall R²: {overall_r2:.3f}, MSE: {overall_mse:.3f}")
    
    # Print results summary
    print(f"\n📊 MODEL PERFORMANCE SUMMARY")
    print(f"{'Model':<25} {'R² Score':<10} {'MSE':<12} {'MAE':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        r2 = result['overall_r2']
        mse = result['overall_mse']
        mae = result['overall_mae']
        print(f"{name:<25} {r2:<10.3f} {mse:<12.3f} {mae:<10.3f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['overall_r2'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Overall R²: {best_result['overall_r2']:.3f}")
    print(f"   Overall MSE: {best_result['overall_mse']:.3f}")
    print(f"   Overall MAE: {best_result['overall_mae']:.3f}")
    
    # Feature-wise performance for best model
    print(f"\n🔍 FEATURE-WISE PERFORMANCE (Best Model)")
    print(f"{'Feature':<40} {'R²':<8} {'MSE':<10}")
    print("-" * 60)
    
    for feature, metrics in best_result['feature_metrics'].items():
        feature_short = feature.replace('_original_', ' - ')[:35]
        print(f"{feature_short:<40} {metrics['R2']:<8.3f} {metrics['MSE']:<10.3f}")
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save predictions
    predictions_df = pd.DataFrame(
        best_result['predictions'], 
        columns=available_radiomics,
        index=y_test.index
    )
    predictions_df.to_csv('radiomics_predictions.csv')
    print(f"   Saved predictions to: radiomics_predictions.csv")
    
    # Save model comparison
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'R2_Score': [results[name]['overall_r2'] for name in results.keys()],
        'MSE': [results[name]['overall_mse'] for name in results.keys()],
        'MAE': [results[name]['overall_mae'] for name in results.keys()]
    }).sort_values('R2_Score', ascending=False)
    
    comparison_df.to_csv('radiomics_model_comparison.csv', index=False)
    print(f"   Saved model comparison to: radiomics_model_comparison.csv")
    
    # Save detailed feature metrics
    feature_metrics_df = pd.DataFrame()
    for model_name, result in results.items():
        for feature, metrics in result['feature_metrics'].items():
            feature_metrics_df = pd.concat([feature_metrics_df, pd.DataFrame({
                'Model': [model_name],
                'Feature': [feature],
                'R2': [metrics['R2']],
                'MSE': [metrics['MSE']],
                'MAE': [metrics['MAE']]
            })], ignore_index=True)
    
    feature_metrics_df.to_csv('radiomics_feature_metrics.csv', index=False)
    print(f"   Saved feature metrics to: radiomics_feature_metrics.csv")
    
    print(f"\n✅ RADIOMICS PREDICTION ANALYSIS COMPLETE!")
    print(f"   Best model: {best_model_name}")
    print(f"   Best R²: {best_result['overall_r2']:.3f}")
    print(f"   Features predicted: {len(available_radiomics)}")
    
    return results, comparison_df

if __name__ == "__main__":
    predict_radiomics_features() 