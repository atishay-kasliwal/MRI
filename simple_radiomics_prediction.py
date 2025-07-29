#!/usr/bin/env python3
"""
Simple Radiomics Feature Prediction
Predicts radiomics features using basic ML models
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def simple_radiomics_prediction():
    print("=== SIMPLE RADIOMICS PREDICTION ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
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
    
    # Encode categorical variables first
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = pd.Categorical(X[col]).codes
    
    # Now fill missing values
    X = X.fillna(0)  # Fill with 0 for simplicity
    y = y.fillna(y.median())
    
    # Select top 5 radiomics features to predict
    top_radiomics_features = [
        'T1_original_firstorder_Minimum',
        'ADC_original_firstorder_Minimum', 
        'T1_original_glcm_Idmn',
        'DWI_original_shape_Maximum2DDiameterSlice',
        'T2_original_ngtdm_Busyness'
    ]
    
    # Filter to only features that exist
    available_radiomics = [feat for feat in top_radiomics_features if feat in y.columns]
    y = y[available_radiomics]
    print(f"   Predicting {len(available_radiomics)} radiomics features")
    
    # Split data 80/20
    print(f"\n📈 Data splitting (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
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
        'Linear Regression': LinearRegression()
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
            
            feature_metrics[feature] = {
                'MSE': mse,
                'R2': r2
            }
        
        # Overall metrics
        overall_mse = np.mean([metrics['MSE'] for metrics in feature_metrics.values()])
        overall_r2 = np.mean([metrics['R2'] for metrics in feature_metrics.values()])
        
        results[name] = {
            'overall_mse': overall_mse,
            'overall_r2': overall_r2,
            'feature_metrics': feature_metrics
        }
        
        print(f"     Overall R²: {overall_r2:.3f}, MSE: {overall_mse:.3f}")
    
    # Print results summary
    print(f"\n📊 MODEL PERFORMANCE SUMMARY")
    print(f"{'Model':<20} {'R² Score':<10} {'MSE':<12}")
    print("-" * 45)
    
    for name, result in results.items():
        r2 = result['overall_r2']
        mse = result['overall_mse']
        print(f"{name:<20} {r2:<10.3f} {mse:<12.3f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['overall_r2'])
    best_result = results[best_model_name]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Overall R²: {best_result['overall_r2']:.3f}")
    print(f"   Overall MSE: {best_result['overall_mse']:.3f}")
    
    # Feature-wise performance
    print(f"\n🔍 FEATURE-WISE PERFORMANCE (Best Model)")
    print(f"{'Feature':<35} {'R²':<8}")
    print("-" * 45)
    
    for feature, metrics in best_result['feature_metrics'].items():
        feature_short = feature.replace('_original_', ' - ')[:30]
        print(f"{feature_short:<35} {metrics['R2']:<8.3f}")
    
    print(f"\n✅ RADIOMICS PREDICTION COMPLETE!")
    print(f"   Best model: {best_model_name}")
    print(f"   Best R²: {best_result['overall_r2']:.3f}")
    
    return results

if __name__ == "__main__":
    simple_radiomics_prediction() 