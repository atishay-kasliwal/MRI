#!/usr/bin/env python3
"""
Comprehensive Conclusions Summary
Generate all key numbers and statistics for research conclusions
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def generate_conclusions_summary():
    print("=== COMPREHENSIVE CONCLUSIONS SUMMARY ===\n")
    
    # Load all relevant data
    try:
        df = pd.read_csv("merged_radiomics_clinical_data.csv")
        print(f"📊 Main dataset loaded: {len(df)} patients")
    except FileNotFoundError:
        print("⚠️  Main dataset not found")
        return
    
    # Load model performance results
    try:
        model_results = pd.read_csv("model_performance_results.csv")
        print(f"📊 Model results loaded: {len(model_results)} configurations")
    except FileNotFoundError:
        print("⚠️  Model results not found")
        model_results = None
    
    # Load top features data
    try:
        top_features = pd.read_csv("top_20_features_golden.csv")
        print(f"📊 Top features loaded: {len(top_features)} features")
    except FileNotFoundError:
        print("⚠️  Top features not found")
        top_features = None
    
    print("\n" + "="*80)
    print("📈 DATASET OVERVIEW")
    print("="*80)
    
    # Dataset statistics
    total_patients = len(df)
    print(f"Total Patients: {total_patients}")
    
    # mRS distribution
    target_col = '90 days mRS'
    df_clean = df.copy()
    non_numeric_mask = df_clean[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean = df_clean[~non_numeric_mask].copy()
    target_data = pd.to_numeric(df_clean[target_col], errors='coerce').dropna()
    
    print(f"Valid mRS Data: {len(target_data)} patients ({len(target_data)/total_patients*100:.1f}%)")
    
    # mRS breakdown
    mrs_counts = target_data.value_counts().sort_index()
    print(f"\n📊 mRS Distribution:")
    for mrs_score, count in mrs_counts.items():
        percentage = (count / len(target_data)) * 100
        print(f"  mRS {int(mrs_score)}: {count} patients ({percentage:.1f}%)")
    
    # Binary outcome
    binary_target = (target_data <= 2).astype(int)
    good_outcome = sum(binary_target == 1)
    poor_outcome = sum(binary_target == 0)
    print(f"\n📊 Binary Outcome (Good vs Poor):")
    print(f"  Good Outcome (mRS 0-2): {good_outcome} patients ({good_outcome/len(binary_target)*100:.1f}%)")
    print(f"  Poor Outcome (mRS 3-5): {poor_outcome} patients ({poor_outcome/len(binary_target)*100:.1f}%)")
    
    # Radiomics features
    feature_columns = [col for col in df_clean.columns if any(prefix in col for prefix in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    print(f"\n📊 Radiomics Features:")
    print(f"  Total Features: {len(feature_columns)}")
    
    # Features by scan type
    scan_types = {'T1': 0, 'T2': 0, 'FLAIR': 0, 'DWI': 0, 'ADC': 0}
    for feature in feature_columns:
        for scan_type in scan_types:
            if feature.startswith(f'{scan_type}_'):
                scan_types[scan_type] += 1
                break
    
    for scan_type, count in scan_types.items():
        print(f"  {scan_type}: {count} features ({count/len(feature_columns)*100:.1f}%)")
    
    print("\n" + "="*80)
    print("🏆 TOP FEATURES ANALYSIS")
    print("="*80)
    
    if top_features is not None:
        print(f"📊 Top 20 Most Important Features:")
        print(f"  Highest Correlation: {top_features.iloc[0]['correlation']:.3f}")
        print(f"  Lowest Correlation: {top_features.iloc[-1]['correlation']:.3f}")
        print(f"  Average Correlation: {top_features['correlation'].mean():.3f}")
        print(f"  Median Correlation: {top_features['correlation'].median():.3f}")
        
        # Top feature by scan type
        scan_type_counts = {}
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            for scan_type in ['T1', 'T2', 'FLAIR', 'DWI', 'ADC']:
                if feature_name.startswith(f'{scan_type}_'):
                    scan_type_counts[scan_type] = scan_type_counts.get(scan_type, 0) + 1
                    break
        
        print(f"\n📊 Top 20 Features by Scan Type:")
        for scan_type, count in sorted(scan_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {scan_type}: {count} features ({count/20*100:.0f}%)")
        
        # Top 5 features
        print(f"\n📊 Top 5 Most Important Features:")
        for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
            feature_name = row['feature']
            correlation = row['correlation']
            print(f"  {i}. {feature_name}: {correlation:.3f}")
    
    print("\n" + "="*80)
    print("🤖 MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    if model_results is not None:
        # Best performance by metric
        metrics = ['F1_Score', 'AUC_Score', 'Accuracy', 'Precision']
        
        for metric in metrics:
            if metric in model_results.columns:
                best_idx = model_results[metric].idxmax()
                best_result = model_results.loc[best_idx]
                print(f"\n📊 Best {metric.replace('_', ' ')}:")
                print(f"  Model: {best_result['Model']}")
                print(f"  Feature Set: {best_result['Feature_Set']}")
                print(f"  Score: {best_result[metric]:.3f}")
        
        # Model comparison
        print(f"\n📊 Model Performance Comparison (F1 Score):")
        models = ['SVM', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'CatBoost']
        
        for model in models:
            model_data = model_results[model_results['Model'] == model]
            if not model_data.empty:
                best_f1 = model_data['F1_Score'].max()
                best_feature_set = model_data.loc[model_data['F1_Score'].idxmax(), 'Feature_Set']
                print(f"  {model}: {best_f1:.3f} ({best_feature_set})")
        
        # Feature set comparison
        print(f"\n📊 Feature Set Performance Comparison (F1 Score):")
        feature_sets = ['All Features', 'Top 100', 'Top 70', 'Top 50']
        
        for feature_set in feature_sets:
            feature_data = model_results[model_results['Feature_Set'] == feature_set]
            if not feature_data.empty:
                avg_f1 = feature_data['F1_Score'].mean()
                max_f1 = feature_data['F1_Score'].max()
                print(f"  {feature_set}: Avg={avg_f1:.3f}, Max={max_f1:.3f}")
    
    print("\n" + "="*80)
    print("📋 TRAIN/TEST SPLIT SUMMARY")
    print("="*80)
    
    # Calculate train/test split
    total_valid = len(target_data)
    train_size = int(0.8 * total_valid)
    test_size = total_valid - train_size
    
    print(f"📊 Overall Split:")
    print(f"  Training Set: {train_size} patients ({train_size/total_valid*100:.1f}%)")
    print(f"  Testing Set: {test_size} patients ({test_size/total_valid*100:.1f}%)")
    
    # Split by mRS category
    print(f"\n📊 Split by mRS Category:")
    for mrs_score in sorted(mrs_counts.index):
        total_mrs = mrs_counts[mrs_score]
        train_mrs = int(0.8 * total_mrs)
        test_mrs = total_mrs - train_mrs
        print(f"  mRS {int(mrs_score)}: {total_mrs} total → {train_mrs} train, {test_mrs} test")
    
    print("\n" + "="*80)
    print("🎯 KEY FINDINGS & CONCLUSIONS")
    print("="*80)
    
    print(f"📊 Dataset Characteristics:")
    print(f"  • Total patients analyzed: {total_patients}")
    print(f"  • Valid mRS outcomes: {len(target_data)} ({len(target_data)/total_patients*100:.1f}%)")
    print(f"  • Radiomics features extracted: {len(feature_columns)}")
    print(f"  • Scan types included: {len([k for k, v in scan_types.items() if v > 0])}")
    
    print(f"\n📊 Clinical Outcomes:")
    print(f"  • Good outcome rate (mRS 0-2): {good_outcome/len(binary_target)*100:.1f}%")
    print(f"  • Poor outcome rate (mRS 3-5): {poor_outcome/len(binary_target)*100:.1f}%")
    print(f"  • Most common mRS score: {mrs_counts.idxmax()} ({mrs_counts.max()} patients)")
    
    if top_features is not None:
        print(f"\n📊 Feature Importance:")
        print(f"  • Highest correlation: {top_features.iloc[0]['correlation']:.3f}")
        print(f"  • Feature correlation range: {top_features['correlation'].min():.3f} - {top_features['correlation'].max():.3f}")
        print(f"  • Dominant scan type in top features: {max(scan_type_counts, key=scan_type_counts.get)}")
    
    if model_results is not None:
        print(f"\n📊 Model Performance:")
        best_f1 = model_results['F1_Score'].max()
        best_model = model_results.loc[model_results['F1_Score'].idxmax()]
        print(f"  • Best F1 Score: {best_f1:.3f}")
        print(f"  • Best performing model: {best_model['Model']}")
        print(f"  • Optimal feature set: {best_model['Feature_Set']}")
        print(f"  • Average F1 Score across all models: {model_results['F1_Score'].mean():.3f}")
    
    print(f"\n📊 Data Quality:")
    print(f"  • Missing data rate: {(total_patients - len(target_data))/total_patients*100:.1f}%")
    print(f"  • Feature-to-sample ratio: {len(feature_columns)}:{len(target_data)} = {len(feature_columns)/len(target_data):.1f}:1")
    
    # Save comprehensive summary
    summary_data = {
        'Metric': [
            'Total Patients', 'Valid mRS Data', 'Good Outcome (mRS 0-2)', 'Poor Outcome (mRS 3-5)',
            'Total Radiomics Features', 'T1 Features', 'T2 Features', 'FLAIR Features', 'DWI Features', 'ADC Features',
            'Best F1 Score', 'Best Model', 'Optimal Feature Set', 'Average F1 Score',
            'Training Set Size', 'Testing Set Size', 'Feature-to-Sample Ratio'
        ],
        'Value': [
            total_patients, len(target_data), good_outcome, poor_outcome,
            len(feature_columns), scan_types['T1'], scan_types['T2'], scan_types['FLAIR'], scan_types['DWI'], scan_types['ADC'],
            best_f1 if model_results is not None else 'N/A',
            best_model['Model'] if model_results is not None else 'N/A',
            best_model['Feature_Set'] if model_results is not None else 'N/A',
            model_results['F1_Score'].mean() if model_results is not None else 'N/A',
            train_size, test_size, f"{len(feature_columns)/len(target_data):.1f}:1"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('conclusions_summary.csv', index=False)
    print(f"\n💾 Saved comprehensive summary to: conclusions_summary.csv")
    
    return summary_df

if __name__ == "__main__":
    generate_conclusions_summary()