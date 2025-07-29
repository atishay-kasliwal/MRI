#!/usr/bin/env python3
"""
Top Features by Scan Type Analysis
Create bar graphs showing top 5 features from each scan type with relevance scores
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_top_features_by_scan_type():
    print("=== TOP FEATURES BY SCAN TYPE ANALYSIS ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Load feature importance data if available
    try:
        feature_importance_df = pd.read_csv("mrs_feature_importance.csv")
        print(f"📊 Feature importance data loaded: {len(feature_importance_df)} features")
        has_importance_data = True
    except FileNotFoundError:
        print("⚠️  Feature importance file not found. Will use correlation analysis.")
        has_importance_data = False
    
    # Define scan types and their feature prefixes
    scan_types = {
        'T1': ['T1_'],
        'T2': ['T2_'],
        'FLAIR': ['FLAIR_'],
        'DWI': ['DWI_'],
        'ADC': ['ADC_']
    }
    
    # Target variable for correlation
    target_col = '90 days mRS'
    
    # Clean target variable
    df_clean = df.copy()
    non_numeric_mask = df_clean[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean = df_clean[~non_numeric_mask].copy()
    target_data = pd.to_numeric(df_clean[target_col], errors='coerce').dropna()
    
    print(f"📋 Valid target data: {len(target_data)} patients")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Top 5 Features by Scan Type - Relevance Analysis', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Analyze each scan type
    for i, (scan_type, prefixes) in enumerate(scan_types.items()):
        print(f"\n📋 Analyzing {scan_type} features...")
        
        # Find features for this scan type
        scan_features = []
        for prefix in prefixes:
            scan_features.extend([col for col in df_clean.columns if col.startswith(prefix)])
        
        print(f"   Found {len(scan_features)} {scan_type} features")
        
        if len(scan_features) == 0:
            print(f"   No features found for {scan_type}")
            continue
        
        # Calculate feature relevance
        feature_scores = []
        
        for feature in scan_features:
            try:
                # Get feature data
                feature_data = pd.to_numeric(df_clean[feature], errors='coerce').dropna()
                
                # Align with target data
                common_indices = feature_data.index.intersection(target_data.index)
                if len(common_indices) < 10:  # Need minimum data points
                    continue
                
                feature_aligned = feature_data.loc[common_indices]
                target_aligned = target_data.loc[common_indices]
                
                # Calculate correlation (absolute value for relevance)
                correlation = abs(feature_aligned.corr(target_aligned))
                
                if not pd.isna(correlation):
                    feature_scores.append({
                        'feature': feature,
                        'correlation': correlation,
                        'sample_size': len(common_indices)
                    })
                    
            except Exception as e:
                continue
        
        if len(feature_scores) == 0:
            print(f"   No valid correlations for {scan_type}")
            continue
        
        # Sort by correlation and get top 5
        feature_scores.sort(key=lambda x: x['correlation'], reverse=True)
        top_5_features = feature_scores[:5]
        
        print(f"   Top 5 features correlation range: {top_5_features[0]['correlation']:.3f} - {top_5_features[-1]['correlation']:.3f}")
        
        # Create bar plot
        ax = axes_flat[i]
        
        features = [f['feature'].replace(f'{scan_type}_', '') for f in top_5_features]
        correlations = [f['correlation'] for f in top_5_features]
        
        # Create horizontal bar chart for better readability
        bars = ax.barh(range(len(features)), correlations, color='skyblue', alpha=0.8)
        
        # Add value labels on bars
        for j, (bar, corr) in enumerate(zip(bars, correlations)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{corr:.3f}', ha='left', va='center', fontweight='bold', fontsize=10)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=10)
        ax.set_xlabel('|Correlation with mRS|', fontsize=11)
        ax.set_title(f'{scan_type} - Top 5 Features\n(n={len(feature_scores)} features)', 
                    fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, max(correlations) * 1.15)  # Add some padding for labels
        
        # Add sample size info
        avg_sample = np.mean([f['sample_size'] for f in top_5_features])
        ax.text(0.02, 0.98, f'Avg samples: {avg_sample:.0f}', 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(scan_types), len(axes_flat)):
        fig.delaxes(axes_flat[i])
    
    plt.tight_layout()
    plt.savefig('top_features_by_scan_type.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved top features analysis to: top_features_by_scan_type.png")
    
    # Create summary table
    print(f"\n📋 SUMMARY OF TOP FEATURES BY SCAN TYPE:")
    print("-" * 80)
    
    summary_data = []
    
    for scan_type, prefixes in scan_types.items():
        print(f"\n{scan_type} SCAN TYPE:")
        
        # Find features for this scan type
        scan_features = []
        for prefix in prefixes:
            scan_features.extend([col for col in df_clean.columns if col.startswith(prefix)])
        
        if len(scan_features) == 0:
            print("   No features found")
            continue
        
        # Calculate feature relevance
        feature_scores = []
        
        for feature in scan_features:
            try:
                feature_data = pd.to_numeric(df_clean[feature], errors='coerce').dropna()
                common_indices = feature_data.index.intersection(target_data.index)
                
                if len(common_indices) < 10:
                    continue
                
                feature_aligned = feature_data.loc[common_indices]
                target_aligned = target_data.loc[common_indices]
                correlation = abs(feature_aligned.corr(target_aligned))
                
                if not pd.isna(correlation):
                    feature_scores.append({
                        'feature': feature,
                        'correlation': correlation,
                        'sample_size': len(common_indices)
                    })
                    
            except Exception as e:
                continue
        
        if len(feature_scores) == 0:
            print("   No valid correlations")
            continue
        
        # Sort and get top 5
        feature_scores.sort(key=lambda x: x['correlation'], reverse=True)
        top_5_features = feature_scores[:5]
        
        print(f"   Total features analyzed: {len(feature_scores)}")
        print(f"   Top correlation: {top_5_features[0]['correlation']:.3f}")
        
        for j, feature_info in enumerate(top_5_features, 1):
            feature_name = feature_info['feature'].replace(f'{scan_type}_', '')
            correlation = feature_info['correlation']
            sample_size = feature_info['sample_size']
            
            print(f"   {j}. {feature_name}: {correlation:.3f} (n={sample_size})")
            
            summary_data.append({
                'Scan_Type': scan_type,
                'Rank': j,
                'Feature_Name': feature_name,
                'Full_Feature_Name': feature_info['feature'],
                'Correlation': round(correlation, 3),
                'Sample_Size': sample_size
            })
    
    # Save summary to CSV
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('top_features_by_scan_type_summary.csv', index=False)
        print(f"\n💾 Saved summary to: top_features_by_scan_type_summary.csv")
        
        return summary_df
    else:
        print(f"\n⚠️  No valid features found for any scan type")
        return None

if __name__ == "__main__":
    analyze_top_features_by_scan_type() 