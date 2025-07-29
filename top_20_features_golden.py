#!/usr/bin/env python3
"""
Top 20 Features - Golden Theme
Create a horizontal bar graph showing top 20 features with golden theme
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_top_20_features_golden():
    print("=== TOP 20 FEATURES - GOLDEN THEME ===\n")
    
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
    
    # Get radiomics features
    feature_columns = [col for col in df_clean.columns if any(prefix in col for prefix in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    print(f"📋 Radiomics features: {len(feature_columns)}")
    
    # Align features with target
    common_indices = df_clean.index.intersection(target_data.index)
    X = df_clean.loc[common_indices, feature_columns]
    y = binary_target.loc[common_indices]
    
    # Handle missing values
    X = X.fillna(0)
    
    # Calculate feature importance using correlation
    feature_scores = []
    
    for feature in feature_columns:
        try:
            feature_data = pd.to_numeric(X[feature], errors='coerce').dropna()
            common_indices_feature = feature_data.index.intersection(y.index)
            
            if len(common_indices_feature) < 10:
                continue
            
            feature_aligned = feature_data.loc[common_indices_feature]
            target_aligned = y.loc[common_indices_feature]
            
            # Calculate correlation (absolute value for importance)
            correlation = abs(feature_aligned.corr(target_aligned))
            
            if not pd.isna(correlation):
                feature_scores.append({
                    'feature': feature,
                    'correlation': correlation,
                    'sample_size': len(common_indices_feature)
                })
                
        except Exception as e:
            continue
    
    # Sort by correlation and get top 20
    feature_scores.sort(key=lambda x: x['correlation'], reverse=True)
    top_20_features = feature_scores[:20]
    
    print(f"📊 Top correlation: {top_20_features[0]['correlation']:.3f}")
    print(f"📊 Bottom correlation: {top_20_features[-1]['correlation']:.3f}")
    
    # Create the horizontal bar chart
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Prepare data for plotting
    features = [f['feature'] for f in top_20_features]
    correlations = [f['correlation'] for f in top_20_features]
    
    # Create golden color gradient
    golden_colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(correlations)))
    
    # Create vertical bar chart
    bars = ax.bar(range(len(features)), correlations, color=golden_colors, 
                  alpha=0.9, edgecolor='#8B7355', linewidth=1)
    
    # Add correlation values above bars
    for i, (bar, corr) in enumerate(zip(bars, correlations)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
               f'{corr:.3f}', ha='center', va='bottom', fontweight='bold', 
               fontsize=10, color='#8B4513')
    
    # Add feature names inside bars, vertically centered
    for i, (bar, feature) in enumerate(zip(bars, features)):
        # Remove scan type prefix and keep important parts
        if feature.startswith('T1_'):
            shortened = feature.replace('T1_', 'T1: ')
        elif feature.startswith('T2_'):
            shortened = feature.replace('T2_', 'T2: ')
        elif feature.startswith('FLAIR_'):
            shortened = feature.replace('FLAIR_', 'FLAIR: ')
        elif feature.startswith('DWI_'):
            shortened = feature.replace('DWI_', 'DWI: ')
        elif feature.startswith('ADC_'):
            shortened = feature.replace('ADC_', 'ADC: ')
        else:
            shortened = feature
        if len(shortened) > 30:
            shortened = shortened[:27] + '...'
        # Choose text color based on bar color for contrast
        bar_color = golden_colors[i]
        text_color = 'white' if np.mean(bar_color[:3]) < 0.7 else '#8B4513'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                shortened, ha='center', va='center', rotation=90, 
                fontsize=10, fontweight='bold', color=text_color, clip_on=True)
    
    # Remove x-axis labels for a clean look
    ax.set_xticks([])
    
    # Customize y-axis
    ax.set_ylabel('|Correlation with mRS Outcome|', fontsize=14, fontweight='bold', color='#8B4513')
    ax.set_ylim(0, max(correlations) * 1.1)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#D4AF37')
    ax.set_axisbelow(True)
    
    # Customize title
    ax.set_title('Top 20 Most Important Radiomics Features\nRanked by Correlation with 90-day mRS Outcome', 
                fontsize=16, fontweight='bold', color='#8B4513', pad=20)
    
    # Remove summary text box - keeping it clean without text overlay
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#D4AF37')
        spine.set_linewidth(1.5)
    
    # Customize tick colors
    ax.tick_params(axis='both', colors='#8B4513')
    
    plt.tight_layout()
    plt.savefig('top_20_features_golden.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"💾 Saved golden-themed top 20 features chart to: top_20_features_golden.png")
    
    # Print detailed breakdown
    print(f"\n📊 TOP 20 FEATURES BREAKDOWN:")
    print("=" * 80)
    
    for i, feature_info in enumerate(top_20_features, 1):
        feature_name = feature_info['feature']
        correlation = feature_info['correlation']
        sample_size = feature_info['sample_size']
        
        # Determine scan type
        if feature_name.startswith('T1_'):
            scan_type = 'T1'
        elif feature_name.startswith('T2_'):
            scan_type = 'T2'
        elif feature_name.startswith('FLAIR_'):
            scan_type = 'FLAIR'
        elif feature_name.startswith('DWI_'):
            scan_type = 'DWI'
        elif feature_name.startswith('ADC_'):
            scan_type = 'ADC'
        else:
            scan_type = 'Unknown'
        
        print(f"{i:2d}. {scan_type:6s} - {feature_name}: {correlation:.3f} (n={sample_size})")
    
    # Create summary by scan type
    print(f"\n📊 FEATURES BY SCAN TYPE:")
    print("=" * 50)
    
    scan_type_counts = {}
    for feature_info in top_20_features:
        feature_name = feature_info['feature']
        if feature_name.startswith('T1_'):
            scan_type = 'T1'
        elif feature_name.startswith('T2_'):
            scan_type = 'T2'
        elif feature_name.startswith('FLAIR_'):
            scan_type = 'FLAIR'
        elif feature_name.startswith('DWI_'):
            scan_type = 'DWI'
        elif feature_name.startswith('ADC_'):
            scan_type = 'ADC'
        else:
            scan_type = 'Unknown'
        
        scan_type_counts[scan_type] = scan_type_counts.get(scan_type, 0) + 1
    
    for scan_type, count in sorted(scan_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{scan_type}: {count} features")
    
    # Save top 20 features data
    top_20_df = pd.DataFrame(top_20_features)
    top_20_df.to_csv('top_20_features_golden.csv', index=False)
    print(f"\n💾 Saved top 20 features data to: top_20_features_golden.csv")
    
    return top_20_df

if __name__ == "__main__":
    create_top_20_features_golden() 