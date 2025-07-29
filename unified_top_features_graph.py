#!/usr/bin/env python3
"""
Unified Top Features Graph
Create a single, aesthetically pleasing graph showing all scan types together
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_unified_top_features_graph():
    print("=== UNIFIED TOP FEATURES GRAPH ===\n")
    
    # Set style for better aesthetics
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Define scan types and their colors
    scan_types = {
        'FLAIR': {'prefix': 'FLAIR_', 'color': '#FF6B6B', 'rank': 1},
        'T1': {'prefix': 'T1_', 'color': '#4ECDC4', 'rank': 2},
        'ADC': {'prefix': 'ADC_', 'color': '#45B7D1', 'rank': 3},
        'T2': {'prefix': 'T2_', 'color': '#96CEB4', 'rank': 4},
        'DWI': {'prefix': 'DWI_', 'color': '#FFEAA7', 'rank': 5}
    }
    
    # Target variable for correlation
    target_col = '90 days mRS'
    
    # Clean target variable
    df_clean = df.copy()
    non_numeric_mask = df_clean[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean = df_clean[~non_numeric_mask].copy()
    target_data = pd.to_numeric(df_clean[target_col], errors='coerce').dropna()
    
    print(f"📋 Valid target data: {len(target_data)} patients")
    
    # Collect all top features data
    all_top_features = []
    
    for scan_type, config in scan_types.items():
        print(f"\n📋 Analyzing {scan_type} features...")
        
        # Find features for this scan type
        scan_features = [col for col in df_clean.columns if col.startswith(config['prefix'])]
        print(f"   Found {len(scan_features)} {scan_type} features")
        
        if len(scan_features) == 0:
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
            continue
        
        # Sort and get top 5
        feature_scores.sort(key=lambda x: x['correlation'], reverse=True)
        top_5_features = feature_scores[:5]
        
        print(f"   Top correlation: {top_5_features[0]['correlation']:.3f}")
        
        for j, feature_info in enumerate(top_5_features, 1):
            feature_name = feature_info['feature'].replace(config['prefix'], '')
            all_top_features.append({
                'Scan_Type': scan_type,
                'Rank': j,
                'Feature_Name': feature_name,
                'Full_Feature_Name': feature_info['feature'],
                'Correlation': feature_info['correlation'],
                'Sample_Size': feature_info['sample_size'],
                'Color': config['color'],
                'Scan_Rank': config['rank']
            })
    
    # Create the unified graph
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Prepare data for plotting
    df_plot = pd.DataFrame(all_top_features)
    
    # Create a combined label for y-axis
    df_plot['Y_Label'] = df_plot['Scan_Type'] + ' - ' + df_plot['Feature_Name']
    
    # Sort by correlation for better visualization
    df_plot = df_plot.sort_values('Correlation', ascending=True)
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(df_plot)), df_plot['Correlation'], 
                   color=df_plot['Color'], alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, df_plot['Correlation'])):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
               f'{corr:.3f}', ha='left', va='center', fontweight='bold', 
               fontsize=10, color='#2C3E50')
    
    # Customize y-axis labels
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['Y_Label'], fontsize=11, fontweight='bold')
    
    # Customize x-axis
    ax.set_xlabel('|Correlation with 90-day mRS|', fontsize=14, fontweight='bold', color='#2C3E50')
    ax.set_xlim(0, df_plot['Correlation'].max() * 1.1)
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--', color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # Customize title
    ax.set_title('Top 5 Features by Scan Type - Unified View\nRanked by Correlation with 90-day mRS', 
                fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
    
    # Add scan type performance summary
    scan_performance = df_plot.groupby('Scan_Type')['Correlation'].max().sort_values(ascending=False)
    
    performance_text = "Scan Type Performance (Max Correlation):\n"
    for i, (scan_type, max_corr) in enumerate(scan_performance.items(), 1):
        performance_text += f"{i}. {scan_type}: {max_corr:.3f}\n"
    
    ax.text(0.02, 0.98, performance_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.9, edgecolor='#BDC3C7'))
    
    # Add total features info
    total_features = len(df_plot)
    ax.text(0.98, 0.02, f'Total Features: {total_features}\nPatients: {len(target_data)}', 
            transform=ax.transAxes, fontsize=10, horizontalalignment='right',
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', 
            facecolor='white', alpha=0.8, edgecolor='#BDC3C7'))
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#BDC3C7')
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    plt.savefig('unified_top_features_graph.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\n💾 Saved unified graph to: unified_top_features_graph.png")
    
    # Create enhanced summary
    print(f"\n📊 ENHANCED SUMMARY:")
    print("=" * 80)
    
    for scan_type in scan_performance.index:
        scan_data = df_plot[df_plot['Scan_Type'] == scan_type].sort_values('Correlation', ascending=False)
        max_corr = scan_data['Correlation'].iloc[0]
        
        print(f"\n🏆 {scan_type} SCAN TYPE (Max: {max_corr:.3f}):")
        print("-" * 50)
        
        for _, row in scan_data.iterrows():
            print(f"  {row['Rank']}. {row['Feature_Name']}: {row['Correlation']:.3f}")
    
    # Save enhanced summary
    df_plot.to_csv('unified_top_features_summary.csv', index=False)
    print(f"\n💾 Saved enhanced summary to: unified_top_features_summary.csv")
    
    # Create performance comparison
    performance_df = pd.DataFrame({
        'Scan_Type': scan_performance.index,
        'Max_Correlation': scan_performance.values,
        'Rank': range(1, len(scan_performance) + 1)
    })
    
    performance_df.to_csv('scan_type_performance_ranking.csv', index=False)
    print(f"💾 Saved performance ranking to: scan_type_performance_ranking.csv")
    
    return df_plot, performance_df

if __name__ == "__main__":
    create_unified_top_features_graph() 