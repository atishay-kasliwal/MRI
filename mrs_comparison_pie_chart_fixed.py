#!/usr/bin/env python3
"""
mRS Comparison Pie Chart - Fixed Version
Create a single pie chart comparing Last mRS vs 90 days mRS for scores 1-5
Without good/poor labels and with better readable percentages
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_mrs_comparison_pie_fixed():
    print("=== MRS COMPARISON PIE CHART - FIXED VERSION ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Clean and prepare data for both mRS columns
    last_mrs_col = 'Last mRS'
    days_90_mrs_col = '90 days mRS'
    
    # Clean Last mRS data
    df_clean_last = df.copy()
    non_numeric_mask_last = df_clean_last[last_mrs_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean_last = df_clean_last[~non_numeric_mask_last].copy()
    last_mrs_data = pd.to_numeric(df_clean_last[last_mrs_col], errors='coerce').dropna()
    
    # Clean 90 days mRS data
    df_clean_90 = df.copy()
    non_numeric_mask_90 = df_clean_90[days_90_mrs_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean_90 = df_clean_90[~non_numeric_mask_90].copy()
    days_90_mrs_data = pd.to_numeric(df_clean_90[days_90_mrs_col], errors='coerce').dropna()
    
    print(f"📋 Last mRS valid patients: {len(last_mrs_data)}")
    print(f"📋 90 days mRS valid patients: {len(days_90_mrs_data)}")
    
    # Filter for mRS scores 1-5 only
    last_mrs_filtered = last_mrs_data[(last_mrs_data >= 1) & (last_mrs_data <= 5)]
    days_90_mrs_filtered = days_90_mrs_data[(days_90_mrs_data >= 1) & (days_90_mrs_data <= 5)]
    
    print(f"📋 Last mRS (1-5): {len(last_mrs_filtered)} patients")
    print(f"📋 90 days mRS (1-5): {len(days_90_mrs_filtered)} patients")
    
    # Count distributions
    last_mrs_counts = last_mrs_filtered.value_counts().sort_index()
    days_90_mrs_counts = days_90_mrs_filtered.value_counts().sort_index()
    
    print(f"\n📊 Last mRS distribution (1-5): {last_mrs_counts.to_dict()}")
    print(f"📊 90 days mRS distribution (1-5): {days_90_mrs_counts.to_dict()}")
    
    # Create the pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('mRS Distribution Comparison (Scores 1-5)', fontsize=16, fontweight='bold')
    
    # Define colors for mRS scores 1-5
    colors = ['#3CB371', '#90EE90', '#FFD700', '#FFA500', '#FF6347']
    
    # Last mRS pie chart
    if len(last_mrs_counts) > 0:
        wedges1, texts1, autotexts1 = ax1.pie(
            last_mrs_counts.values,
            labels=[f'mRS {score}' for score in last_mrs_counts.index],
            autopct='%1.1f%%',
            colors=colors[:len(last_mrs_counts)],
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        # Make all percentage text white and bold for better readability
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax1.set_title(f'Last mRS Distribution\n({len(last_mrs_filtered)} patients)', fontweight='bold', fontsize=14)
    
    # 90 days mRS pie chart
    if len(days_90_mrs_counts) > 0:
        wedges2, texts2, autotexts2 = ax2.pie(
            days_90_mrs_counts.values,
            labels=[f'mRS {score}' for score in days_90_mrs_counts.index],
            autopct='%1.1f%%',
            colors=colors[:len(days_90_mrs_counts)],
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        # Make all percentage text white and bold for better readability
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax2.set_title(f'90 Days mRS Distribution\n({len(days_90_mrs_filtered)} patients)', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('mrs_comparison_pie_chart_fixed.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved fixed comparison pie chart to: mrs_comparison_pie_chart_fixed.png")
    
    # Create detailed comparison table
    print(f"\n📋 DETAILED COMPARISON:")
    print(f"{'mRS Score':<10} {'Last mRS':<15} {'90 Days mRS':<15} {'Difference':<15}")
    print("-" * 55)
    
    all_scores = sorted(set(list(last_mrs_counts.index) + list(days_90_mrs_counts.index)))
    
    for score in all_scores:
        last_count = last_mrs_counts.get(score, 0)
        days_90_count = days_90_mrs_counts.get(score, 0)
        difference = days_90_count - last_count
        
        last_percent = (last_count / len(last_mrs_filtered) * 100) if len(last_mrs_filtered) > 0 else 0
        days_90_percent = (days_90_count / len(days_90_mrs_filtered) * 100) if len(days_90_mrs_filtered) > 0 else 0
        
        print(f"mRS {score:<6} {last_count:<4} ({last_percent:>5.1f}%) {days_90_count:<4} ({days_90_percent:>5.1f}%) {difference:+>4}")
    
    # Save comparison data to CSV
    comparison_data = []
    for score in all_scores:
        last_count = last_mrs_counts.get(score, 0)
        days_90_count = days_90_mrs_counts.get(score, 0)
        last_percent = (last_count / len(last_mrs_filtered) * 100) if len(last_mrs_filtered) > 0 else 0
        days_90_percent = (days_90_count / len(days_90_mrs_filtered) * 100) if len(days_90_mrs_filtered) > 0 else 0
        
        comparison_data.append({
            'mRS_Score': score,
            'Last_mRS_Count': last_count,
            'Last_mRS_Percent': round(last_percent, 1),
            'Days_90_mRS_Count': days_90_count,
            'Days_90_mRS_Percent': round(days_90_percent, 1),
            'Difference': days_90_count - last_count
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('mrs_comparison_summary_fixed.csv', index=False)
    print(f"\n💾 Saved comparison summary to: mrs_comparison_summary_fixed.csv")
    
    return comparison_df

if __name__ == "__main__":
    create_mrs_comparison_pie_fixed() 