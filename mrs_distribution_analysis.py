#!/usr/bin/env python3
"""
mRS Distribution Analysis
Create pie charts and visualizations for mRS score distribution
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_mrs_distribution():
    print("=== MRS DISTRIBUTION ANALYSIS ===\n")
    
    # Load data
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    print(f"📊 Dataset loaded: {len(df)} patients")
    
    # Check all mRS columns
    mrs_columns = [col for col in df.columns if 'mrs' in col.lower()]
    print(f"🎯 MRS columns found: {mrs_columns}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Modified Rankin Scale (mRS) Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Analyze each mRS column
    for i, col in enumerate(mrs_columns):
        if i >= 4:  # Limit to 4 subplots
            break
            
        print(f"\n📋 Analyzing {col}...")
        
        # Clean the data
        df_clean = df.copy()
        non_numeric_mask = df_clean[col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
        df_clean = df_clean[~non_numeric_mask].copy()
        
        # Get valid mRS data
        valid_data = pd.to_numeric(df_clean[col], errors='coerce').dropna()
        
        if len(valid_data) == 0:
            print(f"   No valid data for {col}")
            continue
            
        print(f"   Valid patients: {len(valid_data)}")
        print(f"   Value range: {valid_data.min()}-{valid_data.max()}")
        
        # Count mRS values
        mrs_counts = valid_data.value_counts().sort_index()
        print(f"   Distribution: {mrs_counts.to_dict()}")
        
        # Create pie chart
        ax = axes[i//2, i%2]
        
        # Define colors for each mRS score
        colors = ['#2E8B57', '#3CB371', '#90EE90', '#FFD700', '#FFA500', '#FF6347']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            mrs_counts.values, 
            labels=[f'mRS {score}' for score in mrs_counts.index],
            autopct='%1.1f%%',
            colors=colors[:len(mrs_counts)],
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(f'{col} Distribution\n({len(valid_data)} patients)', fontweight='bold')
        
        # Add summary statistics
        good_outcome = len(valid_data[valid_data <= 2])
        poor_outcome = len(valid_data[valid_data >= 3])
        good_percent = (good_outcome / len(valid_data)) * 100
        poor_percent = (poor_outcome / len(valid_data)) * 100
        
        summary_text = f'Good (0-2): {good_outcome} ({good_percent:.1f}%)\nPoor (3-5): {poor_outcome} ({poor_percent:.1f}%)'
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # If we have fewer than 4 mRS columns, remove empty subplots
    for i in range(len(mrs_columns), 4):
        fig.delaxes(axes[i//2, i%2])
    
    plt.tight_layout()
    plt.savefig('mrs_distribution_pie_charts.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved pie charts to: mrs_distribution_pie_charts.png")
    
    # Create detailed analysis for 90-day mRS
    print(f"\n🎯 DETAILED ANALYSIS: 90 days mRS")
    target_col = '90 days mRS'
    
    # Clean the target variable
    df_clean = df.copy()
    non_numeric_mask = df_clean[target_col].apply(lambda x: not pd.isna(x) and not str(x).replace('.', '').replace('-', '').isdigit())
    df_clean = df_clean[~non_numeric_mask].copy()
    
    # Get valid mRS data
    valid_data = pd.to_numeric(df_clean[target_col], errors='coerce').dropna()
    
    if len(valid_data) > 0:
        # Create detailed pie chart for 90-day mRS
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig2.suptitle('90-Day mRS Distribution - Detailed Analysis', fontsize=16, fontweight='bold')
        
        # Individual mRS scores pie chart
        mrs_counts = valid_data.value_counts().sort_index()
        colors = ['#2E8B57', '#3CB371', '#90EE90', '#FFD700', '#FFA500', '#FF6347']
        
        wedges1, texts1, autotexts1 = ax1.pie(
            mrs_counts.values,
            labels=[f'mRS {score}' for score in mrs_counts.index],
            autopct='%1.1f%%',
            colors=colors[:len(mrs_counts)],
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax1.set_title('Individual mRS Scores\n(90-day follow-up)', fontweight='bold')
        
        # Binary outcome pie chart
        good_outcome = len(valid_data[valid_data <= 2])
        poor_outcome = len(valid_data[valid_data >= 3])
        
        binary_data = [good_outcome, poor_outcome]
        binary_labels = ['Good Outcome\n(mRS 0-2)', 'Poor Outcome\n(mRS 3-5)']
        binary_colors = ['#2E8B57', '#FF6347']
        
        wedges2, texts2, autotexts2 = ax2.pie(
            binary_data,
            labels=binary_labels,
            autopct='%1.1f%%',
            colors=binary_colors,
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax2.set_title('Binary Outcome Classification\n(90-day follow-up)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('90_day_mrs_detailed_analysis.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved detailed analysis to: 90_day_mrs_detailed_analysis.png")
        
        # Print detailed statistics
        print(f"\n📊 DETAILED STATISTICS:")
        print(f"   Total patients: {len(valid_data)}")
        print(f"   Missing data: {len(df) - len(valid_data)} patients")
        
        print(f"\n   Individual mRS scores:")
        for score in sorted(valid_data.unique()):
            count = len(valid_data[valid_data == score])
            percent = (count / len(valid_data)) * 100
            print(f"     mRS {score}: {count} patients ({percent:.1f}%)")
        
        print(f"\n   Binary classification:")
        print(f"     Good outcome (mRS 0-2): {good_outcome} patients ({good_outcome/len(valid_data)*100:.1f}%)")
        print(f"     Poor outcome (mRS 3-5): {poor_outcome} patients ({poor_outcome/len(valid_data)*100:.1f}%)")
        
        # Create bar chart
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        
        # Bar chart of mRS distribution
        mrs_counts.plot(kind='bar', ax=ax3, color=colors[:len(mrs_counts)], alpha=0.8)
        ax3.set_title('90-Day mRS Distribution - Bar Chart', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Modified Rankin Scale Score', fontsize=12)
        ax3.set_ylabel('Number of Patients', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(mrs_counts.values):
            ax3.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('90_day_mrs_bar_chart.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved bar chart to: 90_day_mrs_bar_chart.png")
        
        # Create summary table
        summary_df = pd.DataFrame({
            'mRS Score': mrs_counts.index,
            'Count': mrs_counts.values,
            'Percentage': (mrs_counts.values / len(valid_data) * 100).round(1),
            'Cumulative %': (mrs_counts.values.cumsum() / len(valid_data) * 100).round(1)
        })
        
        print(f"\n📋 SUMMARY TABLE:")
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_df.to_csv('mrs_distribution_summary.csv', index=False)
        print(f"\n💾 Saved summary table to: mrs_distribution_summary.csv")
        
        return summary_df
    else:
        print(f"   No valid data for {target_col}")
        return None

if __name__ == "__main__":
    analyze_mrs_distribution() 