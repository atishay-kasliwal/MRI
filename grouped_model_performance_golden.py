#!/usr/bin/env python3
"""
Grouped Model Performance Chart - Golden Theme
Create a grouped bar chart with golden theme, white background, and no floating text
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_grouped_model_performance_golden():
    print("=== GROUPED MODEL PERFORMANCE CHART - GOLDEN THEME ===\n")
    
    # Load the results from previous analysis
    try:
        results_df = pd.read_csv("model_performance_results.csv")
        print(f"📊 Loaded results: {len(results_df)} model configurations")
    except FileNotFoundError:
        print("⚠️  Model performance results not found. Please run model_performance_comparison.py first.")
        return None
    
    # Filter to include only the models we want to show
    models_to_show = ['SVM', 'Random Forest', 'Extra Trees', 'Gradient Boosting', 'CatBoost']
    feature_sets_to_show = ['All Features', 'Top 100', 'Top 70', 'Top 50']
    
    # Filter the data
    filtered_df = results_df[
        (results_df['Model'].isin(models_to_show)) & 
        (results_df['Feature_Set'].isin(feature_sets_to_show))
    ].copy()
    
    # Create pivot table for easier plotting
    pivot_df = filtered_df.pivot(index='Model', columns='Feature_Set', values='F1_Score')
    
    # Reorder columns to match desired order
    column_order = ['All Features', 'Top 100', 'Top 70', 'Top 50']
    pivot_df = pivot_df[column_order]
    
    print(f"📊 Pivot table shape: {pivot_df.shape}")
    print(f"📊 Models: {list(pivot_df.index)}")
    print(f"📊 Feature sets: {list(pivot_df.columns)}")
    
    # Set up golden theme colors
    golden_colors = {
        'All Features': '#D4AF37',      # Classic gold
        'Top 100': '#FFD700',           # Bright gold
        'Top 70': '#B8860B',            # Dark goldenrod
        'Top 50': '#F4A460'             # Sandy brown
    }
    
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Set up the bar positions
    models = pivot_df.index
    feature_sets = pivot_df.columns
    x = np.arange(len(models))
    width = 0.2  # Width of each bar
    
    # Create bars for each feature set
    bars = []
    for i, (feature_set, color) in enumerate(zip(feature_sets, golden_colors.values())):
        values = pivot_df[feature_set].values
        bar = ax.bar(x + i * width, values, width, label=feature_set, 
                    color=color, alpha=0.9, edgecolor='#8B7355', linewidth=1)
        bars.append(bar)
        
        # Add value labels on bars (no floating text, just on bars)
        for j, (bar_rect, value) in enumerate(zip(bar, values)):
            if not pd.isna(value):
                ax.text(bar_rect.get_x() + bar_rect.get_width()/2, bar_rect.get_height() + 0.005,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold', 
                       fontsize=10, color='#8B4513')
    
    # Customize the chart with golden theme
    ax.set_xlabel('Models', fontsize=14, fontweight='bold', color='#8B4513')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold', color='#8B4513')
    ax.set_title('Model Performance Comparison by Feature Set\nF1 Score Performance', 
                fontsize=16, fontweight='bold', pad=20, color='#8B4513')
    
    # Set x-axis ticks and labels
    ax.set_xticks(x + width * 1.5)  # Center the labels
    ax.set_xticklabels(models, fontsize=12, fontweight='bold', color='#8B4513')
    
    # Add legend with golden theme
    legend = ax.legend(title='Feature Sets', title_fontsize=12, fontsize=11, 
                      loc='upper right', bbox_to_anchor=(1, 1))
    legend.get_title().set_color('#8B4513')
    for text in legend.get_texts():
        text.set_color('#8B4513')
    
    # Add subtle grid
    ax.grid(axis='y', alpha=0.2, linestyle='--', color='#D4AF37')
    ax.set_axisbelow(True)
    
    # Set y-axis limits
    max_value = pivot_df.max().max()
    ax.set_ylim(0, max_value * 1.1)
    
    # Customize spines with golden color
    for spine in ax.spines.values():
        spine.set_color('#D4AF37')
        spine.set_linewidth(1.5)
    
    # Customize tick colors
    ax.tick_params(axis='both', colors='#8B4513')
    
    plt.tight_layout()
    plt.savefig('grouped_model_performance_golden.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"💾 Saved golden-themed grouped model performance chart to: grouped_model_performance_golden.png")
    
    # Create similar charts for other metrics with golden theme
    metrics = ['AUC', 'Accuracy', 'Precision']
    metric_names = ['AUC Score', 'Accuracy', 'Precision']
    
    for metric, metric_name in zip(metrics, metric_names):
        # Create pivot table for this metric
        pivot_metric = filtered_df.pivot(index='Model', columns='Feature_Set', values=metric)
        pivot_metric = pivot_metric[column_order]
        
        # Create the chart
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        # Set white background
        fig2.patch.set_facecolor('white')
        ax2.set_facecolor('white')
        
        # Create bars for each feature set
        for i, (feature_set, color) in enumerate(zip(feature_sets, golden_colors.values())):
            values = pivot_metric[feature_set].values
            bar = ax2.bar(x + i * width, values, width, label=feature_set, 
                         color=color, alpha=0.9, edgecolor='#8B7355', linewidth=1)
            
            # Add value labels on bars
            for j, (bar_rect, value) in enumerate(zip(bar, values)):
                if not pd.isna(value):
                    ax2.text(bar_rect.get_x() + bar_rect.get_width()/2, bar_rect.get_height() + 0.005,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold', 
                            fontsize=10, color='#8B4513')
        
        # Customize the chart with golden theme
        ax2.set_xlabel('Models', fontsize=14, fontweight='bold', color='#8B4513')
        ax2.set_ylabel(metric_name, fontsize=14, fontweight='bold', color='#8B4513')
        ax2.set_title(f'Model Performance Comparison by Feature Set\n{metric_name} Performance', 
                     fontsize=16, fontweight='bold', pad=20, color='#8B4513')
        
        # Set x-axis ticks and labels
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels(models, fontsize=12, fontweight='bold', color='#8B4513')
        
        # Add legend with golden theme
        legend2 = ax2.legend(title='Feature Sets', title_fontsize=12, fontsize=11, 
                            loc='upper right', bbox_to_anchor=(1, 1))
        legend2.get_title().set_color('#8B4513')
        for text in legend2.get_texts():
            text.set_color('#8B4513')
        
        # Add subtle grid
        ax2.grid(axis='y', alpha=0.2, linestyle='--', color='#D4AF37')
        ax2.set_axisbelow(True)
        
        # Set y-axis limits
        max_value = pivot_metric.max().max()
        ax2.set_ylim(0, max_value * 1.1)
        
        # Customize spines with golden color
        for spine in ax2.spines.values():
            spine.set_color('#D4AF37')
            spine.set_linewidth(1.5)
        
        # Customize tick colors
        ax2.tick_params(axis='both', colors='#8B4513')
        
        plt.tight_layout()
        plt.savefig(f'grouped_model_performance_golden_{metric.lower()}.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"💾 Saved golden-themed {metric_name} chart to: grouped_model_performance_golden_{metric.lower()}.png")
    
    # Print detailed comparison
    print(f"\n📊 DETAILED COMPARISON:")
    print("=" * 80)
    
    for model in models_to_show:
        model_data = filtered_df[filtered_df['Model'] == model].sort_values('F1_Score', ascending=False)
        print(f"\n🏆 {model.upper()}:")
        print("-" * 40)
        
        for _, row in model_data.iterrows():
            print(f"  {row['Feature_Set']}: F1={row['F1_Score']:.3f}, AUC={row['AUC']:.3f}, Acc={row['Accuracy']:.3f}")
    
    # Save the pivot table for reference
    pivot_df.to_csv('grouped_model_performance_golden_pivot.csv')
    print(f"\n💾 Saved pivot table to: grouped_model_performance_golden_pivot.csv")
    
    return pivot_df

if __name__ == "__main__":
    create_grouped_model_performance_golden() 