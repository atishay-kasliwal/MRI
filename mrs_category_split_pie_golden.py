#!/usr/bin/env python3
"""
mRS Category Split Pie Chart - Golden Theme
Create a pie chart showing mRS categories with train/test splits using different shades
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_mrs_category_split_pie_golden():
    print("=== MRS CATEGORY SPLIT PIE CHART - GOLDEN THEME ===\n")
    
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
    
    print(f"📋 Valid target data: {len(target_data)} patients")
    
    # Get mRS distribution
    mrs_counts = target_data.value_counts().sort_index()
    print(f"📊 mRS Distribution: {dict(mrs_counts)}")
    
    # Calculate train/test split for each mRS category
    from sklearn.model_selection import train_test_split
    
    train_counts = {}
    test_counts = {}
    
    for mrs_score in mrs_counts.index:
        # Get indices for this mRS score
        mrs_indices = target_data[target_data == mrs_score].index
        
        # Split 80/20
        train_indices, test_indices = train_test_split(
            mrs_indices, test_size=0.2, random_state=42
        )
        
        train_counts[mrs_score] = len(train_indices)
        test_counts[mrs_score] = len(test_indices)
        
        print(f"mRS {mrs_score}: {len(mrs_indices)} total -> {len(train_indices)} train, {len(test_indices)} test")
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Prepare data for pie chart
    sizes = []
    labels = []
    colors = []
    
    # Golden color palette for different mRS scores
    golden_palette = ['#FFD700', '#D4AF37', '#B8860B', '#F4A460', '#DEB887', '#CD853F']
    
    for i, mrs_score in enumerate(sorted(mrs_counts.index)):
        # Train slice (darker shade)
        sizes.append(train_counts[mrs_score])
        labels.append(f'mRS {mrs_score} Train')
        colors.append(golden_palette[i])
        
        # Test slice (lighter shade)
        sizes.append(test_counts[mrs_score])
        labels.append(f'mRS {mrs_score} Test')
        colors.append(plt.cm.Oranges(0.3 + i * 0.1))  # Lighter shade
    
    # Create pie chart with no autopct to avoid text overlap
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        autopct='',  # No percentage text on pie
        colors=colors,
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold', 'color': '#8B4513'},
        wedgeprops={'edgecolor': '#8B7355', 'linewidth': 1.5}
    )
    
    # Add legend outside the pie chart
    legend_elements = []
    legend_labels = []
    
    for i, mrs_score in enumerate(sorted(mrs_counts.index)):
        # Train legend element
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=golden_palette[i], edgecolor='#8B7355', linewidth=1.5))
        legend_labels.append(f'mRS {mrs_score} Train ({train_counts[mrs_score]} patients)')
        
        # Test legend element
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.Oranges(0.3 + i * 0.1), edgecolor='#8B7355', linewidth=1.5))
        legend_labels.append(f'mRS {mrs_score} Test ({test_counts[mrs_score]} patients)')
    
    # Place legend outside the pie chart
    ax.legend(legend_elements, legend_labels, 
             loc='center left', bbox_to_anchor=(1, 0.5),
             fontsize=10, frameon=True, 
             fancybox=True, shadow=True,
             facecolor='white', edgecolor='#D4AF37')
    
    # Customize title
    ax.set_title('mRS Category Distribution with Train/Test Split\n80/20 Stratified Split by mRS Score', 
                fontsize=16, fontweight='bold', color='#8B4513', pad=20)
    
    # Remove summary text box - keeping it clean without text overlay
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#D4AF37')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('mrs_category_split_pie_golden.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"💾 Saved golden-themed mRS category split pie chart to: mrs_category_split_pie_golden.png")
    
    # Create detailed breakdown table
    print(f"\n📊 DETAILED MRS CATEGORY BREAKDOWN:")
    print("=" * 80)
    print(f"{'mRS':<8} {'Total':<10} {'Train':<10} {'Test':<10} {'Train%':<10} {'Test%':<10}")
    print("-" * 80)
    
    for mrs_score in sorted(mrs_counts.index):
        total_mrs = mrs_counts[mrs_score]
        train_mrs = train_counts[mrs_score]
        test_mrs = test_counts[mrs_score]
        train_pct = (train_mrs / total_mrs) * 100
        test_pct = (test_mrs / total_mrs) * 100
        
        print(f"{mrs_score:<8} {total_mrs:<10} {train_mrs:<10} {test_mrs:<10} {train_pct:<10.1f} {test_pct:<10.1f}")
    
    # Calculate totals
    total_patients = sum(mrs_counts.values)
    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    
    print("-" * 80)
    print(f"{'Total':<8} {total_patients:<10} {train_total:<10} {test_total:<10} {(train_total/total_patients*100):<10.1f} {(test_total/total_patients*100):<10.1f}")
    
    # Save category split data
    category_data = []
    for mrs_score in sorted(mrs_counts.index):
        category_data.append({
            'mRS_Score': mrs_score,
            'Total_Patients': mrs_counts[mrs_score],
            'Train_Patients': train_counts[mrs_score],
            'Test_Patients': test_counts[mrs_score],
            'Train_Percentage': (train_counts[mrs_score] / mrs_counts[mrs_score]) * 100,
            'Test_Percentage': (test_counts[mrs_score] / mrs_counts[mrs_score]) * 100
        })
    
    category_df = pd.DataFrame(category_data)
    category_df.to_csv('mrs_category_split_data_golden.csv', index=False)
    print(f"\n💾 Saved category split data to: mrs_category_split_data_golden.csv")
    
    return category_df

if __name__ == "__main__":
    create_mrs_category_split_pie_golden() 