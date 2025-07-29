#!/usr/bin/env python3
"""
Enhanced mRS Category Split Pie Chart - Aesthetically Appealing
Create a beautiful pie chart showing mRS categories with train/test splits
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_mrs_category_split_pie():
    print("=== ENHANCED MRS CATEGORY SPLIT PIE CHART ===\n")
    
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
    
    # Create the enhanced pie chart
    fig, ax = plt.subplots(figsize=(18, 14))
    
    # Set beautiful background with gradient
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')
    
    # Prepare data for pie chart
    sizes = []
    labels = []
    colors = []
    
    # Golden theme color palette - same as top 20 features bar chart
    golden_palette = [
        '#FFD700', '#D4AF37', '#B8860B', '#F4A460', '#DEB887', '#CD853F',  # Golden shades for train
        '#FFE4B5', '#F5DEB3', '#EEE8AA', '#F0E68C', '#BDB76B', '#D2B48C'   # Lighter golden shades for test
    ]
    
    for i, mrs_score in enumerate(sorted(mrs_counts.index)):
        # Train slice (darker golden color)
        sizes.append(train_counts[mrs_score])
        labels.append(f'mRS {int(mrs_score)} Train')
        colors.append(golden_palette[i])
        
        # Test slice (lighter golden shade)
        sizes.append(test_counts[mrs_score])
        labels.append(f'mRS {int(mrs_score)} Test')
        colors.append(golden_palette[i + 6])  # Lighter shade
    
    # Create enhanced pie chart with better styling
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        autopct='',  # No percentage text on pie
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold', 'color': '#8B4513'},
        wedgeprops={
            'edgecolor': '#8B7355', 
            'linewidth': 2.5,
            'alpha': 0.9
        },
        explode=[0.02] * len(sizes)  # Slight separation between slices
    )
    
    # Add beautiful legend with enhanced styling
    legend_elements = []
    legend_labels = []
    
    for i, mrs_score in enumerate(sorted(mrs_counts.index)):
        # Train legend element
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                           facecolor=golden_palette[i], 
                                           edgecolor='#8B7355', 
                                           linewidth=2,
                                           alpha=0.9))
        legend_labels.append(f'mRS {int(mrs_score)} Train ({train_counts[mrs_score]} patients)')
        
        # Test legend element
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                           facecolor=golden_palette[i + 6], 
                                           edgecolor='#8B7355', 
                                           linewidth=2,
                                           alpha=0.9))
        legend_labels.append(f'mRS {int(mrs_score)} Test ({test_counts[mrs_score]} patients)')
    
    # Place enhanced legend
    legend = ax.legend(legend_elements, legend_labels, 
                      loc='center left', bbox_to_anchor=(1.05, 0.5),
                      fontsize=12, frameon=True, 
                      fancybox=True, shadow=True,
                      facecolor='white', edgecolor='#D4AF37',
                      title='Dataset Split by mRS Score')
    
    # Customize legend title separately
    legend.get_title().set_fontsize(14)
    legend.get_title().set_fontweight('bold')
    
    # Customize legend title
    legend.get_title().set_color('#8B4513')
    
    # Enhanced title with better styling
    title_text = 'Modified Rankin Scale (mRS) Distribution\nwith Stratified Train/Test Split (80/20)'
    ax.set_title(title_text, 
                fontsize=20, fontweight='bold', color='#8B4513', pad=30,
                fontfamily='Arial')
    
    # Remove summary text box - keeping it clean without text overlay
    
    # Add subtle grid for better visual appeal
    ax.grid(True, alpha=0.1, linestyle='-', color='#D4AF37')
    
    # Customize spines with elegant styling
    for spine in ax.spines.values():
        spine.set_color('#D4AF37')
        spine.set_linewidth(2)
    
    # Add a subtle background pattern
    ax.set_facecolor('#FAFAFA')
    
    # Add watermark or decorative element
    ax.text(0.5, 0.02, 'Radiomics Analysis | mRS Outcome Prediction', 
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=12, color='#8B4513', alpha=0.7,
            fontfamily='Arial', fontstyle='italic')
    
    plt.tight_layout()
    plt.savefig('mrs_category_split_pie_enhanced.png', dpi=300, bbox_inches='tight', 
                facecolor='#FAFAFA', edgecolor='none')
    print(f"💾 Saved enhanced mRS category split pie chart to: mrs_category_split_pie_enhanced.png")
    
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
        
        print(f"{int(mrs_score):<8} {total_mrs:<10} {train_mrs:<10} {test_mrs:<10} {train_pct:<10.1f} {test_pct:<10.1f}")
    
    # Calculate totals
    total_patients = sum(mrs_counts.values)
    train_total = sum(train_counts.values())
    test_total = sum(test_counts.values())
    
    print("-" * 80)
    print(f"{'Total':<8} {total_patients:<10} {train_total:<10} {test_total:<10} {(train_total/total_patients*100):<10.1f} {(test_total/total_patients*100):<10.1f}")
    
    # Save enhanced category split data
    category_data = []
    for mrs_score in sorted(mrs_counts.index):
        category_data.append({
            'mRS_Score': int(mrs_score),
            'Total_Patients': mrs_counts[mrs_score],
            'Train_Patients': train_counts[mrs_score],
            'Test_Patients': test_counts[mrs_score],
            'Train_Percentage': (train_counts[mrs_score] / mrs_counts[mrs_score]) * 100,
            'Test_Percentage': (test_counts[mrs_score] / mrs_counts[mrs_score]) * 100
        })
    
    category_df = pd.DataFrame(category_data)
    category_df.to_csv('mrs_category_split_data_enhanced.csv', index=False)
    print(f"\n💾 Saved enhanced category split data to: mrs_category_split_data_enhanced.csv")
    
    return category_df

if __name__ == "__main__":
    create_enhanced_mrs_category_split_pie() 