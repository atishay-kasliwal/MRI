#!/usr/bin/env python3
"""
mRS Split Pie Chart - Golden Theme
Create a single pie chart showing 80/20 train-test split with golden theme
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_mrs_split_pie_golden():
    print("=== MRS SPLIT PIE CHART - GOLDEN THEME ===\n")
    
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
    print(f"📋 Binary distribution: Good (0-2): {sum(binary_target == 1)}, Poor (3-5): {sum(binary_target == 0)}")
    
    # Calculate 80/20 split
    total_patients = len(binary_target)
    train_size = int(0.8 * total_patients)
    test_size = total_patients - train_size
    
    # Calculate class distribution in train and test
    from sklearn.model_selection import train_test_split
    X_dummy = np.arange(len(binary_target)).reshape(-1, 1)  # Dummy features for splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X_dummy, binary_target, test_size=0.2, random_state=42, stratify=binary_target
    )
    
    # Count classes in train and test
    train_good = sum(y_train == 1)
    train_poor = sum(y_train == 0)
    test_good = sum(y_test == 1)
    test_poor = sum(y_test == 0)
    
    print(f"📊 Train set: {len(y_train)} patients (Good: {train_good}, Poor: {train_poor})")
    print(f"📊 Test set: {len(y_test)} patients (Good: {test_good}, Poor: {test_poor})")
    
    # Create the pie chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Prepare data for pie chart
    sizes = [train_good, train_poor, test_good, test_poor]
    labels = ['Train (Good)', 'Train (Poor)', 'Test (Good)', 'Test (Poor)']
    
    # Golden theme colors
    golden_colors = ['#FFD700', '#D4AF37', '#B8860B', '#F4A460']  # Different shades of gold
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels,
        autopct='%1.1f%%',
        colors=golden_colors,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold', 'color': '#8B4513'},
        wedgeprops={'edgecolor': '#8B7355', 'linewidth': 2}
    )
    
    # Enhance text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    # Customize title
    ax.set_title('mRS Dataset Split - 80/20 Train/Test Distribution\nStratified by Outcome (Good vs Poor)', 
                fontsize=16, fontweight='bold', color='#8B4513', pad=20)
    
    # Add summary statistics
    train_percent = (len(y_train) / total_patients) * 100
    test_percent = (len(y_test) / total_patients) * 100
    
    summary_text = f"Total Patients: {total_patients}\n"
    summary_text += f"Train Set: {len(y_train)} ({train_percent:.1f}%)\n"
    summary_text += f"Test Set: {len(y_test)} ({test_percent:.1f}%)\n"
    summary_text += f"Train Good: {train_good} ({train_good/len(y_train)*100:.1f}%)\n"
    summary_text += f"Train Poor: {train_poor} ({train_poor/len(y_train)*100:.1f}%)\n"
    summary_text += f"Test Good: {test_good} ({test_good/len(y_test)*100:.1f}%)\n"
    summary_text += f"Test Poor: {test_poor} ({test_poor/len(y_test)*100:.1f}%)"
    
    # Add summary box
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
            facecolor='white', alpha=0.9, edgecolor='#D4AF37', linewidth=2))
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#D4AF37')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('mrs_split_pie_chart_golden.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"💾 Saved golden-themed mRS split pie chart to: mrs_split_pie_chart_golden.png")
    
    # Create detailed breakdown table
    print(f"\n📊 DETAILED SPLIT BREAKDOWN:")
    print("=" * 60)
    print(f"{'Set':<15} {'Good (0-2)':<15} {'Poor (3-5)':<15} {'Total':<10}")
    print("-" * 60)
    print(f"{'Train':<15} {train_good:<15} {train_poor:<15} {len(y_train):<10}")
    print(f"{'Test':<15} {test_good:<15} {test_poor:<15} {len(y_test):<10}")
    print("-" * 60)
    print(f"{'Total':<15} {train_good + test_good:<15} {train_poor + test_poor:<15} {total_patients:<10}")
    
    # Calculate percentages
    print(f"\n📊 PERCENTAGE BREAKDOWN:")
    print("=" * 60)
    print(f"Train Set: {len(y_train)} patients ({train_percent:.1f}%)")
    print(f"  - Good Outcome: {train_good} ({train_good/len(y_train)*100:.1f}%)")
    print(f"  - Poor Outcome: {train_poor} ({train_poor/len(y_train)*100:.1f}%)")
    print(f"Test Set: {len(y_test)} patients ({test_percent:.1f}%)")
    print(f"  - Good Outcome: {test_good} ({test_good/len(y_test)*100:.1f}%)")
    print(f"  - Poor Outcome: {test_poor} ({test_poor/len(y_test)*100:.1f}%)")
    
    # Save split data
    split_data = {
        'Set': ['Train', 'Test', 'Total'],
        'Good_Outcome': [train_good, test_good, train_good + test_good],
        'Poor_Outcome': [train_poor, test_poor, train_poor + test_poor],
        'Total': [len(y_train), len(y_test), total_patients],
        'Percentage': [train_percent, test_percent, 100.0]
    }
    
    split_df = pd.DataFrame(split_data)
    split_df.to_csv('mrs_split_data_golden.csv', index=False)
    print(f"\n💾 Saved split data to: mrs_split_data_golden.csv")
    
    return split_df

if __name__ == "__main__":
    create_mrs_split_pie_golden() 