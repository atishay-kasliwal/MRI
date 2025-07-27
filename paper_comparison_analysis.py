#!/usr/bin/env python3
"""
Paper Comparison Analysis
Compare our implementation with the original meningioma paper to identify missing components
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_paper_components():
    """Analyze what components are described in the original paper"""
    
    print("=== ORIGINAL PAPER COMPONENTS ANALYSIS ===\n")
    
    # Original paper methodology components
    paper_components = {
        "Dataset": {
            "Total patients": "306 patients",
            "Discovery cohort": "230 patients (75%)",
            "Replication cohort": "76 patients (25%)",
            "Target": "Ki-67 < 5% vs ≥ 5%",
            "Tumor type": "WHO Grade I meningiomas",
            "Time period": "2012-2018"
        },
        
        "Imaging": {
            "MRI sequences": "7 total sequences",
            "Sequences": ["T1-weighted", "T1+C (volumetric)", "T2-weighted", "T2-FLAIR", "DWI b0", "DWI b1000", "ADC map"],
            "Preprocessing": ["Coregistration", "Resampling to 1x1x1mm³", "N4ITK bias correction", "Intensity scaling [0,255]"],
            "Segmentation": ["ITK-SNAP v3.8.0", "Enhancing tumor on T1+C", "Peritumoral edema on T2-FLAIR"]
        },
        
        "Radiomics": {
            "Total features": "2520 radiomic features",
            "Morphologic features": "29 features",
            "Feature categories": ["Volume", "Shape", "Size", "Histogram", "Texture"],
            "Normalization": "Z-scoring",
            "Software": "CaPTk v.1.8.1"
        },
        
        "Machine Learning": {
            "Feature selection": "LASSO",
            "Classifier": "Linear SVM",
            "Cross-validation": "Nested 10-fold CV (outer) + 3-fold CV (inner)",
            "Hyperparameter tuning": "Bayesian optimization for SVM C parameter",
            "Model selection": "Least overfitting model closest to average performance"
        },
        
        "Performance": {
            "Discovery cohort": "AUC: 0.84 (95% CI: 0.78-0.90), Sensitivity: 84.1%, Specificity: 73.3%",
            "Replication cohort": "AUC: 0.83 (95% CI: 0.73-0.94), Sensitivity: 82.6%, Specificity: 85.5%",
            "Skull base tumors": "AUC: 0.86",
            "Non-skull base tumors": "AUC: 0.83",
            "Selected features": "60 features in final model"
        },
        
        "Feature Distribution": {
            "DWI (b0, b1000, ADC)": "27 features",
            "T1+C": "13 features",
            "T2-FLAIR": "9 features",
            "T1-weighted": "4 features",
            "T2-weighted": "3 features",
            "Morphologic": "4 features"
        },
        
        "Clinical Variables": {
            "Demographics": ["Age", "Sex", "Tumor laterality", "Skull base vs non-skull base"],
            "Tumor characteristics": ["Volume", "Peritumoral edema volume", "Location"],
            "Outcomes": ["Ki-67 proliferation index", "Tumor recurrence"]
        }
    }
    
    return paper_components

def analyze_our_implementation():
    """Analyze what we've implemented"""
    
    print("=== OUR IMPLEMENTATION ANALYSIS ===\n")
    
    our_implementation = {
        "Dataset": {
            "Total patients": "82 matched patients (radiomics + clinical)",
            "Discovery cohort": "61 patients (75%)",
            "Replication cohort": "21 patients (25%)",
            "Target": "Synthetic binary outcome (T1 mean + cross-modality entropy)",
            "Tumor type": "Stroke patients (not meningiomas)",
            "Time period": "2020-2022"
        },
        
        "Imaging": {
            "MRI sequences": "5 sequences",
            "Sequences": ["T1", "DWI", "ADC", "FLAIR", "T2"],
            "Preprocessing": ["N4 bias correction", "Basic preprocessing"],
            "Segmentation": ["Tumor masks applied"]
        },
        
        "Radiomics": {
            "Total features": "~107 patient-level features",
            "Feature categories": ["First-order statistics", "Shape features", "Texture features", "Cross-modality features"],
            "Normalization": "StandardScaler",
            "Software": "Custom implementation (numpy, scipy, SimpleITK)"
        },
        
        "Machine Learning": {
            "Feature selection": "LASSO",
            "Classifier": "Linear SVM",
            "Cross-validation": "5-fold CV for hyperparameter tuning",
            "Hyperparameter tuning": "Grid search for SVM C parameter",
            "Model selection": "Best C based on CV scores"
        },
        
        "Performance": {
            "Discovery cohort": "AUC: 1.000, Sensitivity: 1.000, Specificity: 1.000",
            "Replication cohort": "AUC: 0.891, Sensitivity: 0.857, Specificity: 0.917",
            "Selected features": "20 features in final model"
        },
        
        "Additional Analyses": {
            "MS Analysis": "Synthetic MS dataset with 100 patients",
            "mRS Analysis": "Comprehensive mRS analysis with 76 clinical patients",
            "Synthetic targets": "Multiple synthetic outcomes for demonstration"
        }
    }
    
    return our_implementation

def identify_missing_components(paper_components, our_implementation):
    """Identify what's missing from our implementation"""
    
    print("=== MISSING COMPONENTS ANALYSIS ===\n")
    
    missing_components = {
        "Critical Missing": [],
        "Important Missing": [],
        "Nice to Have": [],
        "Implemented Differently": []
    }
    
    # Critical missing components
    if "Ki-67" not in str(our_implementation):
        missing_components["Critical Missing"].append("Ki-67 proliferation index as target variable")
    
    if "meningioma" not in str(our_implementation).lower():
        missing_components["Critical Missing"].append("Meningioma-specific dataset and analysis")
    
    if "2520" not in str(our_implementation):
        missing_components["Critical Missing"].append("Comprehensive radiomics feature extraction (2520 features)")
    
    # Important missing components
    if "7 sequences" not in str(our_implementation):
        missing_components["Important Missing"].append("Full 7-sequence MP-MRI (missing T1+C volumetric)")
    
    if "nested cross-validation" not in str(our_implementation).lower():
        missing_components["Important Missing"].append("Nested cross-validation (10-fold outer + 3-fold inner)")
    
    if "bayesian optimization" not in str(our_implementation).lower():
        missing_components["Important Missing"].append("Bayesian optimization for hyperparameter tuning")
    
    if "morphologic features" not in str(our_implementation):
        missing_components["Important Missing"].append("Dedicated morphologic feature analysis (29 features)")
    
    # Nice to have components
    if "skull base" not in str(our_implementation).lower():
        missing_components["Nice to Have"].append("Skull base vs non-skull base subgroup analysis")
    
    if "peritumoral edema" not in str(our_implementation):
        missing_components["Nice to Have"].append("Peritumoral edema analysis")
    
    if "feature distribution" not in str(our_implementation):
        missing_components["Nice to Have"].append("Detailed feature distribution by MRI sequence")
    
    # Implemented differently
    missing_components["Implemented Differently"].append("Patient-level vs tumor-level analysis")
    missing_components["Implemented Differently"].append("Stroke vs meningioma pathology")
    missing_components["Implemented Differently"].append("Synthetic targets vs real Ki-67")
    missing_components["Implemented Differently"].append("Custom radiomics vs CaPTk software")
    
    return missing_components

def create_comparison_visualization(paper_components, our_implementation, missing_components):
    """Create visualization comparing paper vs our implementation"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Dataset comparison
    ax1 = axes[0, 0]
    categories = ['Total Patients', 'Discovery Cohort', 'Replication Cohort']
    paper_values = [306, 230, 76]
    our_values = [82, 61, 21]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, paper_values, width, label='Original Paper', alpha=0.8)
    ax1.bar(x + width/2, our_values, width, label='Our Implementation', alpha=0.8)
    
    ax1.set_xlabel('Cohort Type')
    ax1.set_ylabel('Number of Patients')
    ax1.set_title('Dataset Size Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance comparison
    ax2 = axes[0, 1]
    metrics = ['Discovery AUC', 'Replication AUC', 'Discovery Sens', 'Replication Sens']
    paper_perf = [0.84, 0.83, 0.841, 0.826]
    our_perf = [1.000, 0.891, 1.000, 0.857]
    
    x = np.arange(len(metrics))
    ax2.bar(x - width/2, paper_perf, width, label='Original Paper', alpha=0.8)
    ax2.bar(x + width/2, our_perf, width, label='Our Implementation', alpha=0.8)
    
    ax2.set_xlabel('Performance Metric')
    ax2.set_ylabel('Score')
    ax2.set_title('Model Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature comparison
    ax3 = axes[1, 0]
    feature_categories = ['Total Features', 'Selected Features', 'MRI Sequences']
    paper_features = [2520, 60, 7]
    our_features = [107, 20, 5]
    
    x = np.arange(len(feature_categories))
    ax3.bar(x - width/2, paper_features, width, label='Original Paper', alpha=0.8)
    ax3.bar(x + width/2, our_features, width, label='Our Implementation', alpha=0.8)
    
    ax3.set_xlabel('Feature Category')
    ax3.set_ylabel('Count')
    ax3.set_title('Feature Analysis Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(feature_categories)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Missing components summary
    ax4 = axes[1, 1]
    missing_counts = [len(missing_components["Critical Missing"]), 
                     len(missing_components["Important Missing"]),
                     len(missing_components["Nice to Have"]),
                     len(missing_components["Implemented Differently"])]
    missing_labels = ['Critical Missing', 'Important Missing', 'Nice to Have', 'Implemented Differently']
    
    colors = ['red', 'orange', 'yellow', 'blue']
    ax4.pie(missing_counts, labels=missing_labels, colors=colors, autopct='%1.0f', startangle=90)
    ax4.set_title('Missing Components Summary')
    
    plt.tight_layout()
    return fig

def create_recommendations(missing_components):
    """Create recommendations for improvement"""
    
    print("=== RECOMMENDATIONS FOR IMPROVEMENT ===\n")
    
    recommendations = {
        "High Priority": [],
        "Medium Priority": [],
        "Low Priority": []
    }
    
    # High priority recommendations
    for component in missing_components["Critical Missing"]:
        recommendations["High Priority"].append(f"Implement {component}")
    
    # Medium priority recommendations
    for component in missing_components["Important Missing"]:
        recommendations["Medium Priority"].append(f"Add {component}")
    
    # Low priority recommendations
    for component in missing_components["Nice to Have"]:
        recommendations["Low Priority"].append(f"Consider {component}")
    
    return recommendations

def main():
    """Main function for paper comparison analysis"""
    
    print("=== PAPER COMPARISON ANALYSIS ===\n")
    
    # Analyze components
    paper_components = analyze_paper_components()
    our_implementation = analyze_our_implementation()
    
    # Identify missing components
    missing_components = identify_missing_components(paper_components, our_implementation)
    
    # Print missing components
    for category, components in missing_components.items():
        if components:
            print(f"{category}:")
            for component in components:
                print(f"  - {component}")
            print()
    
    # Create recommendations
    recommendations = create_recommendations(missing_components)
    
    for priority, recs in recommendations.items():
        if recs:
            print(f"{priority}:")
            for rec in recs:
                print(f"  - {rec}")
            print()
    
    # Create visualization
    fig = create_comparison_visualization(paper_components, our_implementation, missing_components)
    fig.savefig('paper_comparison_analysis.png', dpi=300, bbox_inches='tight')
    
    # Save detailed analysis
    with open('paper_comparison_report.txt', 'w') as f:
        f.write("=== PAPER COMPARISON ANALYSIS REPORT ===\n\n")
        
        f.write("ORIGINAL PAPER COMPONENTS:\n")
        for category, details in paper_components.items():
            f.write(f"\n{category}:\n")
            if isinstance(details, dict):
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {details}\n")
        
        f.write("\nOUR IMPLEMENTATION:\n")
        for category, details in our_implementation.items():
            f.write(f"\n{category}:\n")
            if isinstance(details, dict):
                for key, value in details.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {details}\n")
        
        f.write("\nMISSING COMPONENTS:\n")
        for category, components in missing_components.items():
            if components:
                f.write(f"\n{category}:\n")
                for component in components:
                    f.write(f"  - {component}\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        for priority, recs in recommendations.items():
            if recs:
                f.write(f"\n{priority}:\n")
                for rec in recs:
                    f.write(f"  - {rec}\n")
    
    print("=== ANALYSIS COMPLETED ===")
    print("Files generated:")
    print("  - paper_comparison_analysis.png")
    print("  - paper_comparison_report.txt")
    
    return paper_components, our_implementation, missing_components, recommendations

if __name__ == "__main__":
    main() 