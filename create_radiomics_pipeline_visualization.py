#!/usr/bin/env python3
"""
Create Radiomics Pipeline Visualization
Generate a comprehensive pipeline diagram showing all steps in the radiomics similarity analysis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better aesthetics
plt.style.use('seaborn-v0_8')

def create_pipeline_visualization():
    """Create a comprehensive pipeline visualization"""
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define colors for different stages
    colors = {
        'data': '#FF6B6B',
        'preprocessing': '#4ECDC4', 
        'analysis': '#45B7D1',
        'clustering': '#96CEB4',
        'visualization': '#FFEAA7',
        'reporting': '#DDA0DD'
    }
    
    # Define pipeline stages
    stages = [
        {
            'name': 'Data Loading & Preparation',
            'color': colors['data'],
            'x': 10,
            'y': 85,
            'width': 15,
            'height': 8,
            'steps': [
                'Load radiomics data (2020-2022)',
                'Patient-level feature extraction',
                'Modality-specific features',
                'Cross-modality features',
                'Data quality assessment'
            ]
        },
        {
            'name': 'Feature Preprocessing',
            'color': colors['preprocessing'],
            'x': 30,
            'y': 85,
            'width': 15,
            'height': 8,
            'steps': [
                'Handle missing values',
                'Feature standardization',
                'Outlier detection',
                'Feature selection',
                'Data normalization'
            ]
        },
        {
            'name': 'Correlation Analysis',
            'color': colors['analysis'],
            'x': 50,
            'y': 85,
            'width': 15,
            'height': 8,
            'steps': [
                'Pearson correlation matrix',
                'High correlation detection',
                'Feature pair analysis',
                'Modality correlation',
                'Temporal correlation'
            ]
        },
        {
            'name': 'Clustering Analysis',
            'color': colors['clustering'],
            'x': 70,
            'y': 85,
            'width': 15,
            'height': 8,
            'steps': [
                'K-means clustering',
                'Silhouette analysis',
                'Optimal k determination',
                'Cluster validation',
                'Cluster characteristics'
            ]
        },
        {
            'name': 'Dimensionality Reduction',
            'color': colors['analysis'],
            'x': 10,
            'y': 65,
            'width': 15,
            'height': 8,
            'steps': [
                'Principal Component Analysis',
                't-SNE visualization',
                'Variance explanation',
                'Component selection',
                'Dimensionality assessment'
            ]
        },
        {
            'name': 'Modality Analysis',
            'color': colors['analysis'],
            'x': 30,
            'y': 65,
            'width': 15,
            'height': 8,
            'steps': [
                'Modality grouping',
                'Within-modality similarity',
                'Between-modality similarity',
                'Cross-modality features',
                'Modality importance'
            ]
        },
        {
            'name': 'Temporal Analysis',
            'color': colors['analysis'],
            'x': 50,
            'y': 65,
            'width': 15,
            'height': 8,
            'steps': [
                'Year-wise analysis',
                'Temporal consistency',
                'Feature stability',
                'Availability patterns',
                'Temporal trends'
            ]
        },
        {
            'name': 'Feature Importance',
            'color': colors['analysis'],
            'x': 70,
            'y': 65,
            'width': 15,
            'height': 8,
            'steps': [
                'Cluster variance analysis',
                'Feature ranking',
                'Importance scoring',
                'Stability assessment',
                'Feature selection'
            ]
        },
        {
            'name': 'Visualization Generation',
            'color': colors['visualization'],
            'x': 10,
            'y': 45,
            'width': 15,
            'height': 8,
            'steps': [
                'Correlation heatmaps',
                'Cluster visualizations',
                'PCA/t-SNE plots',
                'Modality matrices',
                'Temporal plots'
            ]
        },
        {
            'name': 'Quality Metrics',
            'color': colors['clustering'],
            'x': 30,
            'y': 45,
            'width': 15,
            'height': 8,
            'steps': [
                'Silhouette scores',
                'Calinski-Harabasz',
                'Davies-Bouldin',
                'Cluster separation',
                'Validation metrics'
            ]
        },
        {
            'name': 'Network Analysis',
            'color': colors['analysis'],
            'x': 50,
            'y': 45,
            'width': 15,
            'height': 8,
            'steps': [
                'Feature networks',
                'Correlation networks',
                'Modality networks',
                'Cluster networks',
                'Network metrics'
            ]
        },
        {
            'name': 'Statistical Analysis',
            'color': colors['analysis'],
            'x': 70,
            'y': 45,
            'width': 15,
            'height': 8,
            'steps': [
                'Distribution analysis',
                'Statistical tests',
                'Significance testing',
                'Effect sizes',
                'Confidence intervals'
            ]
        },
        {
            'name': 'Results Compilation',
            'color': colors['reporting'],
            'x': 10,
            'y': 25,
            'width': 15,
            'height': 8,
            'steps': [
                'Summary statistics',
                'Key findings',
                'Insights compilation',
                'Recommendations',
                'Limitations'
            ]
        },
        {
            'name': 'Report Generation',
            'color': colors['reporting'],
            'x': 30,
            'y': 25,
            'width': 15,
            'height': 8,
            'steps': [
                'PDF report creation',
                'Visualization compilation',
                'Text formatting',
                'Professional styling',
                'Final review'
            ]
        },
        {
            'name': 'Output Files',
            'color': colors['reporting'],
            'x': 50,
            'y': 25,
            'width': 15,
            'height': 8,
            'steps': [
                'PNG visualizations',
                'CSV data files',
                'PDF reports',
                'Summary text files',
                'Analysis logs'
            ]
        },
        {
            'name': 'Validation & Review',
            'color': colors['reporting'],
            'x': 70,
            'y': 25,
            'width': 15,
            'height': 8,
            'steps': [
                'Quality assurance',
                'Peer review',
                'Method validation',
                'Result verification',
                'Documentation'
            ]
        }
    ]
    
    # Draw stage boxes
    for stage in stages:
        # Create rounded rectangle for stage
        box = FancyBboxPatch(
            (stage['x'], stage['y']), 
            stage['width'], 
            stage['height'],
            boxstyle="round,pad=0.1",
            facecolor=stage['color'],
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )
        ax.add_patch(box)
        
        # Add stage title
        ax.text(stage['x'] + stage['width']/2, stage['y'] + stage['height'] - 0.5, 
                stage['name'], ha='center', va='top', fontsize=10, fontweight='bold',
                color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
        
        # Add steps
        for i, step in enumerate(stage['steps']):
            y_pos = stage['y'] + stage['height'] - 1.5 - i * 0.8
            if y_pos > stage['y'] + 0.5:  # Ensure text stays within box
                ax.text(stage['x'] + 0.2, y_pos, f'• {step}', 
                       ha='left', va='top', fontsize=7, color='white',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='black', alpha=0.5))
    
    # Add flow arrows
    arrows = [
        # Top row flow
        ((25, 89), (30, 89)),  # Data -> Preprocessing
        ((45, 89), (50, 89)),  # Preprocessing -> Correlation
        ((65, 89), (70, 89)),  # Correlation -> Clustering
        
        # Middle row flow
        ((25, 69), (30, 69)),  # Dimensionality -> Modality
        ((45, 69), (50, 69)),  # Modality -> Temporal
        ((65, 69), (70, 69)),  # Temporal -> Feature Importance
        
        # Bottom row flow
        ((25, 29), (30, 29)),  # Results -> Report
        ((45, 29), (50, 29)),  # Report -> Output
        ((65, 29), (70, 29)),  # Output -> Validation
        
        # Vertical connections
        ((17.5, 85), (17.5, 77)),  # Data -> Dimensionality
        ((37.5, 85), (37.5, 77)),  # Preprocessing -> Modality
        ((57.5, 85), (57.5, 77)),  # Correlation -> Temporal
        ((77.5, 85), (77.5, 77)),  # Clustering -> Feature Importance
        
        ((17.5, 65), (17.5, 57)),  # Dimensionality -> Visualization
        ((37.5, 65), (37.5, 57)),  # Modality -> Quality Metrics
        ((57.5, 65), (57.5, 57)),  # Temporal -> Network Analysis
        ((77.5, 65), (77.5, 57)),  # Feature Importance -> Statistical
        
        ((17.5, 45), (17.5, 37)),  # Visualization -> Results
        ((37.5, 45), (37.5, 37)),  # Quality Metrics -> Report
        ((57.5, 45), (57.5, 37)),  # Network Analysis -> Output
        ((77.5, 45), (77.5, 37)),  # Statistical -> Validation
    ]
    
    # Draw arrows
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc="black", ec="black", linewidth=2)
        ax.add_patch(arrow)
    
    # Add main title
    ax.text(50, 95, 'RADIOMICS SIMILARITY ANALYSIS PIPELINE', 
            ha='center', va='center', fontsize=24, fontweight='bold',
            color='#2c3e50', bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', alpha=0.9))
    
    # Add subtitle
    ax.text(50, 92, 'Comprehensive Workflow for Feature Similarity and Pattern Discovery', 
            ha='center', va='center', fontsize=14, color='#34495e',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#bdc3c7', alpha=0.7))
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['data'], label='Data Processing'),
        patches.Patch(color=colors['preprocessing'], label='Preprocessing'),
        patches.Patch(color=colors['analysis'], label='Analysis'),
        patches.Patch(color=colors['clustering'], label='Clustering'),
        patches.Patch(color=colors['visualization'], label='Visualization'),
        patches.Patch(color=colors['reporting'], label='Reporting')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98),
             fontsize=12, framealpha=0.9, fancybox=True, shadow=True)
    
    # Add statistics box
    stats_text = """
    📊 PIPELINE STATISTICS:
    
    • Total Stages: 16
    • Analysis Steps: 80+
    • Output Files: 5+
    • Visualizations: 20 panels
    • Report Pages: 6
    • Processing Time: ~2-3 minutes
    • Data Points: 150 patients
    • Features: 82 radiomics
    • Modalities: 6 (T1, DWI, ADC, FLAIR, T2, Cross-Modality)
    • Years: 2020-2022
    """
    
    ax.text(85, 80, stats_text, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f1f2f6', alpha=0.9,
                     edgecolor='#34495e', linewidth=2))
    
    # Add methodology box
    methodology_text = """
    🔬 METHODOLOGY:
    
    • Unsupervised Learning
    • K-means Clustering
    • Silhouette Analysis
    • PCA & t-SNE
    • Correlation Analysis
    • Network Analysis
    • Statistical Testing
    • Quality Metrics
    """
    
    ax.text(85, 50, methodology_text, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#e8f4fd', alpha=0.9,
                     edgecolor='#34495e', linewidth=2))
    
    # Add outputs box
    outputs_text = """
    📁 OUTPUTS:
    
    • enhanced_radiomics_similarity_analysis.png
    • enhanced_radiomics_clusters.csv
    • enhanced_high_correlation_pairs.csv
    • enhanced_modality_similarities.csv
    • enhanced_radiomics_similarity_summary.txt
    • enhanced_radiomics_similarity_report.pdf
    """
    
    ax.text(85, 20, outputs_text, ha='left', va='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f8ff', alpha=0.9,
                     edgecolor='#34495e', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('radiomics_pipeline_visualization.png', dpi=300, bbox_inches='tight',
                facecolor='#f8f9fa')
    plt.close()
    
    print("Pipeline visualization saved to: radiomics_pipeline_visualization.png")

def create_detailed_workflow_diagram():
    """Create a detailed workflow diagram with step-by-step process"""
    
    fig, ax = plt.subplots(1, 1, figsize=(24, 18))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define workflow steps
    workflow_steps = [
        {
            'step': 1,
            'title': 'Data Collection',
            'description': 'Load radiomics data from 2020-2022\nExtract patient-level features\nIdentify modality-specific features',
            'x': 10,
            'y': 90,
            'color': '#FF6B6B'
        },
        {
            'step': 2,
            'title': 'Data Preprocessing',
            'description': 'Handle missing values\nStandardize features\nDetect and handle outliers\nNormalize data',
            'x': 30,
            'y': 90,
            'color': '#4ECDC4'
        },
        {
            'step': 3,
            'title': 'Feature Analysis',
            'description': 'Calculate correlation matrix\nIdentify high correlations\nAnalyze feature pairs\nAssess modality relationships',
            'x': 50,
            'y': 90,
            'color': '#45B7D1'
        },
        {
            'step': 4,
            'title': 'Clustering Analysis',
            'description': 'Apply K-means clustering\nPerform silhouette analysis\nDetermine optimal k\nValidate clusters',
            'x': 70,
            'y': 90,
            'color': '#96CEB4'
        },
        {
            'step': 5,
            'title': 'Dimensionality Reduction',
            'description': 'Apply PCA (95% variance)\nPerform t-SNE analysis\nVisualize components\nAssess dimensionality',
            'x': 10,
            'y': 70,
            'color': '#FFEAA7'
        },
        {
            'step': 6,
            'title': 'Modality Analysis',
            'description': 'Group features by modality\nCalculate within-modality similarity\nAnalyze between-modality correlations\nAssess cross-modality features',
            'x': 30,
            'y': 70,
            'color': '#DDA0DD'
        },
        {
            'step': 7,
            'title': 'Temporal Analysis',
            'description': 'Analyze year-wise patterns\nAssess temporal consistency\nEvaluate feature stability\nIdentify temporal trends',
            'x': 50,
            'y': 70,
            'color': '#98D8C8'
        },
        {
            'step': 8,
            'title': 'Feature Importance',
            'description': 'Calculate cluster variance\nRank features by importance\nAssess feature stability\nSelect key features',
            'x': 70,
            'y': 70,
            'color': '#F7DC6F'
        },
        {
            'step': 9,
            'title': 'Quality Assessment',
            'description': 'Calculate silhouette scores\nAssess cluster quality\nValidate results\nPerform statistical tests',
            'x': 10,
            'y': 50,
            'color': '#BB8FCE'
        },
        {
            'step': 10,
            'title': 'Network Analysis',
            'description': 'Create feature networks\nAnalyze correlation networks\nAssess modality networks\nCalculate network metrics',
            'x': 30,
            'y': 50,
            'color': '#85C1E9'
        },
        {
            'step': 11,
            'title': 'Visualization',
            'description': 'Generate correlation heatmaps\nCreate cluster visualizations\nProduce PCA/t-SNE plots\nDesign modality matrices',
            'x': 50,
            'y': 50,
            'color': '#F8C471'
        },
        {
            'step': 12,
            'title': 'Report Generation',
            'description': 'Compile results\nCreate PDF report\nGenerate summary files\nDocument findings',
            'x': 70,
            'y': 50,
            'color': '#82E0AA'
        }
    ]
    
    # Draw workflow steps
    for step_info in workflow_steps:
        # Create step box
        box = FancyBboxPatch(
            (step_info['x'], step_info['y']), 
            18, 
            15,
            boxstyle="round,pad=0.2",
            facecolor=step_info['color'],
            edgecolor='black',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(box)
        
        # Add step number
        ax.text(step_info['x'] + 9, step_info['y'] + 13, 
                f"STEP {step_info['step']}", ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
        
        # Add title
        ax.text(step_info['x'] + 9, step_info['y'] + 10, 
                step_info['title'], ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
        
        # Add description
        ax.text(step_info['x'] + 1, step_info['y'] + 7, 
                step_info['description'], ha='left', va='top', 
                fontsize=8, color='white',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.6))
    
    # Add flow arrows
    for i in range(len(workflow_steps) - 1):
        current = workflow_steps[i]
        next_step = workflow_steps[i + 1]
        
        # Horizontal arrows
        if current['y'] == next_step['y']:
            start = (current['x'] + 18, current['y'] + 7.5)
            end = (next_step['x'], next_step['y'] + 7.5)
        # Vertical arrows
        else:
            start = (current['x'] + 9, current['y'])
            end = (next_step['x'] + 9, next_step['y'] + 15)
        
        arrow = ConnectionPatch(start, end, "data", "data",
                              arrowstyle="->", shrinkA=5, shrinkB=5,
                              mutation_scale=25, fc="black", ec="black", linewidth=3)
        ax.add_patch(arrow)
    
    # Add main title
    ax.text(50, 97, 'DETAILED RADIOMICS SIMILARITY ANALYSIS WORKFLOW', 
            ha='center', va='center', fontsize=28, fontweight='bold',
            color='#2c3e50', bbox=dict(boxstyle="round,pad=0.5", facecolor='#ecf0f1', alpha=0.9))
    
    # Add subtitle
    ax.text(50, 94, 'Step-by-Step Process for Comprehensive Feature Similarity Analysis', 
            ha='center', va='center', fontsize=16, color='#34495e',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#bdc3c7', alpha=0.7))
    
    # Add methodology summary
    methodology_text = """
    🔬 ANALYSIS METHODOLOGY:
    
    • Unsupervised Learning Approach
    • K-means Clustering with Silhouette Analysis
    • Principal Component Analysis (PCA)
    • t-Distributed Stochastic Neighbor Embedding (t-SNE)
    • Pearson Correlation Analysis
    • Network Analysis and Visualization
    • Statistical Quality Assessment
    • Comprehensive Reporting
    """
    
    ax.text(85, 85, methodology_text, ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#e8f4fd', alpha=0.9,
                     edgecolor='#34495e', linewidth=2))
    
    # Add key outputs
    outputs_text = """
    📊 KEY OUTPUTS:
    
    • 20-Panel Visualization Dashboard
    • Patient Cluster Assignments
    • High Correlation Feature Pairs
    • Modality Similarity Matrix
    • Comprehensive Analysis Summary
    • Professional PDF Report (6 pages)
    • Statistical Quality Metrics
    • Network Analysis Results
    """
    
    ax.text(85, 55, outputs_text, ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f0f8ff', alpha=0.9,
                     edgecolor='#34495e', linewidth=2))
    
    # Add data summary
    data_text = """
    📈 DATA SUMMARY:
    
    • 150 Patients (Synthetic Data)
    • 82 Radiomics Features
    • 6 Modalities (T1, DWI, ADC, FLAIR, T2, Cross-Modality)
    • 3 Years (2020-2022)
    • 3 Optimal Clusters
    • 10+ High Correlation Pairs
    • 95% Variance Explained (PCA)
    • 0.52 Silhouette Score
    """
    
    ax.text(85, 25, data_text, ha='left', va='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f1f2f6', alpha=0.9,
                     edgecolor='#34495e', linewidth=2))
    
    plt.tight_layout()
    plt.savefig('detailed_radiomics_workflow.png', dpi=300, bbox_inches='tight',
                facecolor='#f8f9fa')
    plt.close()
    
    print("Detailed workflow diagram saved to: detailed_radiomics_workflow.png")

def main():
    """Create comprehensive pipeline visualizations"""
    
    print("Creating comprehensive radiomics pipeline visualizations...")
    
    # Create main pipeline visualization
    print("Creating main pipeline visualization...")
    create_pipeline_visualization()
    
    # Create detailed workflow diagram
    print("Creating detailed workflow diagram...")
    create_detailed_workflow_diagram()
    
    print("\n=== COMPREHENSIVE PIPELINE VISUALIZATIONS CREATED SUCCESSFULLY ===")
    print("Files generated:")
    print("- radiomics_pipeline_visualization.png (Main pipeline overview)")
    print("- detailed_radiomics_workflow.png (Step-by-step workflow)")
    print("\nFeatures:")
    print("- Professional color-coded stages")
    print("- Properly aligned text and descriptions")
    print("- Flow arrows showing process direction")
    print("- Comprehensive statistics and methodology")
    print("- Enhanced aesthetics with professional styling")

if __name__ == "__main__":
    main() 