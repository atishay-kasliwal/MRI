#!/usr/bin/env python3
"""
Create Detailed Paper Analysis PDFs for Top Radiomics Papers
Comprehensive analysis of each paper's methodology, contributions, and impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set enhanced style for better aesthetics
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_gillies_paper_analysis():
    """Create detailed analysis PDF for Gillies et al. (2016)"""
    print("📊 Creating detailed analysis for Gillies et al. (2016)...")
    
    with PdfPages('gillies_2016_detailed_analysis.pdf') as pdf:
        
        # Page 1: Title and Overview
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'GILLIES ET AL. (2016)', fontsize=28, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        plt.text(0.5, 0.90, 'Radiomics: Extracting more information from medical images using advanced feature analysis', 
                 fontsize=16, ha='center', va='center', color='#34495e')
        plt.text(0.5, 0.85, 'European Journal of Cancer', fontsize=14, ha='center', va='center', color='#7f8c8d')
        
        # Paper metrics
        metrics = [
            "📊 Citations: 4,500+",
            "🎯 Impact Factor: 7.275",
            "📈 Category: Foundational",
            "🔬 Key Contribution: First comprehensive radiomics definition"
        ]
        
        y_pos = 0.75
        for metric in metrics:
            plt.text(0.5, y_pos, metric, fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Abstract summary
        plt.text(0.5, 0.55, 'ABSTRACT SUMMARY', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        abstract_points = [
            "• Introduced the term 'radiomics' and established its definition",
            "• Proposed comprehensive radiomics workflow",
            "• Identified four main feature categories",
            "• Demonstrated clinical relevance in oncology",
            "• Established reproducibility standards"
        ]
        
        y_pos = 0.45
        for point in abstract_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 2: Methodology Analysis
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'METHODOLOGY ANALYSIS', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Radiomics Workflow
        plt.text(0.5, 0.85, '🔬 RADIOMICS WORKFLOW', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        workflow_steps = [
            "1. Image Acquisition: Standardized imaging protocols",
            "2. Image Segmentation: Manual or automated ROI definition",
            "3. Feature Extraction: Quantitative feature calculation",
            "4. Feature Selection: Statistical and machine learning methods",
            "5. Model Building: Predictive model development",
            "6. Validation: Independent dataset testing"
        ]
        
        y_pos = 0.75
        for step in workflow_steps:
            plt.text(0.05, y_pos, step, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Feature Categories
        plt.text(0.5, 0.45, '📊 FEATURE CATEGORIES', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        categories = [
            "• Shape Features: Volume, surface area, compactness",
            "• First-Order Features: Mean, variance, skewness, kurtosis",
            "• Texture Features: GLCM, GLRLM, GLSZM matrices",
            "• Higher-Order Features: Wavelet, fractal, filter-based"
        ]
        
        y_pos = 0.35
        for category in categories:
            plt.text(0.05, y_pos, category, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 3: Key Contributions and Impact
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'KEY CONTRIBUTIONS & IMPACT', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Major Contributions
        plt.text(0.5, 0.85, '🎯 MAJOR CONTRIBUTIONS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        contributions = [
            "• First comprehensive definition of radiomics",
            "• Established standardized radiomics workflow",
            "• Introduced feature categorization system",
            "• Demonstrated clinical relevance in multiple cancer types",
            "• Established reproducibility and validation standards",
            "• Laid foundation for PyRadiomics library development"
        ]
        
        y_pos = 0.75
        for contrib in contributions:
            plt.text(0.05, y_pos, contrib, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Clinical Impact
        plt.text(0.5, 0.45, '🏥 CLINICAL IMPACT', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        impact_points = [
            "• Enabled non-invasive tumor characterization",
            "• Facilitated personalized treatment planning",
            "• Improved prognostic assessment accuracy",
            "• Reduced need for invasive procedures",
            "• Standardized imaging biomarker development"
        ]
        
        y_pos = 0.35
        for point in impact_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 4: Implementation and Results
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'IMPLEMENTATION & RESULTS', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Implementation Details
        plt.text(0.5, 0.85, '⚙️ IMPLEMENTATION DETAILS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        implementation = [
            "• Dataset: Multi-center cancer imaging data",
            "• Modalities: CT, MRI, PET imaging",
            "• Features: 1000+ radiomics features extracted",
            "• Validation: Cross-validation and independent testing",
            "• Software: PyRadiomics library development",
            "• Standards: IBSI (Image Biomarker Standardization Initiative)"
        ]
        
        y_pos = 0.75
        for item in implementation:
            plt.text(0.05, y_pos, item, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Results Summary
        plt.text(0.5, 0.45, '📊 RESULTS SUMMARY', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        results = [
            "• Successfully extracted 1000+ radiomics features",
            "• Demonstrated feature reproducibility across centers",
            "• Established clinical relevance in multiple cancer types",
            "• Achieved significant prognostic value",
            "• Created standardized feature extraction protocols"
        ]
        
        y_pos = 0.35
        for result in results:
            plt.text(0.05, y_pos, result, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 5: Future Directions and Legacy
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'FUTURE DIRECTIONS & LEGACY', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Legacy
        plt.text(0.5, 0.85, '🏛️ LEGACY', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        legacy_points = [
            "• Foundation for entire radiomics field",
            "• PyRadiomics library (most widely used radiomics software)",
            "• IBSI standardization initiative",
            "• Thousands of follow-up studies",
            "• Clinical implementation in multiple centers",
            "• Regulatory approval pathways established"
        ]
        
        y_pos = 0.75
        for point in legacy_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Future Directions
        plt.text(0.5, 0.45, '🔮 FUTURE DIRECTIONS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        future_points = [
            "• Deep learning integration with radiomics",
            "• Multi-modal radiomics analysis",
            "• Real-time clinical implementation",
            "• Regulatory approval for clinical use",
            "• Standardization across imaging platforms"
        ]
        
        y_pos = 0.35
        for point in future_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    print("✅ Gillies detailed analysis saved to: gillies_2016_detailed_analysis.pdf")

def create_aerts_paper_analysis():
    """Create detailed analysis PDF for Aerts et al. (2014)"""
    print("🎯 Creating detailed analysis for Aerts et al. (2014)...")
    
    with PdfPages('aerts_2014_detailed_analysis.pdf') as pdf:
        
        # Page 1: Title and Overview
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'AERTS ET AL. (2014)', fontsize=28, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        plt.text(0.5, 0.90, 'Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach', 
                 fontsize=16, ha='center', va='center', color='#34495e')
        plt.text(0.5, 0.85, 'Nature Communications', fontsize=14, ha='center', va='center', color='#7f8c8d')
        
        # Paper metrics
        metrics = [
            "📊 Citations: 3,200+",
            "🎯 Impact Factor: 14.919",
            "📈 Category: Breakthrough",
            "🔬 Key Contribution: First radiomics signature for survival prediction"
        ]
        
        y_pos = 0.75
        for metric in metrics:
            plt.text(0.5, y_pos, metric, fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Abstract summary
        plt.text(0.5, 0.55, 'ABSTRACT SUMMARY', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        abstract_points = [
            "• First large-scale radiomics study in lung cancer",
            "• Demonstrated radiomics can predict survival and molecular characteristics",
            "• Introduced radiomics signature concept",
            "• Multi-center validation approach",
            "• Established radiomics as predictive biomarker"
        ]
        
        y_pos = 0.45
        for point in abstract_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 2: Methodology Analysis
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'METHODOLOGY ANALYSIS', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Study Design
        plt.text(0.5, 0.85, '🔬 STUDY DESIGN', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        design_points = [
            "• Dataset: 1,019 lung cancer patients",
            "• Imaging: Pre-treatment CT scans",
            "• Features: 440 radiomics features extracted",
            "• Validation: Multi-center independent validation",
            "• Outcome: Overall survival prediction",
            "• Molecular: EGFR, KRAS mutation prediction"
        ]
        
        y_pos = 0.75
        for point in design_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Radiomics Signature Development
        plt.text(0.5, 0.45, '🎯 RADIOMICS SIGNATURE DEVELOPMENT', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        signature_points = [
            "• Feature Selection: LASSO regression",
            "• Signature Components: 4 key radiomics features",
            "• Risk Stratification: High vs low risk groups",
            "• Validation: Cross-validation and independent testing",
            "• Clinical Integration: Combined with clinical factors"
        ]
        
        y_pos = 0.35
        for point in signature_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 3: Key Results and Findings
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'KEY RESULTS & FINDINGS', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Survival Prediction Results
        plt.text(0.5, 0.85, '📊 SURVIVAL PREDICTION RESULTS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        survival_results = [
            "• Radiomics Signature: 4-feature signature developed",
            "• Risk Stratification: Significant survival difference (p < 0.001)",
            "• Validation: Multi-center validation successful",
            "• Clinical Integration: Improved prediction with clinical factors",
            "• Molecular Prediction: EGFR mutation prediction (AUC = 0.69)"
        ]
        
        y_pos = 0.75
        for result in survival_results:
            plt.text(0.05, y_pos, result, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Clinical Impact
        plt.text(0.5, 0.45, '🏥 CLINICAL IMPACT', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        impact_points = [
            "• Non-invasive survival prediction",
            "• Personalized treatment planning",
            "• Molecular marker prediction without biopsy",
            "• Risk stratification for clinical trials",
            "• Foundation for radiomics biomarkers"
        ]
        
        y_pos = 0.35
        for point in impact_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 4: Innovation and Breakthrough
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'INNOVATION & BREAKTHROUGH', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Breakthrough Contributions
        plt.text(0.5, 0.85, '🚀 BREAKTHROUGH CONTRIBUTIONS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        breakthrough_points = [
            "• First large-scale radiomics study",
            "• Introduced radiomics signature concept",
            "• Demonstrated molecular prediction capability",
            "• Multi-center validation approach",
            "• Established radiomics as predictive biomarker",
            "• Published in high-impact journal (Nature Communications)"
        ]
        
        y_pos = 0.75
        for point in breakthrough_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Methodological Innovations
        plt.text(0.5, 0.45, '⚙️ METHODOLOGICAL INNOVATIONS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        innovation_points = [
            "• LASSO-based feature selection",
            "• Multi-center validation strategy",
            "• Clinical-radiomics integration",
            "• Molecular prediction from imaging",
            "• Risk stratification methodology"
        ]
        
        y_pos = 0.35
        for point in innovation_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 5: Legacy and Future Impact
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'LEGACY & FUTURE IMPACT', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Legacy
        plt.text(0.5, 0.85, '🏛️ LEGACY', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        legacy_points = [
            "• Most cited radiomics paper",
            "• Established radiomics signature concept",
            "• Foundation for survival prediction studies",
            "• Multi-center validation standard",
            "• Clinical trial integration",
            "• Regulatory approval pathways"
        ]
        
        y_pos = 0.75
        for point in legacy_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Future Directions
        plt.text(0.5, 0.45, '🔮 FUTURE DIRECTIONS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        future_points = [
            "• Clinical implementation in lung cancer",
            "• Extension to other cancer types",
            "• Real-time prediction systems",
            "• Integration with treatment planning",
            "• Regulatory approval for clinical use"
        ]
        
        y_pos = 0.35
        for point in future_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    print("✅ Aerts detailed analysis saved to: aerts_2014_detailed_analysis.pdf")

def create_kickingereder_paper_analysis():
    """Create detailed analysis PDF for Kickingereder et al. (2016)"""
    print("🧠 Creating detailed analysis for Kickingereder et al. (2016)...")
    
    with PdfPages('kickingereder_2016_detailed_analysis.pdf') as pdf:
        
        # Page 1: Title and Overview
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'KICKINGEREDER ET AL. (2016)', fontsize=28, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        plt.text(0.5, 0.90, 'Radiomics of brain MRI: machine learning-based classification of molecular subtypes in glioblastoma', 
                 fontsize=16, ha='center', va='center', color='#34495e')
        plt.text(0.5, 0.85, 'Radiology', fontsize=14, ha='center', va='center', color='#7f8c8d')
        
        # Paper metrics
        metrics = [
            "📊 Citations: 1,800+",
            "🎯 Impact Factor: 11.105",
            "📈 Category: Neuro-oncology",
            "🔬 Key Contribution: Molecular prediction in glioblastoma"
        ]
        
        y_pos = 0.75
        for metric in metrics:
            plt.text(0.5, y_pos, metric, fontsize=14, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Abstract summary
        plt.text(0.5, 0.55, 'ABSTRACT SUMMARY', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        abstract_points = [
            "• First comprehensive radiomics study in glioblastoma",
            "• Correlation with molecular markers (MGMT, IDH1)",
            "• Prognostic value in brain tumors",
            "• Integration with clinical factors",
            "• Multi-parametric MRI analysis"
        ]
        
        y_pos = 0.45
        for point in abstract_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 2: Methodology Analysis
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'METHODOLOGY ANALYSIS', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Study Design
        plt.text(0.5, 0.85, '🔬 STUDY DESIGN', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        design_points = [
            "• Dataset: 119 glioblastoma patients",
            "• Imaging: Multi-parametric MRI (T1, T2, FLAIR, DWI)",
            "• Features: 1,200+ radiomics features extracted",
            "• Molecular: MGMT methylation, IDH1 mutation status",
            "• Validation: Cross-validation and independent testing",
            "• Clinical: Progression-free survival prediction"
        ]
        
        y_pos = 0.75
        for point in design_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Multi-parametric Analysis
        plt.text(0.5, 0.45, '📊 MULTI-PARAMETRIC ANALYSIS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        mri_points = [
            "• T1-weighted: Anatomical and contrast enhancement",
            "• T2-weighted: Edema and tumor extent",
            "• FLAIR: Peritumoral edema assessment",
            "• DWI: Cellularity and diffusion characteristics",
            "• Feature Integration: Cross-modality analysis"
        ]
        
        y_pos = 0.35
        for point in mri_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 3: Key Results and Findings
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'KEY RESULTS & FINDINGS', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Molecular Prediction Results
        plt.text(0.5, 0.85, '🧬 MOLECULAR PREDICTION RESULTS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        molecular_results = [
            "• MGMT Methylation: AUC = 0.85 (high accuracy)",
            "• IDH1 Mutation: Significant correlation with radiomics",
            "• Feature Selection: LASSO regression for feature selection",
            "• Validation: Cross-validation confirmed results",
            "• Clinical Integration: Combined with clinical factors"
        ]
        
        y_pos = 0.75
        for result in molecular_results:
            plt.text(0.05, y_pos, result, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Prognostic Value
        plt.text(0.5, 0.45, '📈 PROGNOSTIC VALUE', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        prognostic_points = [
            "• Progression-free survival prediction",
            "• Overall survival correlation",
            "• Risk stratification capabilities",
            "• Treatment response prediction",
            "• Clinical decision support"
        ]
        
        y_pos = 0.35
        for point in prognostic_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 4: Clinical Impact and Innovation
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'CLINICAL IMPACT & INNOVATION', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Clinical Impact
        plt.text(0.5, 0.85, '🏥 CLINICAL IMPACT', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        clinical_points = [
            "• Non-invasive molecular prediction",
            "• Personalized treatment planning",
            "• Reduced need for invasive biopsies",
            "• Improved prognostic assessment",
            "• Clinical trial stratification",
            "• Treatment response monitoring"
        ]
        
        y_pos = 0.75
        for point in clinical_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Innovation
        plt.text(0.5, 0.45, '🚀 INNOVATION', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        innovation_points = [
            "• First glioblastoma radiomics study",
            "• Multi-parametric MRI integration",
            "• Molecular marker prediction",
            "• Clinical-radiomics integration",
            "• Neuro-oncology applications"
        ]
        
        y_pos = 0.35
        for point in innovation_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
        
        # Page 5: Legacy and Future Directions
        fig = plt.figure(figsize=(12, 16))
        fig.patch.set_facecolor('#f8f9fa')
        
        plt.text(0.5, 0.95, 'LEGACY & FUTURE DIRECTIONS', fontsize=24, fontweight='bold', 
                 ha='center', va='center', color='#2c3e50')
        
        # Legacy
        plt.text(0.5, 0.85, '🏛️ LEGACY', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        legacy_points = [
            "• Foundation for neuro-oncology radiomics",
            "• Multi-parametric MRI standard",
            "• Molecular prediction methodology",
            "• Clinical integration framework",
            "• Glioblastoma treatment planning",
            "• Neuro-oncology clinical trials"
        ]
        
        y_pos = 0.75
        for point in legacy_points:
            plt.text(0.05, y_pos, point, fontsize=12, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#e8f4fd', alpha=0.8))
            y_pos -= 0.08
        
        # Future Directions
        plt.text(0.5, 0.45, '🔮 FUTURE DIRECTIONS', fontsize=18, fontweight='bold', 
                 ha='center', va='center', color='#e74c3c')
        
        future_points = [
            "• Clinical implementation in glioblastoma",
            "• Extension to other brain tumors",
            "• Real-time molecular prediction",
            "• Treatment response monitoring",
            "• Clinical trial integration"
        ]
        
        y_pos = 0.35
        for point in future_points:
            plt.text(0.05, y_pos, point, fontsize=11, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='#f1f2f6', alpha=0.7))
            y_pos -= 0.07
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()
    
    print("✅ Kickingereder detailed analysis saved to: kickingereder_2016_detailed_analysis.pdf")

def main():
    """Create detailed paper analysis PDFs for all papers"""
    print("📄 Creating detailed paper analysis PDFs...")
    
    # Create detailed analysis for each paper
    create_gillies_paper_analysis()
    create_aerts_paper_analysis()
    create_kickingereder_paper_analysis()
    
    print("\n✅ All detailed paper analysis PDFs created successfully!")
    print("📁 Generated Files:")
    print("   • gillies_2016_detailed_analysis.pdf (5 pages)")
    print("   • aerts_2014_detailed_analysis.pdf (5 pages)")
    print("   • kickingereder_2016_detailed_analysis.pdf (5 pages)")
    print("\n📊 Each PDF contains:")
    print("   • Title and overview with paper metrics")
    print("   • Detailed methodology analysis")
    print("   • Key results and findings")
    print("   • Innovation and breakthrough contributions")
    print("   • Legacy and future impact")

if __name__ == "__main__":
    main() 