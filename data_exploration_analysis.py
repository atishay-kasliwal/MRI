#!/usr/bin/env python3
"""
Data Exploration and Analysis Suggestions
Comprehensive analysis of the merged radiomics-clinical dataset to suggest research opportunities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def explore_dataset_and_suggest_analyses():
    """Explore the dataset and suggest research opportunities"""
    
    print("=== DATA EXPLORATION AND RESEARCH SUGGESTIONS ===\n")
    
    # Load the merged dataset
    df = pd.read_csv("merged_radiomics_clinical_data.csv")
    
    print("📊 DATASET OVERVIEW")
    print(f"  Patients: {len(df)}")
    print(f"  Features: {len(df.columns)}")
    print(f"  Years: {sorted(df['Year'].unique())}")
    
    # Identify feature categories
    radiomics_cols = [col for col in df.columns if any(mod in col for mod in ['T1_', 'T2_', 'FLAIR_', 'DWI_', 'ADC_'])]
    clinical_cols = [col for col in df.columns if col not in radiomics_cols and col not in ['PatientID', 'Year', 'AvailableModalities']]
    mrs_cols = [col for col in df.columns if 'mrs' in col.lower()]
    
    print(f"\n🔬 FEATURE BREAKDOWN")
    print(f"  Radiomics: {len(radiomics_cols)} features")
    print(f"  Clinical: {len(clinical_cols)} features")
    print(f"  mRS scores: {len(mrs_cols)} features")
    
    # Analyze mRS data
    print(f"\n🏥 mRS ANALYSIS")
    for col in mrs_cols:
        valid_mrs = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(valid_mrs) > 0:
            print(f"  {col}: {len(valid_mrs)} patients, range: {valid_mrs.min()}-{valid_mrs.max()}")
    
    # Analyze clinical variables
    print(f"\n💊 CLINICAL VARIABLES ANALYSIS")
    key_clinical = ['Sex', 'Age', 'Diabetes', 'Hypertension', 'AFIB', 'Prior Stroke', 'Smoking hx']
    for var in key_clinical:
        if var in df.columns:
            valid_values = df[var].dropna()
            if len(valid_values) > 0:
                if var == 'Age':
                    age_numeric = pd.to_numeric(valid_values, errors='coerce').dropna()
                    print(f"  {var}: {len(age_numeric)} patients, mean: {age_numeric.mean():.1f}±{age_numeric.std():.1f}")
                else:
                    print(f"  {var}: {len(valid_values)} patients")
    
    # Analyze radiomics features
    print(f"\n🔬 RADIOMICS FEATURE ANALYSIS")
    modality_counts = {}
    for col in radiomics_cols:
        for modality in ['T1', 'T2', 'FLAIR', 'DWI', 'ADC']:
            if col.startswith(f"{modality}_"):
                modality_counts[modality] = modality_counts.get(modality, 0) + 1
                break
    
    for modality, count in modality_counts.items():
        print(f"  {modality}: {count} features")
    
    # Suggest research opportunities
    print(f"\n" + "="*60)
    print("🎯 RESEARCH OPPORTUNITIES AND ANALYSIS SUGGESTIONS")
    print("="*60)
    
    # 1. mRS Prediction Models
    print(f"\n1️⃣ MRS PREDICTION MODELS")
    print("   Primary Outcome: Predict mRS scores using radiomics + clinical features")
    print("   Opportunities:")
    print("   • Predict 90-day mRS from baseline radiomics")
    print("   • Predict Last mRS from discharge radiomics")
    print("   • Binary classification: Good outcome (mRS 0-2) vs Poor outcome (mRS 3-5)")
    print("   • Multi-class classification: All mRS levels (0-5)")
    print("   • Survival analysis: Time to mRS improvement")
    
    # 2. Radiomics-Clinical Correlations
    print(f"\n2️⃣ RADIOMICS-CLINICAL CORRELATIONS")
    print("   Explore relationships between imaging features and clinical variables")
    print("   Opportunities:")
    print("   • Age-related radiomics changes")
    print("   • Sex differences in radiomics features")
    print("   • Comorbidity effects on radiomics (Diabetes, Hypertension, etc.)")
    print("   • Smoking history impact on imaging features")
    
    # 3. Multi-modal Radiomics Analysis
    print(f"\n3️⃣ MULTI-MODAL RADIOMICS ANALYSIS")
    print("   Compare and combine features across 5 MRI modalities")
    print("   Opportunities:")
    print("   • Modality-specific feature importance")
    print("   • Feature fusion strategies")
    print("   • Cross-modal correlations")
    print("   • Optimal modality combinations for prediction")
    
    # 4. Temporal Analysis
    print(f"\n4️⃣ TEMPORAL ANALYSIS (2020-2024)")
    print("   Analyze trends and changes over 5 years")
    print("   Opportunities:")
    print("   • Year-to-year radiomics variations")
    print("   • Treatment protocol changes impact")
    print("   • Patient population evolution")
    print("   • Seasonal effects on outcomes")
    
    # 5. Feature Selection and Dimensionality Reduction
    print(f"\n5️⃣ FEATURE SELECTION AND DIMENSIONALITY REDUCTION")
    print("   Handle high-dimensional radiomics data")
    print("   Opportunities:")
    print("   • LASSO/Ridge regression for feature selection")
    print("   • Principal Component Analysis (PCA)")
    print("   • Recursive Feature Elimination (RFE)")
    print("   • SHAP values for feature importance")
    
    # 6. Subgroup Analysis
    print(f"\n6️⃣ SUBGROUP ANALYSIS")
    print("   Analyze specific patient populations")
    print("   Opportunities:")
    print("   • Age groups (young vs elderly)")
    print("   • Sex-specific models")
    print("   • Comorbidity subgroups")
    print("   • Severity-based analysis (mRS 0-1 vs 2-3 vs 4-5)")
    
    # 7. Validation and Generalization
    print(f"\n7️⃣ VALIDATION AND GENERALIZATION")
    print("   Ensure model robustness and clinical applicability")
    print("   Opportunities:")
    print("   • Cross-validation strategies")
    print("   • External validation on new data")
    print("   • Model interpretability")
    print("   • Clinical decision support system development")
    
    # 8. Advanced Machine Learning
    print(f"\n8️⃣ ADVANCED MACHINE LEARNING APPROACHES")
    print("   Leverage modern ML techniques")
    print("   Opportunities:")
    print("   • Ensemble methods (Random Forest, XGBoost, CatBoost)")
    print("   • Deep learning for radiomics")
    print("   • Transfer learning from pre-trained models")
    print("   • Multi-task learning (predict multiple outcomes)")
    
    # 9. Clinical Translation
    print(f"\n9️⃣ CLINICAL TRANSLATION")
    print("   Bridge the gap between research and clinical practice")
    print("   Opportunities:")
    print("   • Risk stratification models")
    print("   • Treatment response prediction")
    print("   • Prognostic biomarker identification")
    print("   • Personalized medicine approaches")
    
    # 10. Comparative Studies
    print(f"\n🔟 COMPARATIVE STUDIES")
    print("   Compare with existing literature and methods")
    print("   Opportunities:")
    print("   • Compare with traditional clinical scores")
    print("   • Benchmark against published radiomics studies")
    print("   • Multi-center validation")
    print("   • Meta-analysis integration")
    
    # Specific analysis suggestions based on data availability
    print(f"\n" + "="*60)
    print("📋 SPECIFIC ANALYSIS RECOMMENDATIONS")
    print("="*60)
    
    # Check data availability for specific analyses
    print(f"\n✅ IMMEDIATE OPPORTUNITIES (High data availability):")
    
    # 90-day mRS prediction
    valid_90day = pd.to_numeric(df['90 days mRS'], errors='coerce').dropna()
    if len(valid_90day) > 100:
        print(f"   • 90-day mRS prediction: {len(valid_90day)} patients available")
        print(f"     - Good outcome (mRS 0-2): {len(valid_90day[valid_90day <= 2])} patients")
        print(f"     - Poor outcome (mRS 3-5): {len(valid_90day[valid_90day >= 3])} patients")
    
    # Last mRS prediction
    valid_last = pd.to_numeric(df['Last mRS'], errors='coerce').dropna()
    if len(valid_last) > 100:
        print(f"   • Last mRS prediction: {len(valid_last)} patients available")
        print(f"     - Good outcome (mRS 0-2): {len(valid_last[valid_last <= 2])} patients")
        print(f"     - Poor outcome (mRS 3-5): {len(valid_last[valid_last >= 3])} patients")
    
    # Age analysis
    if 'Age' in df.columns:
        age_data = pd.to_numeric(df['Age'], errors='coerce').dropna()
        if len(age_data) > 100:
            print(f"   • Age-based analysis: {len(age_data)} patients")
            print(f"     - Young (<65): {len(age_data[age_data < 65])} patients")
            print(f"     - Elderly (≥65): {len(age_data[age_data >= 65])} patients")
    
    print(f"\n⚠️  MODERATE OPPORTUNITIES (Limited data):")
    
    # Baseline mRS prediction
    valid_baseline = pd.to_numeric(df['Baseline mRS'], errors='coerce').dropna()
    if len(valid_baseline) < 50:
        print(f"   • Baseline mRS prediction: Only {len(valid_baseline)} patients available")
    
    # Discharge mRS prediction
    valid_discharge = pd.to_numeric(df['Discharge mRS'], errors='coerce').dropna()
    if len(valid_discharge) < 30:
        print(f"   • Discharge mRS prediction: Only {len(valid_discharge)} patients available")
    
    print(f"\n🎯 RECOMMENDED STARTING POINTS:")
    print(f"   1. Start with 90-day mRS prediction (largest sample size)")
    print(f"   2. Focus on binary classification (Good vs Poor outcome)")
    print(f"   3. Use all 5 modalities for comprehensive analysis")
    print(f"   4. Include key clinical variables (Age, Sex, comorbidities)")
    print(f"   5. Implement cross-validation for robust results")
    
    print(f"\n📈 NEXT STEPS:")
    print(f"   1. Create feature importance analysis")
    print(f"   2. Build baseline prediction models")
    print(f"   3. Compare different ML algorithms")
    print(f"   4. Validate results with clinical experts")
    print(f"   5. Prepare for publication")
    
    return df

if __name__ == "__main__":
    explore_dataset_and_suggest_analyses() 