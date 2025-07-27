# Advanced Radiomics Research Pipeline

A comprehensive, state-of-the-art radiomics analysis pipeline for medical imaging research, featuring advanced machine learning, deep learning, and statistical analysis techniques.

## 🚀 Features

### Core Modules
1. **Radiogenomics Analysis** - Correlate radiomics features with genomics data
2. **Deep Radiomics** - Deep learning feature extraction and fusion
3. **Survival Analysis** - Time-to-event analysis with Cox models and Kaplan-Meier curves
4. **Radiomics Signature Development** - Feature selection, nomogram creation, risk stratification
5. **Explainable AI** - Model interpretability with SHAP, LIME, and feature importance
6. **Multi-Modal Integration** - Combine radiomics, clinical, and genomics data
7. **Clustering & Phenotyping** - Unsupervised learning for phenotype discovery
8. **Longitudinal & Delta Radiomics** - Feature change over time analysis
9. **Feature Robustness & Harmonization** - Multi-center data harmonization
10. **Clinical Trial Integration** - Stratification and enrichment strategies

### Advanced Capabilities
- **Synthetic Data Generation** - Realistic simulation for testing and validation
- **Golden Theme Visualization** - Professional, publication-ready plots
- **Comprehensive PDF Reports** - Multi-page analysis reports
- **Modular Architecture** - Easy to extend and customize
- **Cross-Validation** - Robust model evaluation
- **Feature Engineering** - Advanced feature selection and engineering

## 📁 Project Structure

```
mri/
├── create_complete_radiomics_research_pipeline.py  # Main pipeline script
├── create_advanced_radiomics_analysis.py          # Advanced analysis modules
├── create_advanced_radiomics_visualizations.py    # Visualization utilities
├── create_comprehensive_radiomics_pipeline.py     # Comprehensive pipeline
├── create_real_data_analysis.py                   # Real data analysis
├── results/                                       # Data and results
│   ├── radiomics_2020_only.csv
│   ├── radiomics_2022_only.csv
│   ├── mrs_2020_patients.csv
│   ├── mrs_2021_patients.csv
│   └── mrs_2022_patients.csv
├── scripts/                                       # Utility scripts
└── notebooks/                                     # Jupyter notebooks
```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd mri
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lifelines
pip install xgboost lightgbm catboost  # Optional: for advanced models
pip install umap-learn  # Optional: for UMAP dimensionality reduction
```

## 📊 Usage

### Basic Usage
```python
from create_complete_radiomics_research_pipeline import CompleteRadiomicsResearchPipeline

# Initialize pipeline
pipeline = CompleteRadiomicsResearchPipeline()

# Run complete analysis
pipeline.run_pipeline('radiomics_research_report.pdf')
```

### Individual Modules
```python
# Run specific modules
pipeline.section_radiogenomics(pdf)
pipeline.section_deep_radiomics(pdf)
pipeline.section_survival_analysis(pdf)
```

### Custom Data Integration
```python
# Load your own data
pipeline.data = your_radiomics_data
pipeline.genomics = your_genomics_data
pipeline.run_pipeline('custom_analysis.pdf')
```

## 🔬 Key Analysis Modules

### 1. Radiogenomics Analysis
- Correlation analysis between radiomics and genomics features
- Gene mutation frequency analysis
- Outcome impact by gene status
- Combined model performance comparison

### 2. Deep Radiomics
- Simulated deep learning feature extraction
- Feature space comparison (PCA, t-SNE)
- Feature fusion performance analysis
- Feature importance in combined models

### 3. Survival Analysis
- Kaplan-Meier survival curves
- Cox Proportional Hazards models
- Risk stratification
- Time-dependent analysis

### 4. Radiomics Signature Development
- LASSO feature selection
- Signature performance validation
- Nomogram creation
- Risk group stratification

### 5. Explainable AI
- Multi-model feature importance comparison
- Permutation importance analysis
- Partial dependence plots
- Prediction confidence analysis

### 6. Multi-Modal Integration
- Cross-modality performance comparison
- Feature correlation analysis
- Combined model optimization
- Incremental value assessment

### 7. Clustering & Phenotyping
- UMAP/t-SNE dimensionality reduction
- K-means clustering
- Phenotype discovery
- Cluster-outcome associations

## 📈 Output

The pipeline generates comprehensive PDF reports containing:
- **Executive Summary** - Overview of all analyses
- **Data Overview** - Dataset characteristics and quality
- **Module-specific Analysis** - Detailed results for each module
- **Visualizations** - Publication-ready figures and plots
- **Statistical Results** - Performance metrics and significance tests

## 🎨 Visualization Features

- **Golden Theme** - Professional color scheme
- **Publication Quality** - High-resolution plots
- **Modular Design** - Easy to customize
- **Multi-page Reports** - Comprehensive documentation

## 🔧 Customization

### Adding New Modules
```python
def section_custom_analysis(self, pdf):
    """Custom analysis module"""
    # Your analysis code here
    pass
```

### Custom Visualizations
```python
# Use the golden theme colors
GOLDEN_COLORS = {
    'primary_gold': '#B8860B',
    'secondary_gold': '#DAA520',
    # ... more colors
}
```

### Data Integration
```python
# Load custom data
pipeline.data = pd.read_csv('your_radiomics_data.csv')
pipeline.genomics = pd.read_csv('your_genomics_data.csv')
```

## 📚 Dependencies

### Core Dependencies
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning
- `lifelines` - Survival analysis

### Optional Dependencies
- `xgboost` - Gradient boosting
- `lightgbm` - Light gradient boosting
- `catboost` - Categorical boosting
- `umap-learn` - UMAP dimensionality reduction
- `shap` - Model interpretability
- `lime` - Local interpretability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and support:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🔬 Research Applications

This pipeline is designed for:
- **Medical Imaging Research** - Radiomics analysis
- **Clinical Studies** - Patient outcome prediction
- **Biomarker Discovery** - Feature identification
- **Clinical Trials** - Stratification and enrichment
- **Precision Medicine** - Personalized treatment strategies

## 📊 Performance

The pipeline has been tested with:
- **200+ patients** - Synthetic data validation
- **30+ radiomics features** - Comprehensive feature analysis
- **15+ genomic features** - Multi-modal integration
- **Multiple timepoints** - Longitudinal analysis

---

**Note:** This pipeline uses synthetic data for demonstration. For real clinical applications, ensure proper data handling and regulatory compliance. 