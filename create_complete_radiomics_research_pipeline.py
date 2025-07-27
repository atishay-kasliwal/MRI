import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set golden theme colors
GOLDEN_COLORS = {
    'primary_gold': '#B8860B',
    'secondary_gold': '#DAA520', 
    'light_gold': '#F4A460',
    'pale_gold': '#F5DEB3',
    'dark_gold': '#8B6914',
    'black': '#000000',
    'white': '#FFFFFF',
    'grey': '#808080',
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['axes.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['axes.edgecolor'] = GOLDEN_COLORS['dark_gold']
plt.rcParams['axes.labelcolor'] = GOLDEN_COLORS['black']
plt.rcParams['xtick.color'] = GOLDEN_COLORS['black']
plt.rcParams['ytick.color'] = GOLDEN_COLORS['black']
plt.rcParams['text.color'] = GOLDEN_COLORS['black']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

class CompleteRadiomicsResearchPipeline:
    def __init__(self):
        self.data = None
        self.genomics = None
        self.longitudinal = None
        self.multi_center = None
        self.scaler = StandardScaler()
        print("🚀 Initialized Complete Radiomics Research Pipeline")

    def generate_synthetic_data(self):
        """Generate comprehensive synthetic data for all modules"""
        print("📊 Generating synthetic data for all modules...")
        np.random.seed(42)
        
        # Parameters
        n_patients = 200
        n_radiomics = 30
        n_genes = 15
        n_timepoints = 3
        n_centers = 4
        
        # Generate radiomics features with realistic correlations
        radiomics_features = []
        for i in range(n_radiomics):
            if i < 10:  # Shape features
                feature = np.random.normal(0, 1, n_patients) + np.random.normal(0, 0.1, n_patients)
            elif i < 20:  # Texture features
                feature = np.random.normal(0, 1, n_patients) + np.random.normal(0, 0.2, n_patients)
            else:  # Intensity features
                feature = np.random.normal(0, 1, n_patients) + np.random.normal(0, 0.3, n_patients)
            radiomics_features.append(feature)
        
        X_radiomics = np.column_stack(radiomics_features)
        
        # Clinical variables
        age = np.random.normal(65, 12, n_patients)
        sex = np.random.choice([0, 1], n_patients, p=[0.6, 0.4])
        bmi = np.random.normal(28, 5, n_patients)
        
        # Genomics data (binary mutations)
        genomics = np.random.choice([0, 1], (n_patients, n_genes), p=[0.8, 0.2])
        
        # Longitudinal data (3 timepoints)
        longitudinal = np.zeros((n_patients, n_timepoints, n_radiomics))
        for t in range(n_timepoints):
            longitudinal[:, t, :] = X_radiomics + np.random.normal(0, 0.1 * (t + 1), (n_patients, n_radiomics))
        
        # Multi-center data
        centers = np.random.choice(range(n_centers), n_patients)
        
        # Outcomes
        # Binary outcome based on radiomics + clinical + genomics
        outcome_prob = (0.3 * np.mean(X_radiomics[:, :10], axis=1) + 
                       0.2 * (age - 65) / 12 + 
                       0.1 * sex + 
                       0.4 * np.sum(genomics[:, :5], axis=1) / 5)
        outcome = (outcome_prob > np.median(outcome_prob)).astype(int)
        
        # Survival data
        base_survival = np.random.exponential(24, n_patients)
        survival_time = base_survival * (1 + 0.5 * outcome + 0.3 * (age - 65) / 12)
        event = np.random.choice([0, 1], n_patients, p=[0.3, 0.7])
        
        # Create DataFrames
        self.data = pd.DataFrame(X_radiomics, 
                                columns=[f'radiomics_{i+1:02d}' for i in range(n_radiomics)])
        self.data['age'] = age
        self.data['sex'] = sex
        self.data['bmi'] = bmi
        self.data['center'] = centers
        self.data['outcome'] = outcome
        self.data['survival_time'] = survival_time
        self.data['event'] = event
        
        self.genomics = pd.DataFrame(genomics, 
                                   columns=[f'gene_{i+1:02d}' for i in range(n_genes)])
        self.longitudinal = longitudinal
        self.multi_center = centers
        
        print(f"✅ Generated data: {n_patients} patients, {n_radiomics} radiomics features, {n_genes} genes")

    def run_pipeline(self, output_pdf='complete_radiomics_research_report.pdf'):
        """Run the complete pipeline"""
        self.generate_synthetic_data()
        
        with PdfPages(output_pdf) as pdf:
            self.section_intro(pdf)
            self.section_radiogenomics(pdf)
            self.section_deep_radiomics(pdf)
            self.section_survival_analysis(pdf)
            self.section_signature_development(pdf)
            self.section_explainable_ai(pdf)
            self.section_multimodal_integration(pdf)
            self.section_clustering_phenotyping(pdf)
            self.section_longitudinal_delta(pdf)
            self.section_robustness_harmonization(pdf)
            self.section_clinical_trial(pdf)
        
        print(f"🎉 Pipeline complete! Report saved to {output_pdf}")

    def section_intro(self, pdf):
        """Introduction section"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, "Complete Radiomics Research Pipeline", 
                ha='center', va='center', fontsize=24, fontweight='bold', 
                color=GOLDEN_COLORS['primary_gold'])
        
        # Subtitle
        ax.text(0.5, 0.8, "Advanced Analysis Modules (2020-2024)", 
                ha='center', va='center', fontsize=16, 
                color=GOLDEN_COLORS['dark_gold'])
        
        # Modules overview
        modules = [
            "1. Radiogenomics Analysis",
            "2. Deep Radiomics & Feature Learning", 
            "3. Survival & Time-to-Event Analysis",
            "4. Radiomics Signature Development",
            "5. Explainable AI & Model Interpretability",
            "6. Multi-Modal Data Integration",
            "7. Clustering & Phenotyping",
            "8. Longitudinal & Delta Radiomics",
            "9. Feature Robustness & Harmonization",
            "10. Clinical Trial Integration"
        ]
        
        y_pos = 0.65
        for i, module in enumerate(modules):
            color = GOLDEN_COLORS['blue'] if i < 3 else GOLDEN_COLORS['grey']
            ax.text(0.1, y_pos, module, ha='left', va='center', fontsize=12, 
                   color=color, fontweight='bold' if i < 3 else 'normal')
            y_pos -= 0.05
        
        # Dataset info
        ax.text(0.5, 0.1, f"Dataset: {len(self.data)} patients, {len(self.data.columns)-6} radiomics features", 
                ha='center', va='center', fontsize=12, color=GOLDEN_COLORS['dark_gold'])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_radiogenomics(self, pdf):
        """Radiogenomics analysis section"""
        print("🧬 Implementing Radiogenomics Analysis...")
        
        # 1. Correlation analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Radiogenomics Analysis', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # Correlation heatmap
        radiomics_cols = [col for col in self.data.columns if col.startswith('radiomics')]
        correlation_matrix = self.data[radiomics_cols].corrwith(self.genomics.iloc[:, 0]).abs()
        
        axes[0, 0].bar(range(len(correlation_matrix)), correlation_matrix, 
                      color=GOLDEN_COLORS['blue'], alpha=0.7)
        axes[0, 0].set_title('Radiomics-Gene Correlation (Gene_01)', fontweight='bold')
        axes[0, 0].set_xlabel('Radiomics Features')
        axes[0, 0].set_ylabel('Absolute Correlation')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Gene mutation frequency
        gene_freq = self.genomics.sum() / len(self.genomics)
        axes[0, 1].bar(range(len(gene_freq)), gene_freq, color=GOLDEN_COLORS['green'], alpha=0.7)
        axes[0, 1].set_title('Gene Mutation Frequency', fontweight='bold')
        axes[0, 1].set_xlabel('Genes')
        axes[0, 1].set_ylabel('Mutation Frequency')
        
        # Outcome by gene status
        gene_impact = []
        for gene in self.genomics.columns:
            gene_0_outcome = self.data[self.genomics[gene] == 0]['outcome'].mean()
            gene_1_outcome = self.data[self.genomics[gene] == 1]['outcome'].mean()
            gene_impact.append(gene_1_outcome - gene_0_outcome)
        
        axes[1, 0].bar(range(len(gene_impact)), gene_impact, 
                      color=[GOLDEN_COLORS['red'] if x > 0 else GOLDEN_COLORS['blue'] for x in gene_impact],
                      alpha=0.7)
        axes[1, 0].set_title('Outcome Impact by Gene Mutation', fontweight='bold')
        axes[1, 0].set_xlabel('Genes')
        axes[1, 0].set_ylabel('Outcome Difference (Mutated - Wild)')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Combined model performance
        X_combined = np.column_stack([self.data[radiomics_cols], self.genomics])
        X_train, X_test, y_train, y_test = train_test_split(X_combined, self.data['outcome'], 
                                                           test_size=0.3, random_state=42)
        
        # Radiomics only
        lr_radiomics = LogisticRegression(random_state=42)
        lr_radiomics.fit(X_train[:, :len(radiomics_cols)], y_train)
        auc_radiomics = roc_auc_score(y_test, lr_radiomics.predict_proba(X_test[:, :len(radiomics_cols)])[:, 1])
        
        # Combined
        lr_combined = LogisticRegression(random_state=42)
        lr_combined.fit(X_train, y_train)
        auc_combined = roc_auc_score(y_test, lr_combined.predict_proba(X_test)[:, 1])
        
        models = ['Radiomics Only', 'Radiomics + Genomics']
        aucs = [auc_radiomics, auc_combined]
        colors = [GOLDEN_COLORS['blue'], GOLDEN_COLORS['green']]
        
        axes[1, 1].bar(models, aucs, color=colors, alpha=0.7)
        axes[1, 1].set_title('Model Performance Comparison', fontweight='bold')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_ylim(0, 1)
        for i, v in enumerate(aucs):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_deep_radiomics(self, pdf):
        """Deep radiomics analysis section"""
        print("🤖 Implementing Deep Radiomics Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Deep Radiomics & Feature Learning', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # Simulate deep features (in practice, these would come from CNN)
        radiomics_cols = [col for col in self.data.columns if col.startswith('radiomics')]
        X_radiomics = self.data[radiomics_cols].values
        
        # Simulate deep features with different characteristics
        np.random.seed(42)
        deep_features = []
        for i in range(10):
            if i < 3:  # Low-level features (edges, textures)
                deep_feat = np.random.normal(0, 1, len(X_radiomics)) + 0.3 * X_radiomics[:, i]
            elif i < 7:  # Mid-level features (patterns)
                deep_feat = np.random.normal(0, 1, len(X_radiomics)) + 0.2 * np.mean(X_radiomics[:, i:i+3], axis=1)
            else:  # High-level features (semantic)
                deep_feat = np.random.normal(0, 1, len(X_radiomics)) + 0.1 * np.mean(X_radiomics, axis=1)
            deep_features.append(deep_feat)
        
        X_deep = np.column_stack(deep_features)
        
        # 1. Feature space comparison
        # PCA for radiomics
        pca_radiomics = PCA(n_components=2)
        radiomics_pca = pca_radiomics.fit_transform(X_radiomics)
        
        # PCA for deep features
        pca_deep = PCA(n_components=2)
        deep_pca = pca_deep.fit_transform(X_deep)
        
        scatter1 = axes[0, 0].scatter(radiomics_pca[:, 0], radiomics_pca[:, 1], 
                                    c=self.data['outcome'], cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('Radiomics Features (PCA)', fontweight='bold')
        axes[0, 0].set_xlabel(f'PC1 ({pca_radiomics.explained_variance_ratio_[0]:.1%})')
        axes[0, 0].set_ylabel(f'PC2 ({pca_radiomics.explained_variance_ratio_[1]:.1%})')
        
        scatter2 = axes[0, 1].scatter(deep_pca[:, 0], deep_pca[:, 1], 
                                    c=self.data['outcome'], cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Deep Features (PCA)', fontweight='bold')
        axes[0, 1].set_xlabel(f'PC1 ({pca_deep.explained_variance_ratio_[0]:.1%})')
        axes[0, 1].set_ylabel(f'PC2 ({pca_deep.explained_variance_ratio_[1]:.1%})')
        
        # 2. t-SNE comparison
        tsne_radiomics = TSNE(n_components=2, random_state=42, perplexity=30)
        radiomics_tsne = tsne_radiomics.fit_transform(X_radiomics)
        
        tsne_deep = TSNE(n_components=2, random_state=42, perplexity=30)
        deep_tsne = tsne_deep.fit_transform(X_deep)
        
        axes[1, 0].scatter(radiomics_tsne[:, 0], radiomics_tsne[:, 1], 
                          c=self.data['outcome'], cmap='viridis', alpha=0.7)
        axes[1, 0].set_title('Radiomics Features (t-SNE)', fontweight='bold')
        axes[1, 0].set_xlabel('t-SNE 1')
        axes[1, 0].set_ylabel('t-SNE 2')
        
        axes[1, 1].scatter(deep_tsne[:, 0], deep_tsne[:, 1], 
                          c=self.data['outcome'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_title('Deep Features (t-SNE)', fontweight='bold')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Additional page: Feature fusion and performance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature fusion performance
        X_combined = np.column_stack([X_radiomics, X_deep])
        X_train, X_test, y_train, y_test = train_test_split(X_combined, self.data['outcome'], 
                                                           test_size=0.3, random_state=42)
        
        # Compare different feature sets
        feature_sets = {
            'Radiomics Only': X_radiomics,
            'Deep Only': X_deep,
            'Combined': X_combined
        }
        
        auc_scores = []
        for name, features in feature_sets.items():
            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(features, self.data['outcome'], 
                                                                       test_size=0.3, random_state=42)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_train_f, y_train_f)
            auc = roc_auc_score(y_test_f, lr.predict_proba(X_test_f)[:, 1])
            auc_scores.append(auc)
        
        colors = [GOLDEN_COLORS['blue'], GOLDEN_COLORS['green'], GOLDEN_COLORS['purple']]
        bars = axes[0].bar(feature_sets.keys(), auc_scores, color=colors, alpha=0.7)
        axes[0].set_title('Feature Set Performance Comparison', fontweight='bold')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_ylim(0, 1)
        for bar, auc in zip(bars, auc_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Feature importance comparison
        lr_combined = LogisticRegression(random_state=42, max_iter=1000)
        lr_combined.fit(X_train, y_train)
        
        radiomics_importance = np.abs(lr_combined.coef_[0][:len(radiomics_cols)])
        deep_importance = np.abs(lr_combined.coef_[0][len(radiomics_cols):])
        
        axes[1].bar(range(len(radiomics_importance)), radiomics_importance, 
                   color=GOLDEN_COLORS['blue'], alpha=0.7, label='Radiomics')
        axes[1].bar(range(len(radiomics_importance), len(radiomics_importance) + len(deep_importance)), 
                   deep_importance, color=GOLDEN_COLORS['green'], alpha=0.7, label='Deep')
        axes[1].set_title('Feature Importance in Combined Model', fontweight='bold')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Absolute Coefficient')
        axes[1].legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_survival_analysis(self, pdf):
        """Survival analysis section"""
        print("⏰ Implementing Survival Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Survival & Time-to-Event Analysis', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. Kaplan-Meier curves
        from lifelines import KaplanMeierFitter
        
        kmf = KaplanMeierFitter()
        
        # Overall survival
        kmf.fit(self.data['survival_time'], self.data['event'])
        kmf.plot_survival_function(ax=axes[0, 0], color=GOLDEN_COLORS['blue'])
        axes[0, 0].set_title('Overall Survival', fontweight='bold')
        axes[0, 0].set_xlabel('Time (months)')
        axes[0, 0].set_ylabel('Survival Probability')
        
        # Survival by outcome
        kmf_0 = KaplanMeierFitter()
        kmf_1 = KaplanMeierFitter()
        
        kmf_0.fit(self.data[self.data['outcome'] == 0]['survival_time'], 
                 self.data[self.data['outcome'] == 0]['event'])
        kmf_1.fit(self.data[self.data['outcome'] == 1]['survival_time'], 
                 self.data[self.data['outcome'] == 1]['event'])
        
        kmf_0.plot_survival_function(ax=axes[0, 1], color=GOLDEN_COLORS['blue'], label='Outcome 0')
        kmf_1.plot_survival_function(ax=axes[0, 1], color=GOLDEN_COLORS['red'], label='Outcome 1')
        axes[0, 1].set_title('Survival by Outcome', fontweight='bold')
        axes[0, 1].set_xlabel('Time (months)')
        axes[0, 1].set_ylabel('Survival Probability')
        axes[0, 1].legend()
        
        # 2. Cox Proportional Hazards
        from lifelines import CoxPHFitter
        
        # Prepare data for Cox model
        cox_data = self.data.copy()
        radiomics_cols = [col for col in self.data.columns if col.startswith('radiomics')]
        
        # Add some clinical variables
        cox_data['age_scaled'] = (cox_data['age'] - cox_data['age'].mean()) / cox_data['age'].std()
        
        # Fit Cox model
        cph = CoxPHFitter()
        cph.fit(cox_data[['survival_time', 'event', 'age_scaled', 'sex'] + radiomics_cols[:5]], 
               duration_col='survival_time', event_col='event')
        
        # Plot hazard ratios
        cph.plot(ax=axes[1, 0])
        axes[1, 0].set_title('Cox Model: Hazard Ratios', fontweight='bold')
        axes[1, 0].set_xlabel('Variables')
        axes[1, 0].set_ylabel('Hazard Ratio')
        
        # 3. Risk stratification
        # Calculate risk score
        risk_score = cph.predict_partial_hazard(cox_data[['age_scaled', 'sex'] + radiomics_cols[:5]])
        
        # Split into risk groups
        risk_quartiles = pd.qcut(risk_score, 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        cox_data['risk_group'] = risk_quartiles
        
        # Plot survival by risk group
        for group in ['Low', 'Medium-Low', 'Medium-High', 'High']:
            group_data = cox_data[cox_data['risk_group'] == group]
            kmf_group = KaplanMeierFitter()
            kmf_group.fit(group_data['survival_time'], group_data['event'])
            kmf_group.plot_survival_function(ax=axes[1, 1], label=group)
        
        axes[1, 1].set_title('Survival by Risk Group', fontweight='bold')
        axes[1, 1].set_xlabel('Time (months)')
        axes[1, 1].set_ylabel('Survival Probability')
        axes[1, 1].legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_signature_development(self, pdf):
        """Radiomics signature development section"""
        print("🔬 Implementing Radiomics Signature Development...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Radiomics Signature Development', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. Feature selection with LASSO
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        
        radiomics_cols = [col for col in self.data.columns if col.startswith('radiomics')]
        X = self.data[radiomics_cols].values
        y = self.data['outcome'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # LASSO for feature selection
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        
        # Get selected features
        selected_mask = lasso.coef_ != 0
        selected_features = [radiomics_cols[i] for i in range(len(radiomics_cols)) if selected_mask[i]]
        feature_importance = np.abs(lasso.coef_)
        
        # Plot feature importance
        top_features = np.argsort(feature_importance)[-10:]  # Top 10 features
        axes[0, 0].barh(range(len(top_features)), feature_importance[top_features], 
                       color=GOLDEN_COLORS['blue'], alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_features)))
        axes[0, 0].set_yticklabels([radiomics_cols[i] for i in top_features])
        axes[0, 0].set_title('LASSO Feature Selection', fontweight='bold')
        axes[0, 0].set_xlabel('Absolute Coefficient')
        
        # 2. Signature performance
        # Use selected features for prediction
        X_selected = X_scaled[:, selected_mask]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        
        # Train model on selected features
        lr_signature = LogisticRegression(random_state=42, max_iter=1000)
        lr_signature.fit(X_train, y_train)
        
        # Compare with full model
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        lr_full = LogisticRegression(random_state=42, max_iter=1000)
        lr_full.fit(X_train_full, y_train_full)
        
        # Calculate AUCs
        from sklearn.metrics import roc_curve, auc
        
        # Full model
        y_pred_full = lr_full.predict_proba(X_test_full)[:, 1]
        fpr_full, tpr_full, _ = roc_curve(y_test_full, y_pred_full)
        auc_full = auc(fpr_full, tpr_full)
        
        # Signature model
        y_pred_sig = lr_signature.predict_proba(X_test)[:, 1]
        fpr_sig, tpr_sig, _ = roc_curve(y_test, y_pred_sig)
        auc_sig = auc(fpr_sig, tpr_sig)
        
        # Plot ROC curves
        axes[0, 1].plot(fpr_full, tpr_full, color=GOLDEN_COLORS['blue'], 
                       label=f'Full Model (AUC={auc_full:.3f})', linewidth=2)
        axes[0, 1].plot(fpr_sig, tpr_sig, color=GOLDEN_COLORS['green'], 
                       label=f'Signature (AUC={auc_sig:.3f})', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_title('ROC Comparison: Full vs Signature', fontweight='bold')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Nomogram visualization
        # Create a simple nomogram based on top features
        n_selected = len(selected_features)
        if n_selected > 0:
            # Calculate risk scores for different feature combinations
            risk_scores = []
            
            for i in range(10):
                # Generate random feature values
                values = np.random.normal(0, 1, n_selected)
                
                # Calculate risk score
                risk = lr_signature.intercept_[0] + np.sum(lr_signature.coef_[0] * values)
                risk_scores.append(1 / (1 + np.exp(-risk)))  # Convert to probability
            
            # Plot risk score distribution
            axes[1, 0].hist(risk_scores, bins=10, color=GOLDEN_COLORS['blue'], alpha=0.7)
            axes[1, 0].set_title('Risk Score Distribution', fontweight='bold')
            axes[1, 0].set_xlabel('Risk Score')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(np.mean(risk_scores), color=GOLDEN_COLORS['red'], 
                              linestyle='--', label=f'Mean: {np.mean(risk_scores):.3f}')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No features selected by LASSO', 
                           ha='center', va='center', fontsize=12, color=GOLDEN_COLORS['grey'])
            axes[1, 0].set_title('Nomogram (No Features Selected)', fontweight='bold')
        
        # 4. Risk stratification
        # Calculate risk scores for all patients
        all_risk_scores = lr_signature.predict_proba(X_scaled[:, selected_mask])[:, 1]
        
        # Create risk groups
        risk_groups = pd.qcut(all_risk_scores, 4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        # Calculate outcome rates by risk group
        outcome_rates = []
        for group in ['Low', 'Medium-Low', 'Medium-High', 'High']:
            group_mask = risk_groups == group
            rate = self.data.loc[group_mask, 'outcome'].mean()
            outcome_rates.append(rate)
        
        # Plot risk stratification
        colors = [GOLDEN_COLORS['green'], GOLDEN_COLORS['light_gold'], 
                 GOLDEN_COLORS['orange'], GOLDEN_COLORS['red']]
        bars = axes[1, 1].bar(['Low', 'Medium-Low', 'Medium-High', 'High'], 
                             outcome_rates, color=colors, alpha=0.7)
        axes[1, 1].set_title('Outcome Rate by Risk Group', fontweight='bold')
        axes[1, 1].set_ylabel('Outcome Rate')
        axes[1, 1].set_ylim(0, 1)
        
        for bar, rate in zip(bars, outcome_rates):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_explainable_ai(self, pdf):
        """Explainable AI section"""
        print("🤖 Implementing Explainable AI Analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Explainable AI & Model Interpretability', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. Feature importance from different models
        radiomics_cols = [col for col in self.data.columns if col.startswith('radiomics')]
        X = self.data[radiomics_cols].values
        y = self.data['outcome'].values
        
        # Train multiple models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Get feature importance from each model
        importance_data = {}
        for name, model in models.items():
            model.fit(X, y)
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = model.feature_importances_
            else:
                importance_data[name] = np.abs(model.coef_[0])
        
        # Plot feature importance comparison
        top_features_idx = np.argsort(importance_data['Random Forest'])[-5:]  # Top 5 features
        x_pos = np.arange(len(top_features_idx))
        width = 0.25
        
        for i, (name, importance) in enumerate(importance_data.items()):
            axes[0, 0].bar(x_pos + i*width, importance[top_features_idx], width, 
                          label=name, alpha=0.7)
        
        axes[0, 0].set_title('Feature Importance Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Top Features')
        axes[0, 0].set_ylabel('Importance Score')
        axes[0, 0].set_xticks(x_pos + width)
        axes[0, 0].set_xticklabels([radiomics_cols[i] for i in top_features_idx], rotation=45)
        axes[0, 0].legend()
        
        # 2. Permutation importance
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        perm_importance = permutation_importance(rf_model, X, y, n_repeats=10, random_state=42)
        
        # Plot permutation importance
        top_perm_idx = np.argsort(perm_importance.importances_mean)[-8:]
        axes[0, 1].barh(range(len(top_perm_idx)), perm_importance.importances_mean[top_perm_idx], 
                       color=GOLDEN_COLORS['green'], alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_perm_idx)))
        axes[0, 1].set_yticklabels([radiomics_cols[i] for i in top_perm_idx])
        axes[0, 1].set_title('Permutation Importance', fontweight='bold')
        axes[0, 1].set_xlabel('Importance Score')
        
        # 3. Partial dependence plots (simplified)
        # Show how individual features affect predictions
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Calculate partial dependence for top feature
        top_feature_idx = np.argmax(perm_importance.importances_mean)
        feature_values = np.linspace(X[:, top_feature_idx].min(), X[:, top_feature_idx].max(), 50)
        
        partial_dependence = []
        for val in feature_values:
            X_temp = X_test.copy()
            X_temp[:, top_feature_idx] = val
            pred = rf_model.predict_proba(X_temp)[:, 1].mean()
            partial_dependence.append(pred)
        
        axes[1, 0].plot(feature_values, partial_dependence, color=GOLDEN_COLORS['blue'], linewidth=2)
        axes[1, 0].set_title(f'Partial Dependence: {radiomics_cols[top_feature_idx]}', fontweight='bold')
        axes[1, 0].set_xlabel('Feature Value')
        axes[1, 0].set_ylabel('Predicted Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model confidence analysis
        # Analyze prediction confidence
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        y_pred = rf_model.predict(X_test)
        
        # Calculate confidence (distance from decision boundary)
        confidence = np.abs(y_pred_proba - 0.5) * 2  # Scale to 0-1
        
        # Plot confidence distribution
        correct_predictions = (y_pred == y_test)
        
        axes[1, 1].hist(confidence[correct_predictions], bins=20, alpha=0.7, 
                       color=GOLDEN_COLORS['green'], label='Correct Predictions', density=True)
        axes[1, 1].hist(confidence[~correct_predictions], bins=20, alpha=0.7, 
                       color=GOLDEN_COLORS['red'], label='Incorrect Predictions', density=True)
        axes[1, 1].set_title('Prediction Confidence Analysis', fontweight='bold')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_multimodal_integration(self, pdf):
        """Multi-modal integration section"""
        print("🔗 Implementing Multi-Modal Integration...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Multi-Modal Data Integration', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])

        # Prepare data
        radiomics_cols = [col for col in self.data.columns if col.startswith('radiomics')]
        clinical_cols = ['age', 'sex', 'bmi']
        genomics_cols = self.genomics.columns.tolist()
        X_radiomics = self.data[radiomics_cols].values
        X_clinical = self.data[clinical_cols].values
        X_genomics = self.genomics.values
        y = self.data['outcome'].values

        # Standardize all
        scaler = StandardScaler()
        X_radiomics_scaled = scaler.fit_transform(X_radiomics)
        X_clinical_scaled = scaler.fit_transform(X_clinical)
        X_genomics_scaled = scaler.fit_transform(X_genomics)
        X_all = np.concatenate([X_radiomics_scaled, X_clinical_scaled, X_genomics_scaled], axis=1)

        # Compare model performance
        from sklearn.model_selection import cross_val_score
        model = LogisticRegression(max_iter=1000, random_state=42)
        auc_radiomics = cross_val_score(model, X_radiomics_scaled, y, cv=5, scoring='roc_auc').mean()
        auc_clinical = cross_val_score(model, X_clinical_scaled, y, cv=5, scoring='roc_auc').mean()
        auc_genomics = cross_val_score(model, X_genomics_scaled, y, cv=5, scoring='roc_auc').mean()
        auc_all = cross_val_score(model, X_all, y, cv=5, scoring='roc_auc').mean()

        aucs = [auc_radiomics, auc_clinical, auc_genomics, auc_all]
        labels = ['Radiomics', 'Clinical', 'Genomics', 'All Combined']
        colors = [GOLDEN_COLORS['blue'], GOLDEN_COLORS['orange'], GOLDEN_COLORS['green'], GOLDEN_COLORS['purple']]
        bars = axes[0].bar(labels, aucs, color=colors, alpha=0.7)
        axes[0].set_title('Model Performance by Modality', fontweight='bold')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_ylim(0, 1)
        for bar, auc in zip(bars, aucs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Correlation heatmap between modalities
        combined_df = pd.concat([
            pd.DataFrame(X_radiomics_scaled, columns=[f'R_{i+1}' for i in range(X_radiomics.shape[1])]),
            pd.DataFrame(X_clinical_scaled, columns=clinical_cols),
            pd.DataFrame(X_genomics_scaled, columns=[f'G_{i+1}' for i in range(X_genomics.shape[1])])
        ], axis=1)
        corr = combined_df.corr()
        sns.heatmap(corr, ax=axes[1], cmap='YlOrBr', cbar=True, square=True, xticklabels=False, yticklabels=False)
        axes[1].set_title('Feature Correlation Heatmap', fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_clustering_phenotyping(self, pdf):
        """Clustering and phenotyping section"""
        print("🔎 Implementing Clustering & Phenotyping...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Clustering & Phenotyping', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])

        # Use UMAP if available, else t-SNE
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            method = 'UMAP'
        except ImportError:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            method = 't-SNE'

        radiomics_cols = [col for col in self.data.columns if col.startswith('radiomics')]
        X = self.data[radiomics_cols].values
        y = self.data['outcome'].values
        X_embedded = reducer.fit_transform(X)

        # K-means clustering
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_embedded)

        # Plot embedding colored by cluster
        scatter = axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=clusters, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'{method} Embedding with K-means Clusters', fontweight='bold')
        axes[0].set_xlabel(f'{method} 1')
        axes[0].set_ylabel(f'{method} 2')
        legend1 = axes[0].legend(*scatter.legend_elements(), title="Cluster")
        axes[0].add_artist(legend1)

        # Cluster-outcome association
        cluster_outcomes = [np.mean(y[clusters == i]) for i in range(n_clusters)]
        axes[1].bar([f'Cluster {i+1}' for i in range(n_clusters)], cluster_outcomes, color=GOLDEN_COLORS['blue'], alpha=0.7)
        axes[1].set_title('Outcome Rate by Cluster', fontweight='bold')
        axes[1].set_ylabel('Mean Outcome')
        axes[1].set_ylim(0, 1)
        for i, v in enumerate(cluster_outcomes):
            axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def section_longitudinal_delta(self, pdf):
        """Longitudinal and delta radiomics (placeholder)"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, "[Longitudinal & Delta Radiomics]\nFeature change over time, delta analysis\n(placeholder - to be implemented)",
                ha='center', va='center', fontsize=14, color=GOLDEN_COLORS['secondary_gold'])
        pdf.savefig(fig)
        plt.close()

    def section_robustness_harmonization(self, pdf):
        """Feature robustness and harmonization (placeholder)"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, "[Feature Robustness & Harmonization]\nComBat, multi-center harmonization\n(placeholder - to be implemented)",
                ha='center', va='center', fontsize=14, color=GOLDEN_COLORS['grey'])
        pdf.savefig(fig)
        plt.close()

    def section_clinical_trial(self, pdf):
        """Clinical trial integration (placeholder)"""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, "[Clinical Trial Integration]\nStratification, enrichment, trial design\n(placeholder - to be implemented)",
                ha='center', va='center', fontsize=14, color=GOLDEN_COLORS['pink'])
        pdf.savefig(fig)
        plt.close()

def main():
    pipeline = CompleteRadiomicsResearchPipeline()
    pipeline.run_pipeline('complete_radiomics_research_report.pdf')

if __name__ == "__main__":
    main() 