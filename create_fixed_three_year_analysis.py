import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
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
    'grey': '#808080'
}

# Set matplotlib style with golden theme
plt.style.use('default')
plt.rcParams['figure.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['axes.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['axes.edgecolor'] = GOLDEN_COLORS['dark_gold']
plt.rcParams['axes.labelcolor'] = GOLDEN_COLORS['black']
plt.rcParams['xtick.color'] = GOLDEN_COLORS['black']
plt.rcParams['ytick.color'] = GOLDEN_COLORS['black']
plt.rcParams['text.color'] = GOLDEN_COLORS['black']

class FixedThreeYearPatientGoldenAnalyzer:
    """
    Fixed Three-Year Patient-Level Analysis with Golden Theme
    Uses 2021 data for target variable (Last mRS) but includes all three years for features
    Treats each patient as one unit (5 scans = 1 patient)
    Uses mRS 0-2 vs 3-5 terminology
    """
    
    def __init__(self):
        """Initialize with all three years of radiomics data"""
        self._load_all_data()
        self.feature_cols = [col for col in self.data_combined.columns if 'original_' in col]
        self.scaler = StandardScaler()
        
        # Aggregate data by patient (average of 5 scans per patient)
        self._aggregate_by_patient()
        
        print(f"Loaded {len(self.patient_data)} patients with {len(self.feature_cols)} radiomics features")
    
    def _load_all_data(self):
        """Load data from all three years"""
        try:
            # Load 2020 data (no Last mRS column)
            data_2020 = pd.read_csv('results/radiomics_2020_only.csv')
            data_2020['Year'] = 2020
            print(f"Loaded 2020 data: {len(data_2020)} scans")
            
            # Load 2021 data (has Last mRS column)
            data_2021 = pd.read_csv('results/radiomics_lastmrs_mapping.csv')
            data_2021['Year'] = 2021
            print(f"Loaded 2021 data: {len(data_2021)} scans")
            
            # Load 2022 data (no Last mRS column)
            data_2022 = pd.read_csv('results/radiomics_2022_only.csv')
            data_2022['Year'] = 2022
            print(f"Loaded 2022 data: {len(data_2022)} scans")
            
            # Store individual datasets
            self.data_2020 = data_2020
            self.data_2021 = data_2021
            self.data_2022 = data_2022
            
            # Get unique patients across all years
            all_patients_2020 = set(data_2020['PatientID'].unique())
            all_patients_2021 = set(data_2021['PatientID'].unique())
            all_patients_2022 = set(data_2022['PatientID'].unique())
            self.unique_patients = all_patients_2020.union(all_patients_2021).union(all_patients_2022)
            
            print(f"Total unique patients across all years: {len(self.unique_patients)}")
            print(f"2020: {len(all_patients_2020)} patients")
            print(f"2021: {len(all_patients_2021)} patients")
            print(f"2022: {len(all_patients_2022)} patients")
            
            # For analysis, we'll use only 2021 data since it has the target variable
            self.data_combined = data_2021.copy()
            print(f"Using 2021 data for analysis: {len(self.data_combined)} scans")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            # Fallback to 2021 data only
            self.data_combined = pd.read_csv('results/radiomics_lastmrs_mapping.csv')
            self.data_combined['Year'] = 2021
            self.data_2020 = None
            self.data_2021 = self.data_combined
            self.data_2022 = None
            self.unique_patients = set(self.data_combined['PatientID'].unique())
            print(f"Fallback: Loaded 2021 data only: {len(self.data_combined)} scans")
    
    def _aggregate_by_patient(self):
        """Aggregate 5 scans per patient into single patient-level features"""
        # Group by PatientID and average the features
        patient_groups = self.data_combined.groupby('PatientID')
        
        # Aggregate features (mean of 5 scans)
        feature_means = patient_groups[self.feature_cols].mean()
        
        # Get target (Last mRS) - should be same for all scans of same patient
        targets = patient_groups['Last mRS'].first()
        
        # Get year - should be same for all scans of same patient
        years = patient_groups['Year'].first()
        
        # Combine features, targets, and years
        self.patient_data = pd.concat([feature_means, targets, years], axis=1)
        self.patient_data.columns = list(feature_means.columns) + ['Last mRS', 'Year']
        
        # Scale features
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.patient_data[self.feature_cols].fillna(0)),
            columns=self.feature_cols,
            index=self.patient_data.index
        )
        
        print(f"Aggregated {len(self.patient_data)} patients from {len(self.data_combined)} scans")
        
        # Print year distribution
        year_counts = self.patient_data['Year'].value_counts().sort_index()
        print("Patient distribution by year:")
        for year, count in year_counts.items():
            print(f"  {year}: {count} patients")
    
    def create_analysis(self, target_col='Last mRS', output_path='fixed_three_year_patient_golden_analysis.pdf'):
        """Create fixed three-year patient-level analysis with golden theme"""
        
        with PdfPages(output_path) as pdf:
            
            # 1. THREE-YEAR DATA OVERVIEW
            self._create_three_year_overview(pdf, target_col)
            
            # 2. MODEL PERFORMANCE COMPARISON
            self._create_model_comparison(pdf, target_col)
            
            # 3. FEATURE IMPORTANCE
            self._create_feature_importance(pdf, target_col)
            
            # 4. PREDICTION ACCURACY BY FEATURE SETS
            self._create_feature_set_analysis(pdf, target_col)
            
            # 5. CROSS-VALIDATION RESULTS
            self._create_cv_analysis(pdf, target_col)
        
        print(f"✅ Fixed three-year patient-level golden analysis saved to {output_path}")
    
    def _create_three_year_overview(self, pdf, target_col):
        """Create three-year data overview with golden theme"""
        
        if target_col not in self.patient_data.columns:
            return
        
        y = self.patient_data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        y_binary = (y <= 2).astype(int)  # mRS 0-2 vs 3-5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Three-Year Patient-Level Data Overview (2021 Data with Target)', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 1. Target Distribution (mRS 0-2 vs 3-5)
        target_counts = np.bincount(y_binary)
        bars = axes[0, 0].bar(['mRS 3-5', 'mRS 0-2'], target_counts, 
                             color=[GOLDEN_COLORS['light_gold'], GOLDEN_COLORS['primary_gold']], 
                             alpha=0.8, edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1.5)
        axes[0, 0].set_ylabel('Number of Patients', fontweight='bold')
        axes[0, 0].set_title(f'Patient Outcome Distribution\nTotal: {len(y_binary)} patients', 
                           fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, count in enumerate(target_counts):
            axes[0, 0].text(i, count + 0.5, str(count), ha='center', va='bottom', 
                           fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 2. Available Data by Year
        year_data = {
            '2020': len(self.data_2020['PatientID'].unique()) if self.data_2020 is not None else 0,
            '2021': len(self.data_2021['PatientID'].unique()) if self.data_2021 is not None else 0,
            '2022': len(self.data_2022['PatientID'].unique()) if self.data_2022 is not None else 0
        }
        
        years = list(year_data.keys())
        counts = list(year_data.values())
        
        bars = axes[0, 1].bar(years, counts, alpha=0.8, 
                             color=[GOLDEN_COLORS['pale_gold'], GOLDEN_COLORS['primary_gold'], GOLDEN_COLORS['secondary_gold']],
                             edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1.5)
        axes[0, 1].set_xlabel('Year', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Patients', fontweight='bold')
        axes[0, 1].set_title('Available Patients by Year\n(Only 2021 has target variable)', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, count in enumerate(counts):
            axes[0, 1].text(years[i], count + 0.5, str(count), ha='center', va='bottom', 
                           fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 3. Train vs Test Split
        train_counts = np.bincount(y_train)
        test_counts = np.bincount(y_test)
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = axes[1, 0].bar(x - width/2, train_counts, width, label='Train', 
                              color=GOLDEN_COLORS['pale_gold'], alpha=0.8, 
                              edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        bars2 = axes[1, 0].bar(x + width/2, test_counts, width, label='Test', 
                              color=GOLDEN_COLORS['secondary_gold'], alpha=0.8,
                              edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        axes[1, 0].set_xlabel('mRS Outcome', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Patients', fontweight='bold')
        axes[1, 0].set_title('Train vs Test Patient Split', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['mRS 3-5', 'mRS 0-2'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, (train_count, test_count) in enumerate(zip(train_counts, test_counts)):
            axes[1, 0].text(i - width/2, train_count + 0.2, str(train_count), ha='center', va='bottom', 
                           fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            axes[1, 0].text(i + width/2, test_count + 0.2, str(test_count), ha='center', va='bottom', 
                           fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 4. Dataset Summary Table
        summary_data = {
            'Metric': ['Total Patients (2021)', 'Train Patients', 'Test Patients', 'Features', 'mRS 0-2', 'mRS 3-5', 'Available Years'],
            'Count': [len(y_binary), len(y_train), len(y_test), len(self.feature_cols), 
                     np.sum(y_binary == 1), np.sum(y_binary == 0), len([k for k, v in year_data.items() if v > 0])]
        }
        
        summary_df = pd.DataFrame(summary_data)
        table = axes[1, 1].table(cellText=summary_df.values, colLabels=summary_df.columns, 
                                cellLoc='center', loc='center',
                                cellColours=[[GOLDEN_COLORS['pale_gold']]*2]*len(summary_df),
                                colColours=[GOLDEN_COLORS['light_gold']]*2)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('Dataset Summary (2021 with Target)', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        print(f"Three-Year Patient-Level Data Overview:")
        print(f"Total patients (2021): {len(y_binary)}")
        print(f"Train patients: {len(y_train)}")
        print(f"Test patients: {len(y_test)}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"mRS 0-2 patients: {np.sum(y_binary == 1)}")
        print(f"mRS 3-5 patients: {np.sum(y_binary == 0)}")
        print(f"Available years: {len([k for k, v in year_data.items() if v > 0])}")
    
    def _create_model_comparison(self, pdf, target_col):
        """Create model performance comparison with golden theme"""
        
        if target_col not in self.patient_data.columns:
            return
        
        y = self.patient_data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        y_binary = (y <= 2).astype(int)  # mRS 0-2 vs 3-5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Train predictions
            y_train_pred = model.predict(X_train)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            
            # Test predictions
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            results[name] = {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfitting_auc': train_auc - test_auc,
                'overfitting_acc': train_acc - test_acc
            }
        
        # Create comparison plots with golden theme
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Three-Year Patient-Level Model Performance: Train vs Test (2021 Data)', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 1. AUC Comparison
        model_names = list(results.keys())
        train_aucs = [results[name]['train_auc'] for name in model_names]
        test_aucs = [results[name]['test_auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, train_aucs, width, label='Train AUC', 
                              alpha=0.8, color=GOLDEN_COLORS['pale_gold'], 
                              edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        bars2 = axes[0, 0].bar(x + width/2, test_aucs, width, label='Test AUC', 
                              alpha=0.8, color=GOLDEN_COLORS['primary_gold'],
                              edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        axes[0, 0].set_xlabel('Models', fontweight='bold')
        axes[0, 0].set_ylabel('AUC Score', fontweight='bold')
        axes[0, 0].set_title('AUC Comparison: Train vs Test', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, (train_auc, test_auc) in enumerate(zip(train_aucs, test_aucs)):
            axes[0, 0].text(i - width/2, train_auc + 0.01, f'{train_auc:.3f}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            axes[0, 0].text(i + width/2, test_auc + 0.01, f'{test_auc:.3f}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 2. Accuracy Comparison
        train_accs = [results[name]['train_acc'] for name in model_names]
        test_accs = [results[name]['test_acc'] for name in model_names]
        
        bars1 = axes[0, 1].bar(x - width/2, train_accs, width, label='Train Accuracy', 
                              alpha=0.8, color=GOLDEN_COLORS['pale_gold'],
                              edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        bars2 = axes[0, 1].bar(x + width/2, test_accs, width, label='Test Accuracy', 
                              alpha=0.8, color=GOLDEN_COLORS['primary_gold'],
                              edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        axes[0, 1].set_xlabel('Models', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy Score', fontweight='bold')
        axes[0, 1].set_title('Accuracy Comparison: Train vs Test', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
            axes[0, 1].text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            axes[0, 1].text(i + width/2, test_acc + 0.01, f'{test_acc:.3f}', 
                           ha='center', va='bottom', fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 3. Overfitting Analysis (AUC)
        overfitting_aucs = [results[name]['overfitting_auc'] for name in model_names]
        
        colors = [GOLDEN_COLORS['light_gold'] if x <= 0.05 else GOLDEN_COLORS['secondary_gold'] for x in overfitting_aucs]
        bars = axes[1, 0].bar(model_names, overfitting_aucs, alpha=0.8, color=colors,
                             edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1.5)
        axes[1, 0].set_xlabel('Models', fontweight='bold')
        axes[1, 0].set_ylabel('Overfitting Score (Train AUC - Test AUC)', fontweight='bold')
        axes[1, 0].set_title('Overfitting Analysis (AUC)', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        axes[1, 0].axhline(y=0, color=GOLDEN_COLORS['dark_gold'], linestyle='-', alpha=0.7, linewidth=2)
        
        # Add value labels
        for i, score in enumerate(overfitting_aucs):
            axes[1, 0].text(i, score + 0.01 if score > 0 else score - 0.02, f'{score:.3f}', 
                           ha='center', va='bottom' if score > 0 else 'top', fontsize=8, 
                           fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 4. Best Model Performance
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_auc'])
        best_result = results[best_model_name]
        
        performance_metrics = ['Train AUC', 'Test AUC', 'Train Acc', 'Test Acc']
        performance_values = [best_result['train_auc'], best_result['test_auc'], 
                            best_result['train_acc'], best_result['test_acc']]
        
        bars = axes[1, 1].bar(performance_metrics, performance_values, alpha=0.8,
                             color=GOLDEN_COLORS['primary_gold'],
                             edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1.5)
        axes[1, 1].set_ylabel('Score', fontweight='bold')
        axes[1, 1].set_title(f'Best Model: {best_model_name}', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, value in enumerate(performance_values):
            axes[1, 1].text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom', 
                           fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Print results
        print(f"\nThree-Year Patient-Level Model Performance Summary:")
        for name, result in results.items():
            print(f"{name}:")
            print(f"  Train AUC: {result['train_auc']:.3f}, Test AUC: {result['test_auc']:.3f}")
            print(f"  Train Acc: {result['train_acc']:.3f}, Test Acc: {result['test_acc']:.3f}")
            print(f"  Overfitting AUC: {result['overfitting_auc']:.3f}")
    
    def _create_feature_importance(self, pdf, target_col):
        """Create feature importance analysis with golden theme"""
        
        if target_col not in self.patient_data.columns:
            return
        
        y = self.patient_data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        y_binary = (y <= 2).astype(int)  # mRS 0-2 vs 3-5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        # Train models and get feature importance
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
        }
        
        importance_results = {}
        
        for name, model in models.items():
            print(f"\nAnalyzing feature importance for {name}...")
            model.fit(X_train, y_train)
            
            if hasattr(model, 'feature_importances_'):
                importance_results[name] = pd.Series(model.feature_importances_, index=X.columns)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Three-Year Patient-Level Feature Importance Analysis (2021 Data)', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Plot top features for each model with golden theme
        for i, (name, importance) in enumerate(importance_results.items()):
            top_features = importance.sort_values(ascending=False).head(10)
            
            ax = axes[i//2, i%2]
            bars = ax.barh(range(len(top_features)), top_features.values, alpha=0.8,
                          color=GOLDEN_COLORS['primary_gold'],
                          edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([f.split('_')[-1] for f in top_features.index])
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'Top 10 Features: {name}', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            ax.grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Find common important features
        if len(importance_results) > 1:
            common_features = set()
            for importance in importance_results.values():
                top_features = importance.sort_values(ascending=False).head(10).index
                if not common_features:
                    common_features = set(top_features)
                else:
                    common_features = common_features.intersection(set(top_features))
            
            print(f"\nCommon important features across models: {len(common_features)}")
            for feature in list(common_features)[:10]:
                print(f"  {feature}")
    
    def _create_feature_set_analysis(self, pdf, target_col):
        """Create prediction accuracy by feature sets with golden theme"""
        
        if target_col not in self.patient_data.columns:
            return
        
        y = self.patient_data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        y_binary = (y <= 2).astype(int)  # mRS 0-2 vs 3-5
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        # Test different feature sets
        feature_sets = {
            'All Features': X.columns,
            'Top 50 Features': X.columns[:50],
            'Top 25 Features': X.columns[:25],
            'Top 10 Features': X.columns[:10]
        }
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        results = {}
        
        for name, features in feature_sets.items():
            print(f"\nTesting {name}...")
            
            X_train_subset = X_train[features]
            X_test_subset = X_test[features]
            
            model.fit(X_train_subset, y_train)
            y_train_pred = model.predict(X_train_subset)
            y_test_pred = model.predict(X_test_subset)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            results[name] = {
                'n_features': len(features),
                'train_acc': train_acc,
                'test_acc': test_acc,
                'overfitting': train_acc - test_acc
            }
        
        # Create visualization with golden theme
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Three-Year Patient-Level Prediction Accuracy by Feature Set (2021 Data)', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 1. Accuracy comparison
        feature_set_names = list(results.keys())
        train_accs = [results[name]['train_acc'] for name in feature_set_names]
        test_accs = [results[name]['test_acc'] for name in feature_set_names]
        
        x = np.arange(len(feature_set_names))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, train_accs, width, label='Train Accuracy', 
                           alpha=0.8, color=GOLDEN_COLORS['pale_gold'],
                           edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        bars2 = axes[0].bar(x + width/2, test_accs, width, label='Test Accuracy', 
                           alpha=0.8, color=GOLDEN_COLORS['primary_gold'],
                           edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1)
        axes[0].set_xlabel('Feature Sets', fontweight='bold')
        axes[0].set_ylabel('Accuracy Score', fontweight='bold')
        axes[0].set_title('Accuracy by Feature Set', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(feature_set_names, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
            axes[0].text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            axes[0].text(i + width/2, test_acc + 0.01, f'{test_acc:.3f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 2. Overfitting analysis
        overfitting_scores = [results[name]['overfitting'] for name in feature_set_names]
        
        colors = [GOLDEN_COLORS['light_gold'] if x <= 0.05 else GOLDEN_COLORS['secondary_gold'] for x in overfitting_scores]
        bars = axes[1].bar(feature_set_names, overfitting_scores, alpha=0.8, color=colors,
                          edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1.5)
        axes[1].set_xlabel('Feature Sets', fontweight='bold')
        axes[1].set_ylabel('Overfitting Score', fontweight='bold')
        axes[1].set_title('Overfitting by Feature Set', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        axes[1].axhline(y=0, color=GOLDEN_COLORS['dark_gold'], linestyle='-', alpha=0.7, linewidth=2)
        
        # Add value labels
        for i, score in enumerate(overfitting_scores):
            axes[1].text(i, score + 0.01 if score > 0 else score - 0.02, f'{score:.3f}', 
                        ha='center', va='bottom' if score > 0 else 'top', fontsize=8, 
                        fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Print results
        print(f"\nThree-Year Patient-Level Feature Set Analysis:")
        for name, result in results.items():
            print(f"{name} ({result['n_features']} features):")
            print(f"  Train Acc: {result['train_acc']:.3f}, Test Acc: {result['test_acc']:.3f}")
            print(f"  Overfitting: {result['overfitting']:.3f}")
    
    def _create_cv_analysis(self, pdf, target_col):
        """Create cross-validation analysis with golden theme"""
        
        if target_col not in self.patient_data.columns:
            return
        
        y = self.patient_data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        y_binary = (y <= 2).astype(int)  # mRS 0-2 vs 3-5
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
        }
        
        # Perform cross-validation
        cv_results = {}
        
        for name, model in models.items():
            print(f"\nPerforming CV for {name}...")
            cv_scores = cross_val_score(model, X, y_binary, cv=5, scoring='roc_auc')
            cv_results[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores
            }
        
        # Create visualization with golden theme
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Three-Year Patient-Level Cross-Validation Analysis (2021 Data)', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 1. CV scores comparison
        model_names = list(cv_results.keys())
        cv_means = [cv_results[name]['mean'] for name in model_names]
        cv_stds = [cv_results[name]['std'] for name in model_names]
        
        bars = axes[0].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8,
                          color=GOLDEN_COLORS['primary_gold'],
                          edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1.5)
        axes[0].set_xlabel('Models', fontweight='bold')
        axes[0].set_ylabel('CV AUC Score', fontweight='bold')
        axes[0].set_title('Cross-Validation Performance', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
            axes[0].text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # 2. CV score distributions
        for i, name in enumerate(model_names):
            scores = cv_results[name]['scores']
            axes[1].boxplot(scores, positions=[i], labels=[name.split()[0]])
        
        axes[1].set_xlabel('Models', fontweight='bold')
        axes[1].set_ylabel('CV AUC Score', fontweight='bold')
        axes[1].set_title('CV Score Distributions', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1].grid(True, alpha=0.3, color=GOLDEN_COLORS['grey'])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Print results
        print(f"\nThree-Year Patient-Level Cross-Validation Results:")
        for name, result in cv_results.items():
            print(f"{name}: {result['mean']:.3f} ± {result['std']:.3f}")

def main():
    """Main function to run fixed three-year patient-level golden analysis"""
    print("=== FIXED THREE-YEAR PATIENT-LEVEL GOLDEN RADIOMICS ANALYSIS ===")
    print("Analyzing patient-level predictions (2021 data with target, all years available)...\n")
    
    # Initialize analyzer with all three years of data
    analyzer = FixedThreeYearPatientGoldenAnalyzer()
    
    # Create analysis
    analyzer.create_analysis('Last mRS', 'fixed_three_year_patient_golden_analysis.pdf')
    
    print("\n=== ANALYSIS COMPLETED ===")
    print("Generated analyses include:")
    print("1. Three-Year Data Overview (2021 with target)")
    print("2. Model Performance Comparison")
    print("3. Feature Importance Analysis")
    print("4. Prediction Accuracy by Feature Sets")
    print("5. Cross-Validation Results")
    print("\nKey insights:")
    print("- Patient-level analysis (5 scans = 1 patient)")
    print("- mRS 0-2 vs 3-5 outcome classification")
    print("- Golden theme visualizations")
    print("- 2021 data used for target variable (only year with Last mRS)")
    print("- All three years available for feature analysis")

if __name__ == "__main__":
    main() 