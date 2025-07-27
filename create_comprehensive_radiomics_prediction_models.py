import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import (roc_curve, auc, classification_report, confusion_matrix, 
                           mean_absolute_error, r2_score, roc_auc_score, accuracy_score)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveRadiomicsPredictor:
    """
    Comprehensive Radiomics Prediction Models
    Implements state-of-the-art prediction models for radiomics analysis
    """
    
    def __init__(self, data_path):
        """Initialize with radiomics data"""
        self.data = pd.read_csv(data_path)
        self.feature_cols = [col for col in self.data.columns if 'original_' in col]
        self.scaler = StandardScaler()
        
        # Prepare scaled features
        self.features_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.data[self.feature_cols].fillna(0)),
            columns=self.feature_cols,
            index=self.data.index
        )
        
        print(f"Loaded {len(self.features_scaled)} samples with {len(self.feature_cols)} radiomics features")
    
    def create_prediction_models(self, target_col='Last mRS', output_path='comprehensive_radiomics_predictions.pdf'):
        """Create comprehensive prediction models for radiomics"""
        
        with PdfPages(output_path) as pdf:
            
            # 1. CLASSIFICATION MODELS
            self._create_classification_models(pdf, target_col)
            
            # 2. REGRESSION MODELS
            self._create_regression_models(pdf, target_col)
            
            # 3. FEATURE SELECTION ANALYSIS
            self._create_feature_selection_analysis(pdf, target_col)
            
            # 4. MODEL COMPARISON
            self._create_model_comparison(pdf, target_col)
            
            # 5. ADVANCED MODELS
            self._create_advanced_models(pdf, target_col)
            
            # 6. CLINICAL INTEGRATION
            self._create_clinical_integration(pdf, target_col)
        
        print(f"✅ Comprehensive prediction models saved to {output_path}")
    
    def _create_classification_models(self, pdf, target_col):
        """Create classification models for radiomics"""
        
        # Prepare target for classification
        if target_col in self.data.columns:
            y = self.data[target_col].dropna()
            X = self.features_scaled.loc[y.index]
            
            # Create binary classification target (good vs poor outcome)
            y_binary = (y <= 2).astype(int)  # mRS 0-2 vs 3-6
            
            print(f"\nClassification Target: {target_col}")
            print(f"Binary distribution: {y_binary.value_counts().to_dict()}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
            )
            
            # Define models
            models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'SVM': SVC(kernel='linear', probability=True, random_state=42),
                'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
            }
            
            # Train and evaluate models
            results = {}
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Radiomics Classification Models', fontsize=16, fontweight='bold')
            
            for i, (name, model) in enumerate(models.items()):
                print(f"\nTraining {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc_score = auc(fpr, tpr)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                results[name] = {
                    'auc': auc_score,
                    'accuracy': accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'fpr': fpr,
                    'tpr': tpr
                }
                
                # Plot ROC curve
                ax = axes[i//3, i%3]
                ax.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.set_xlabel('1 - Specificity')
                ax.set_ylabel('Sensitivity')
                ax.set_title(f'{name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
            
            # Print results
            print(f"\nClassification Results:")
            for name, result in results.items():
                print(f"{name}: AUC={result['auc']:.3f}, Acc={result['accuracy']:.3f}, "
                      f"Sens={result['sensitivity']:.3f}, Spec={result['specificity']:.3f}")
    
    def _create_regression_models(self, pdf, target_col):
        """Create regression models for radiomics"""
        
        if target_col not in self.data.columns:
            return
        
        y = self.data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        
        print(f"\nRegression Target: {target_col}")
        print(f"Target range: {y.min():.2f} - {y.max():.2f}")
        print(f"Target mean: {y.mean():.2f} ± {y.std():.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Define regression models
        models = {
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'Lasso Regression': Lasso(alpha=0.01, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=False)
        }
        
        # Train and evaluate models
        results = {}
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Radiomics Regression Models', fontsize=16, fontweight='bold')
        
        for i, (name, model) in enumerate(models.items()):
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mae': mae,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            # Plot predictions vs actual
            ax = axes[i//4, i%4]
            ax.scatter(y_test, y_pred, alpha=0.6)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title(f'{name}\nMAE={mae:.3f}, R²={r2:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Print results
        print(f"\nRegression Results:")
        for name, result in results.items():
            print(f"{name}: MAE={result['mae']:.3f}, R²={result['r2']:.3f}")
    
    def _create_feature_selection_analysis(self, pdf, target_col):
        """Create feature selection analysis"""
        
        if target_col not in self.data.columns:
            return
        
        y = self.data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        
        # Create binary target for feature selection
        y_binary = (y <= 2).astype(int) if target_col == 'Last mRS' else y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Selection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Statistical Feature Selection (F-test)
        selector_f = SelectKBest(score_func=f_classif, k=20)
        selector_f.fit(X_train, y_train)
        f_scores = pd.Series(selector_f.scores_, index=X.columns).sort_values(ascending=False)
        
        top_f_features = f_scores.head(15)
        bars = axes[0, 0].bar(range(len(top_f_features)), top_f_features.values, 
                             color='blue', alpha=0.7)
        axes[0, 0].set_xticks(range(len(top_f_features)))
        axes[0, 0].set_xticklabels([f.split('_')[-1] for f in top_f_features.index], rotation=45)
        axes[0, 0].set_ylabel('F-score')
        axes[0, 0].set_title('Top 15 Features (F-test)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        top_rf_features = rf_importance.head(15)
        bars = axes[0, 1].bar(range(len(top_rf_features)), top_rf_features.values, 
                             color='green', alpha=0.7)
        axes[0, 1].set_xticks(range(len(top_rf_features)))
        axes[0, 1].set_xticklabels([f.split('_')[-1] for f in top_rf_features.index], rotation=45)
        axes[0, 1].set_ylabel('Importance')
        axes[0, 1].set_title('Top 15 Features (Random Forest)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. LASSO Feature Selection
        lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, max_iter=1000)
        lasso.fit(X_train, y_train)
        lasso_coef = pd.Series(np.abs(lasso.coef_[0]), index=X.columns).sort_values(ascending=False)
        
        top_lasso_features = lasso_coef.head(15)
        bars = axes[1, 0].bar(range(len(top_lasso_features)), top_lasso_features.values, 
                             color='red', alpha=0.7)
        axes[1, 0].set_xticks(range(len(top_lasso_features)))
        axes[1, 0].set_xticklabels([f.split('_')[-1] for f in top_lasso_features.index], rotation=45)
        axes[1, 0].set_ylabel('|Coefficient|')
        axes[1, 0].set_title('Top 15 Features (LASSO)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature Selection Comparison
        common_features = set(top_f_features.head(10).index) & set(top_rf_features.head(10).index) & set(top_lasso_features.head(10).index)
        
        if common_features:
            axes[1, 1].bar(range(len(common_features)), [1]*len(common_features), 
                          color='purple', alpha=0.7)
            axes[1, 1].set_xticks(range(len(common_features)))
            axes[1, 1].set_xticklabels([f.split('_')[-1] for f in common_features], rotation=45)
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title(f'Common Top Features\n({len(common_features)} features)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
    
    def _create_model_comparison(self, pdf, target_col):
        """Create comprehensive model comparison"""
        
        if target_col not in self.data.columns:
            return
        
        y = self.data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        y_binary = (y <= 2).astype(int) if target_col == 'Last mRS' else y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        # Define models with hyperparameter tuning
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', probability=True, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
        }
        
        # Train models and collect results
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            
            # Final training
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = auc(fpr, tpr)
            
            results[name] = {
                'auc': auc_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'fpr': fpr,
                'tpr': tpr
            }
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. ROC Curves
        for name, result in results.items():
            ax1.plot(result['fpr'], result['tpr'], 
                    label=f'{name} (AUC = {result["auc"]:.3f})', linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax1.set_xlabel('1 - Specificity')
        ax1.set_ylabel('Sensitivity')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance Bar Chart
        model_names = list(results.keys())
        auc_scores = [results[name]['auc'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, auc_scores, width, label='Test AUC', alpha=0.7)
        bars2 = ax2.bar(x + width/2, cv_means, width, label='CV AUC', alpha=0.7, yerr=cv_stds, capsize=5)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('Model Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Print results
        print(f"\nModel Comparison Results:")
        for name, result in results.items():
            print(f"{name}: Test AUC={result['auc']:.3f}, CV AUC={result['cv_mean']:.3f}±{result['cv_std']:.3f}")
    
    def _create_advanced_models(self, pdf, target_col):
        """Create advanced prediction models"""
        
        if target_col not in self.data.columns:
            return
        
        y = self.data[target_col].dropna()
        X = self.features_scaled.loc[y.index]
        y_binary = (y <= 2).astype(int) if target_col == 'Last mRS' else y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Advanced Prediction Models', fontsize=16, fontweight='bold')
        
        # 1. Ensemble Model
        print("\nTraining Ensemble Model...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        # Train individual models
        rf.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)
        
        # Ensemble predictions
        rf_pred = rf.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
        lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
        
        # Average ensemble
        ensemble_pred = (rf_pred + xgb_pred + lgb_pred) / 3
        
        # Calculate ensemble metrics
        fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_pred)
        auc_ens = auc(fpr_ens, tpr_ens)
        
        axes[0, 0].plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC = {auc_ens:.3f})', linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('1 - Specificity')
        axes[0, 0].set_ylabel('Sensitivity')
        axes[0, 0].set_title('Ensemble Model')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Selection with Pipeline
        print("\nTraining Pipeline with Feature Selection...")
        pipeline = Pipeline([
            ('feature_selection', SelectKBest(score_func=f_classif, k=20)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        pipeline_pred = pipeline.predict_proba(X_test)[:, 1]
        
        fpr_pipe, tpr_pipe, _ = roc_curve(y_test, pipeline_pred)
        auc_pipe = auc(fpr_pipe, tpr_pipe)
        
        axes[0, 1].plot(fpr_pipe, tpr_pipe, label=f'Pipeline (AUC = {auc_pipe:.3f})', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('1 - Specificity')
        axes[0, 1].set_ylabel('Sensitivity')
        axes[0, 1].set_title('Pipeline with Feature Selection')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Hyperparameter Tuning
        print("\nTraining with Hyperparameter Tuning...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, None],
            'min_samples_split': [2, 5, 10]
        }
        
        rf_tuned = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf_tuned, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_pred = best_model.predict_proba(X_test)[:, 1]
        
        fpr_best, tpr_best, _ = roc_curve(y_test, best_pred)
        auc_best = auc(fpr_best, tpr_best)
        
        axes[1, 0].plot(fpr_best, tpr_best, label=f'Tuned RF (AUC = {auc_best:.3f})', linewidth=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('1 - Specificity')
        axes[1, 0].set_ylabel('Sensitivity')
        axes[1, 0].set_title('Hyperparameter Tuned Model')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Model Performance Summary
        models_summary = {
            'Ensemble': auc_ens,
            'Pipeline': auc_pipe,
            'Tuned RF': auc_best
        }
        
        model_names = list(models_summary.keys())
        auc_scores = list(models_summary.values())
        
        bars = axes[1, 1].bar(model_names, auc_scores, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('Advanced Models Performance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        print(f"\nAdvanced Models Results:")
        for name, score in models_summary.items():
            print(f"{name}: AUC = {score:.3f}")
    
    def _create_clinical_integration(self, pdf, target_col):
        """Create clinical integration analysis"""
        
        if target_col not in self.data.columns:
            return
        
        # Identify clinical features
        clinical_features = []
        for col in self.data.columns:
            if col not in self.feature_cols and col != target_col and col != 'PatientID':
                if self.data[col].dtype in ['int64', 'float64']:
                    clinical_features.append(col)
        
        if not clinical_features:
            print("No clinical features found for integration")
            return
        
        y = self.data[target_col].dropna()
        X_radiomics = self.features_scaled.loc[y.index]
        X_clinical = self.data.loc[y.index, clinical_features].fillna(0)
        y_binary = (y <= 2).astype(int) if target_col == 'Last mRS' else y
        
        # Split data
        X_train_rad, X_test_rad, X_train_clin, X_test_clin, y_train, y_test = train_test_split(
            X_radiomics, X_clinical, y_binary, test_size=0.25, random_state=42, stratify=y_binary
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Clinical Integration Analysis', fontsize=16, fontweight='bold')
        
        # 1. Radiomics vs Clinical vs Combined
        print("\nComparing Radiomics vs Clinical vs Combined...")
        
        # Radiomics only
        rf_rad = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_rad.fit(X_train_rad, y_train)
        rad_pred = rf_rad.predict_proba(X_test_rad)[:, 1]
        fpr_rad, tpr_rad, _ = roc_curve(y_test, rad_pred)
        auc_rad = auc(fpr_rad, tpr_rad)
        
        # Clinical only
        rf_clin = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clin.fit(X_train_clin, y_train)
        clin_pred = rf_clin.predict_proba(X_test_clin)[:, 1]
        fpr_clin, tpr_clin, _ = roc_curve(y_test, clin_pred)
        auc_clin = auc(fpr_clin, tpr_clin)
        
        # Combined
        X_train_combined = pd.concat([X_train_rad, X_train_clin], axis=1)
        X_test_combined = pd.concat([X_test_rad, X_test_clin], axis=1)
        
        rf_combined = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_combined.fit(X_train_combined, y_train)
        combined_pred = rf_combined.predict_proba(X_test_combined)[:, 1]
        fpr_combined, tpr_combined, _ = roc_curve(y_test, combined_pred)
        auc_combined = auc(fpr_combined, tpr_combined)
        
        axes[0, 0].plot(fpr_rad, tpr_rad, label=f'Radiomics (AUC = {auc_rad:.3f})', linewidth=2)
        axes[0, 0].plot(fpr_clin, tpr_clin, label=f'Clinical (AUC = {auc_clin:.3f})', linewidth=2)
        axes[0, 0].plot(fpr_combined, tpr_combined, label=f'Combined (AUC = {auc_combined:.3f})', linewidth=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('1 - Specificity')
        axes[0, 0].set_ylabel('Sensitivity')
        axes[0, 0].set_title('Radiomics vs Clinical vs Combined')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Feature Importance Comparison
        rad_importance = pd.Series(rf_rad.feature_importances_, index=X_radiomics.columns).sort_values(ascending=False)
        clin_importance = pd.Series(rf_clin.feature_importances_, index=clinical_features).sort_values(ascending=False)
        combined_importance = pd.Series(rf_combined.feature_importances_, index=X_train_combined.columns).sort_values(ascending=False)
        
        # Top radiomics features
        top_rad = rad_importance.head(10)
        bars = axes[0, 1].bar(range(len(top_rad)), top_rad.values, color='blue', alpha=0.7)
        axes[0, 1].set_xticks(range(len(top_rad)))
        axes[0, 1].set_xticklabels([f.split('_')[-1] for f in top_rad.index], rotation=45)
        axes[0, 1].set_ylabel('Importance')
        axes[0, 1].set_title('Top Radiomics Features')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top clinical features
        top_clin = clin_importance.head(10)
        bars = axes[1, 0].bar(range(len(top_clin)), top_clin.values, color='green', alpha=0.7)
        axes[1, 0].set_xticks(range(len(top_clin)))
        axes[1, 0].set_xticklabels(top_clin.index, rotation=45)
        axes[1, 0].set_ylabel('Importance')
        axes[1, 0].set_title('Top Clinical Features')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance comparison
        models = ['Radiomics', 'Clinical', 'Combined']
        auc_scores = [auc_rad, auc_clin, auc_combined]
        
        bars = axes[1, 1].bar(models, auc_scores, color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('Performance Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        print(f"\nClinical Integration Results:")
        print(f"Radiomics only: AUC = {auc_rad:.3f}")
        print(f"Clinical only: AUC = {auc_clin:.3f}")
        print(f"Combined: AUC = {auc_combined:.3f}")

def main():
    """Main function to run comprehensive radiomics prediction models"""
    print("=== COMPREHENSIVE RADIOMICS PREDICTION MODELS ===")
    print("Creating state-of-the-art prediction models...\n")
    
    # Initialize predictor with your data
    predictor = ComprehensiveRadiomicsPredictor('results/radiomics_lastmrs_mapping.csv')
    
    # Create comprehensive prediction models
    predictor.create_prediction_models('Last mRS', 'comprehensive_radiomics_predictions.pdf')
    
    print("\n=== PREDICTION MODELS COMPLETED ===")
    print("Generated models include:")
    print("1. Classification Models (Logistic, RF, SVM, XGBoost, LightGBM, CatBoost)")
    print("2. Regression Models (Ridge, Lasso, RF, SVR, XGBoost, LightGBM, CatBoost)")
    print("3. Feature Selection Analysis (F-test, RF, LASSO)")
    print("4. Model Comparison and Cross-validation")
    print("5. Advanced Models (Ensemble, Pipeline, Hyperparameter Tuning)")
    print("6. Clinical Integration Analysis")
    print("\nBest practices implemented:")
    print("- Multiple model types for comprehensive comparison")
    print("- Feature selection and importance analysis")
    print("- Cross-validation and hyperparameter tuning")
    print("- Ensemble methods for improved performance")
    print("- Clinical integration for real-world applicability")

if __name__ == "__main__":
    main() 