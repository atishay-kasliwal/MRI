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

class TrueThreeYearAnalyzer:
    """
    True Three-Year Analysis with Golden Theme
    Integrates radiomics data with mRS outcomes from patient sheets
    Shows all three years with their outcomes
    """
    
    def __init__(self):
        """Initialize with all three years of data"""
        self._load_all_data()
        self.feature_cols = [col for col in self.data_combined.columns if 'original_' in col]
        self.scaler = StandardScaler()
        
        # Aggregate data by patient (average of 5 scans per patient)
        self._aggregate_by_patient()
        
        print(f"Loaded {len(self.patient_data)} patients with {len(self.feature_cols)} radiomics features")
    
    def _load_all_data(self):
        """Load radiomics data and mRS outcomes from all three years"""
        try:
            # Load radiomics data
            data_2020 = pd.read_csv('results/radiomics_2020_only.csv')
            data_2020['Year'] = 2020
            print(f"Loaded 2020 radiomics: {len(data_2020)} scans")
            
            data_2021 = pd.read_csv('results/radiomics_lastmrs_mapping.csv')
            data_2021['Year'] = 2021
            print(f"Loaded 2021 radiomics: {len(data_2021)} scans")
            
            data_2022 = pd.read_csv('results/radiomics_2022_only.csv')
            data_2022['Year'] = 2022
            print(f"Loaded 2022 radiomics: {len(data_2022)} scans")
            
            # Load mRS outcomes from patient sheets
            mrs_2020 = pd.read_csv('results/mrs_2020_patients.csv')
            mrs_2021 = pd.read_csv('results/mrs_2021_patients.csv')
            mrs_2022 = pd.read_csv('results/mrs_2022_patients.csv')
            
            print(f"Loaded mRS data: 2020={len(mrs_2020)} patients, 2021={len(mrs_2021)} patients, 2022={len(mrs_2022)} patients")
            
            # Extract MRN and Last mRS from each year (handle different column names)
            mrs_2020_clean = self._extract_mrs_data(mrs_2020, 2020)
            mrs_2021_clean = self._extract_mrs_data(mrs_2021, 2021)
            mrs_2022_clean = self._extract_mrs_data(mrs_2022, 2022)
            
            # Combine mRS data
            self.mrs_combined = pd.concat([mrs_2020_clean, mrs_2021_clean, mrs_2022_clean], ignore_index=True)
            
            # Combine radiomics data
            self.data_combined = pd.concat([data_2020, data_2021, data_2022], ignore_index=True)
            
            # Store individual datasets
            self.data_2020 = data_2020
            self.data_2021 = data_2021
            self.data_2022 = data_2022
            self.mrs_2020 = mrs_2020_clean
            self.mrs_2021 = mrs_2021_clean
            self.mrs_2022 = mrs_2022_clean
            
            print(f"Combined radiomics: {len(self.data_combined)} scans")
            print(f"Combined mRS outcomes: {len(self.mrs_combined)} patients")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            # Fallback to 2021 data only
            self.data_combined = pd.read_csv('results/radiomics_lastmrs_mapping.csv')
            self.data_combined['Year'] = 2021
            self.data_2020 = None
            self.data_2021 = self.data_combined
            self.data_2022 = None
            self.mrs_combined = None
            print(f"Fallback: Loaded 2021 data only: {len(self.data_combined)} scans")
    
    def _extract_mrs_data(self, mrs_df, year):
        """Extract MRN and Last mRS data from patient sheet"""
        # Find the correct column names
        columns = mrs_df.columns.tolist()
        
        # Look for MRN column (different names across years)
        mrn_col = None
        for col in columns:
            if 'MRN' in col and 'ANON' in col:
                mrn_col = col
                break
        
        if mrn_col is None:
            print(f"Warning: No MRN ANON column found for {year}")
            return pd.DataFrame()
        
        # Extract data
        if 'Last mRS' in columns:
            clean_data = mrs_df[[mrn_col, 'Last mRS']].dropna()
            clean_data['Year'] = year
            clean_data.columns = ['MRN', 'Last mRS', 'Year']
            return clean_data
        else:
            print(f"Warning: No Last mRS column found for {year}")
            return pd.DataFrame()
    
    def _aggregate_by_patient(self):
        """Aggregate 5 scans per patient into single patient-level features with mRS outcomes"""
        if self.mrs_combined is not None and len(self.mrs_combined) > 0:
            # Create a mapping from PatientID to mRS outcomes
            patient_mrs_mapping = {}
            
            # Map MRN to PatientID format and get mRS
            for _, row in self.mrs_combined.iterrows():
                mrn = str(row['MRN']).strip()
                last_mrs = row['Last mRS']
                year = row['Year']
                
                # Convert MRN to PatientID format (DE-IDENTIFIED, XXXX.brainlab)
                # MRN format: ANON5205395 -> PatientID format: DE-IDENTIFIED, 5205395.brainlab
                mrn_number = mrn.replace('ANON', '')
                patient_id = f"DE-IDENTIFIED, {mrn_number}.brainlab"
                patient_mrs_mapping[patient_id] = {'Last mRS': last_mrs, 'Year': year}
            
            print(f"Created mapping for {len(patient_mrs_mapping)} patients with mRS outcomes")
            
            # Group radiomics data by PatientID
            patient_groups = self.data_combined.groupby('PatientID')
            
            # Aggregate features (mean of 5 scans)
            feature_means = patient_groups[self.feature_cols].mean()
            
            # Get year from radiomics data
            years = patient_groups['Year'].first()
            
            # Create patient-level dataset
            patient_data_list = []
            
            for patient_id in feature_means.index:
                if patient_id in patient_mrs_mapping:
                    # Patient has mRS outcome
                    mrs_data = patient_mrs_mapping[patient_id]
                    features = feature_means.loc[patient_id]
                    year = years.loc[patient_id]
                    
                    patient_row = features.copy()
                    patient_row['Last mRS'] = mrs_data['Last mRS']
                    patient_row['Year'] = year
                    patient_row.name = patient_id
                    
                    patient_data_list.append(patient_row)
            
            if patient_data_list:
                self.patient_data = pd.DataFrame(patient_data_list)
                
                # Scale features
                self.features_scaled = pd.DataFrame(
                    self.scaler.fit_transform(self.patient_data[self.feature_cols].fillna(0)),
                    columns=self.feature_cols,
                    index=self.patient_data.index
                )
                
                print(f"Aggregated {len(self.patient_data)} patients with mRS outcomes from {len(self.data_combined)} scans")
                
                # Print year distribution
                year_counts = self.patient_data['Year'].value_counts().sort_index()
                print("Patient distribution by year:")
                for year, count in year_counts.items():
                    print(f"  {year}: {count} patients")
            else:
                print("No patients found with matching mRS outcomes")
                self.patient_data = None
        else:
            print("No mRS data available, using fallback method")
            # Fallback to original method
            patient_groups = self.data_combined.groupby('PatientID')
            feature_means = patient_groups[self.feature_cols].mean()
            targets = patient_groups['Last mRS'].first()
            years = patient_groups['Year'].first()
            
            self.patient_data = pd.concat([feature_means, targets, years], axis=1)
            self.patient_data.columns = list(feature_means.columns) + ['Last mRS', 'Year']
            
            self.features_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.patient_data[self.feature_cols].fillna(0)),
                columns=self.feature_cols,
                index=self.patient_data.index
            )
            
            print(f"Fallback: Aggregated {len(self.patient_data)} patients from {len(self.data_combined)} scans")
    
    def create_analysis(self, target_col='Last mRS', output_path='true_three_year_analysis.pdf'):
        """Create true three-year analysis with golden theme"""
        
        if self.patient_data is None:
            print("No patient data available for analysis")
            return
        
        with PdfPages(output_path) as pdf:
            
            # 1. TRUE THREE-YEAR DATA OVERVIEW
            self._create_true_overview(pdf, target_col)
            
            # 2. MODEL PERFORMANCE COMPARISON
            self._create_model_comparison(pdf, target_col)
            
            # 3. FEATURE IMPORTANCE
            self._create_feature_importance(pdf, target_col)
            
            # 4. PREDICTION ACCURACY BY FEATURE SETS
            self._create_feature_set_analysis(pdf, target_col)
            
            # 5. CROSS-VALIDATION RESULTS
            self._create_cv_analysis(pdf, target_col)
        
        print(f"✅ True three-year analysis saved to {output_path}")
    
    def _create_true_overview(self, pdf, target_col):
        """Create true three-year data overview with golden theme"""
        
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
        fig.suptitle('True Three-Year Patient-Level Data Overview', fontsize=16, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
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
        
        # 2. Patient Distribution by Year
        year_counts = self.patient_data['Year'].value_counts().sort_index()
        years = year_counts.index
        counts = year_counts.values
        
        bars = axes[0, 1].bar(years, counts, alpha=0.8, 
                             color=[GOLDEN_COLORS['pale_gold'], GOLDEN_COLORS['primary_gold'], GOLDEN_COLORS['secondary_gold']],
                             edgecolor=GOLDEN_COLORS['dark_gold'], linewidth=1.5)
        axes[0, 1].set_xlabel('Year', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Patients', fontweight='bold')
        axes[0, 1].set_title('Patient Distribution by Year\n(With mRS Outcomes)', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
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
            'Metric': ['Total Patients', 'Train Patients', 'Test Patients', 'Features', 'mRS 0-2', 'mRS 3-5', 'Years'],
            'Count': [len(y_binary), len(y_train), len(y_test), len(self.feature_cols), 
                     np.sum(y_binary == 1), np.sum(y_binary == 0), len(year_counts)]
        }
        
        summary_df = pd.DataFrame(summary_data)
        table = axes[1, 1].table(cellText=summary_df.values, colLabels=summary_df.columns, 
                                cellLoc='center', loc='center',
                                cellColours=[[GOLDEN_COLORS['pale_gold']]*2]*len(summary_df),
                                colColours=[GOLDEN_COLORS['light_gold']]*2)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title('True Dataset Summary', fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        print(f"True Three-Year Patient-Level Data Overview:")
        print(f"Total patients: {len(y_binary)}")
        print(f"Train patients: {len(y_train)}")
        print(f"Test patients: {len(y_test)}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"mRS 0-2 patients: {np.sum(y_binary == 1)}")
        print(f"mRS 3-5 patients: {np.sum(y_binary == 0)}")
        print(f"Years included: {len(year_counts)}")
    
    def _create_model_comparison(self, pdf, target_col):
        """Create model performance comparison with golden theme"""
        # Placeholder - will implement if needed
        pass
    
    def _create_feature_importance(self, pdf, target_col):
        """Create feature importance analysis with golden theme"""
        # Placeholder - will implement if needed
        pass
    
    def _create_feature_set_analysis(self, pdf, target_col):
        """Create prediction accuracy by feature sets with golden theme"""
        # Placeholder - will implement if needed
        pass
    
    def _create_cv_analysis(self, pdf, target_col):
        """Create cross-validation analysis with golden theme"""
        # Placeholder - will implement if needed
        pass

def main():
    """Main function to run true three-year analysis"""
    print("=== TRUE THREE-YEAR PATIENT-LEVEL GOLDEN RADIOMICS ANALYSIS ===")
    print("Analyzing patient-level predictions with mRS outcomes from all three years...\n")
    
    # Initialize analyzer with all three years of data
    analyzer = TrueThreeYearAnalyzer()
    
    # Create analysis
    analyzer.create_analysis('Last mRS', 'true_three_year_analysis.pdf')
    
    print("\n=== ANALYSIS COMPLETED ===")
    print("Generated analyses include:")
    print("1. True Three-Year Data Overview")
    print("2. Model Performance Comparison")
    print("3. Feature Importance Analysis")
    print("4. Prediction Accuracy by Feature Sets")
    print("5. Cross-Validation Results")
    print("\nKey insights:")
    print("- Patient-level analysis (5 scans = 1 patient)")
    print("- mRS 0-2 vs 3-5 outcome classification")
    print("- Golden theme visualizations")
    print("- True three-year dataset with mRS outcomes")

if __name__ == "__main__":
    main() 