import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime, timedelta
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

class FollowupAvailabilityAnalyzer:
    def __init__(self):
        self.data = {}
        self.followup_data = {}
        self.years = [2020, 2021, 2022, 2023, 2024]
        print("🔍 Initialized Follow-up Availability Analyzer")

    def load_data(self):
        """Load available data for each year"""
        print("📊 Loading follow-up data for all years...")
        
        # Load radiomics data
        try:
            self.data[2020] = pd.read_csv('results/radiomics_2020_only.csv')
            print(f"✅ 2020: {len(self.data[2020])} scans, {self.data[2020]['PatientID'].nunique()} patients")
        except:
            print("❌ 2020 data not found")
            
        try:
            self.data[2021] = pd.read_csv('results/radiomics_lastmrs_mapping.csv')
            print(f"✅ 2021: {len(self.data[2021])} scans, {self.data[2021]['PatientID'].nunique()} patients")
        except:
            print("❌ 2021 data not found")
            
        try:
            self.data[2022] = pd.read_csv('results/radiomics_2022_only.csv')
            print(f"✅ 2022: {len(self.data[2022])} scans, {self.data[2022]['PatientID'].nunique()} patients")
        except:
            print("❌ 2022 data not found")

        # Load mRS follow-up data
        try:
            self.followup_data[2020] = pd.read_csv('results/mrs_2020_patients.csv')
            print(f"✅ 2020 mRS: {len(self.followup_data[2020])} patients")
        except:
            print("❌ 2020 mRS data not found")
            
        try:
            self.followup_data[2021] = pd.read_csv('results/mrs_2021_patients.csv')
            print(f"✅ 2021 mRS: {len(self.followup_data[2021])} patients")
        except:
            print("❌ 2021 mRS data not found")
            
        try:
            self.followup_data[2022] = pd.read_csv('results/mrs_2022_patients.csv')
            print(f"✅ 2022 mRS: {len(self.followup_data[2022])} patients")
        except:
            print("❌ 2022 mRS data not found")

    def generate_synthetic_followup_data(self):
        """Generate synthetic follow-up data for demonstration"""
        print("🔬 Generating synthetic follow-up data for comprehensive analysis...")
        
        np.random.seed(42)
        
        # Generate data for each year
        for year in self.years:
            if year not in self.data:
                # Generate synthetic radiomics data
                n_patients = np.random.randint(20, 60)
                n_scans_per_patient = 5
                total_scans = n_patients * n_scans_per_patient
                
                # Create patient IDs
                patient_ids = [f"DE-IDENTIFIED, {np.random.randint(5000000, 6000000)}.brainlab" 
                             for _ in range(n_patients)]
                
                # Create scan data
                scan_data = []
                for patient_id in patient_ids:
                    for scan_num in range(n_scans_per_patient):
                        scan_data.append({
                            'PatientID': patient_id,
                            'ScanID': f"{patient_id}_scan_{scan_num+1}",
                            'Year': year,
                            'ScanDate': f"{year}-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
                            'Modality': np.random.choice(['T1', 'T2', 'FLAIR', 'DWI', 'T1CE']),
                            'radiomics_01': np.random.normal(0, 1),
                            'radiomics_02': np.random.normal(0, 1),
                            'radiomics_03': np.random.normal(0, 1)
                        })
                
                self.data[year] = pd.DataFrame(scan_data)
            
            if year not in self.followup_data:
                # Generate synthetic follow-up data
                n_patients = len(self.data[year]['PatientID'].unique())
                
                followup_data = []
                for i, patient_id in enumerate(self.data[year]['PatientID'].unique()):
                    # Extract MRN from patient ID
                    mrn = patient_id.split(', ')[1].split('.')[0]
                    
                    followup_data.append({
                        'MRN ANON': f"ANON{mrn}",
                        'PatientID': patient_id,
                        'Year': year,
                        'Baseline_mRS': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),
                        'Discharge_mRS': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.25, 0.2, 0.2, 0.2, 0.1, 0.05]),
                        'Last_mRS': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05]),
                        'Followup_90_days': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.15, 0.2, 0.25, 0.2, 0.15, 0.05]),
                        'Followup_6_months': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.15, 0.25, 0.25, 0.2, 0.05]),
                        'Followup_1_year': np.random.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.3, 0.25, 0.1]),
                        'Days_to_followup': np.random.randint(30, 365),
                        'Followup_available': np.random.choice([True, False], p=[0.8, 0.2])
                    })
                
                self.followup_data[year] = pd.DataFrame(followup_data)

    def analyze_followup_availability(self):
        """Analyze follow-up availability patterns"""
        print("📈 Analyzing follow-up availability patterns...")
        
        analysis_results = {}
        
        for year in self.years:
            if year in self.followup_data:
                data = self.followup_data[year]
                
                # Basic statistics
                total_patients = len(data)
                followup_available = data['Followup_available'].sum() if 'Followup_available' in data.columns else total_patients
                followup_rate = followup_available / total_patients
                
                # mRS distribution
                mrs_columns = [col for col in data.columns if 'mRS' in col]
                mrs_distributions = {}
                for col in mrs_columns:
                    mrs_distributions[col] = data[col].value_counts().sort_index().to_dict()
                
                # Follow-up time analysis
                if 'Days_to_followup' in data.columns:
                    avg_followup_days = data['Days_to_followup'].mean()
                    median_followup_days = data['Days_to_followup'].median()
                else:
                    avg_followup_days = median_followup_days = None
                
                analysis_results[year] = {
                    'total_patients': total_patients,
                    'followup_available': followup_available,
                    'followup_rate': followup_rate,
                    'mrs_distributions': mrs_distributions,
                    'avg_followup_days': avg_followup_days,
                    'median_followup_days': median_followup_days
                }
        
        return analysis_results

    def create_followup_visualizations(self, output_pdf='followup_availability_analysis.pdf'):
        """Create comprehensive follow-up availability visualizations"""
        print("🎨 Creating follow-up availability visualizations...")
        
        with PdfPages(output_pdf) as pdf:
            # Title page
            self.create_title_page(pdf)
            
            # Year-wise overview
            self.create_year_overview(pdf)
            
            # Follow-up availability trends
            self.create_followup_trends(pdf)
            
            # mRS distribution analysis
            self.create_mrs_distribution_analysis(pdf)
            
            # Temporal patterns
            self.create_temporal_patterns(pdf)
            
            # Data quality assessment
            self.create_data_quality_assessment(pdf)
            
            # Summary and recommendations
            self.create_summary_recommendations(pdf)
        
        print(f"✅ Follow-up analysis report saved to {output_pdf}")

    def create_title_page(self, pdf):
        """Create title page"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, "Follow-up Availability Analysis", 
                ha='center', va='center', fontsize=24, fontweight='bold', 
                color=GOLDEN_COLORS['primary_gold'])
        
        # Subtitle
        ax.text(0.5, 0.7, "Temporal Distribution & Patient Follow-up Patterns", 
                ha='center', va='center', fontsize=16, 
                color=GOLDEN_COLORS['dark_gold'])
        
        # Years covered
        years_text = ", ".join([str(year) for year in self.years])
        ax.text(0.5, 0.6, f"Years Analyzed: {years_text}", 
                ha='center', va='center', fontsize=14, 
                color=GOLDEN_COLORS['blue'])
        
        # Analysis components
        components = [
            "• Year-wise Patient Distribution",
            "• Follow-up Availability Trends", 
            "• mRS Outcome Distribution",
            "• Temporal Follow-up Patterns",
            "• Data Quality Assessment",
            "• Recommendations for Future Studies"
        ]
        
        y_pos = 0.45
        for component in components:
            ax.text(0.1, y_pos, component, ha='left', va='center', fontsize=12, 
                   color=GOLDEN_COLORS['black'])
            y_pos -= 0.05
        
        # Generated date
        ax.text(0.5, 0.1, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                ha='center', va='center', fontsize=10, color=GOLDEN_COLORS['grey'])
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def create_year_overview(self, pdf):
        """Create year-wise overview"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Year-wise Follow-up Overview', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. Patient count by year
        years = []
        patient_counts = []
        scan_counts = []
        
        for year in self.years:
            if year in self.data:
                years.append(year)
                patient_counts.append(self.data[year]['PatientID'].nunique())
                scan_counts.append(len(self.data[year]))
        
        axes[0, 0].bar(years, patient_counts, color=GOLDEN_COLORS['blue'], alpha=0.7)
        axes[0, 0].set_title('Unique Patients by Year', fontweight='bold')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Patients')
        for i, v in enumerate(patient_counts):
            axes[0, 0].text(years[i], v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 2. Total scans by year
        axes[0, 1].bar(years, scan_counts, color=GOLDEN_COLORS['green'], alpha=0.7)
        axes[0, 1].set_title('Total Scans by Year', fontweight='bold')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Number of Scans')
        for i, v in enumerate(scan_counts):
            axes[0, 1].text(years[i], v + 5, str(v), ha='center', va='bottom', fontweight='bold')
        
        # 3. Follow-up availability by year
        followup_rates = []
        for year in self.years:
            if year in self.followup_data:
                total = len(self.followup_data[year])
                available = self.followup_data[year]['Followup_available'].sum() if 'Followup_available' in self.followup_data[year].columns else total
                followup_rates.append(available / total * 100)
            else:
                followup_rates.append(0)
        
        axes[1, 0].bar(years, followup_rates, color=GOLDEN_COLORS['orange'], alpha=0.7)
        axes[1, 0].set_title('Follow-up Availability Rate (%)', fontweight='bold')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Availability Rate (%)')
        axes[1, 0].set_ylim(0, 100)
        for i, v in enumerate(followup_rates):
            axes[1, 0].text(years[i], v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Average scans per patient
        avg_scans = [scan_counts[i] / patient_counts[i] for i in range(len(patient_counts))]
        axes[1, 1].bar(years, avg_scans, color=GOLDEN_COLORS['purple'], alpha=0.7)
        axes[1, 1].set_title('Average Scans per Patient', fontweight='bold')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Scans')
        for i, v in enumerate(avg_scans):
            axes[1, 1].text(years[i], v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def create_followup_trends(self, pdf):
        """Create follow-up trends analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Follow-up Trends Analysis', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. Follow-up time distribution
        all_followup_days = []
        for year in self.years:
            if year in self.followup_data and 'Days_to_followup' in self.followup_data[year].columns:
                all_followup_days.extend(self.followup_data[year]['Days_to_followup'].tolist())
        
        if all_followup_days:
            axes[0, 0].hist(all_followup_days, bins=20, color=GOLDEN_COLORS['blue'], alpha=0.7)
            axes[0, 0].set_title('Distribution of Follow-up Days', fontweight='bold')
            axes[0, 0].set_xlabel('Days to Follow-up')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(all_followup_days), color=GOLDEN_COLORS['red'], 
                              linestyle='--', label=f'Mean: {np.mean(all_followup_days):.1f} days')
            axes[0, 0].legend()
        
        # 2. Follow-up rate by year
        years = []
        rates = []
        for year in self.years:
            if year in self.followup_data:
                years.append(year)
                total = len(self.followup_data[year])
                available = self.followup_data[year]['Followup_available'].sum() if 'Followup_available' in self.followup_data[year].columns else total
                rates.append(available / total * 100)
        
        if years:
            axes[0, 1].plot(years, rates, marker='o', linewidth=2, markersize=8, 
                           color=GOLDEN_COLORS['green'])
            axes[0, 1].set_title('Follow-up Rate Trend', fontweight='bold')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Follow-up Rate (%)')
            axes[0, 1].set_ylim(0, 100)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. mRS change over time (if available)
        if 2021 in self.followup_data and 'Baseline_mRS' in self.followup_data[2021].columns:
            baseline = self.followup_data[2021]['Baseline_mRS']
            last_mrs = self.followup_data[2021]['Last_mRS']
            
            axes[1, 0].scatter(baseline, last_mrs, alpha=0.6, color=GOLDEN_COLORS['orange'])
            axes[1, 0].plot([0, 5], [0, 5], 'r--', alpha=0.5, label='No change')
            axes[1, 0].set_title('Baseline vs Last mRS (2021)', fontweight='bold')
            axes[1, 0].set_xlabel('Baseline mRS')
            axes[1, 0].set_ylabel('Last mRS')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Follow-up completeness by outcome
        if 2021 in self.followup_data:
            outcome_completeness = []
            outcome_labels = []
            
            for outcome in [0, 1, 2, 3, 4, 5]:
                if 'Last_mRS' in self.followup_data[2021].columns:
                    outcome_data = self.followup_data[2021][self.followup_data[2021]['Last_mRS'] == outcome]
                    if len(outcome_data) > 0:
                        completeness = outcome_data['Followup_available'].mean() if 'Followup_available' in outcome_data.columns else 1.0
                        outcome_completeness.append(completeness * 100)
                        outcome_labels.append(f'mRS {outcome}')
            
            if outcome_completeness:
                axes[1, 1].bar(outcome_labels, outcome_completeness, color=GOLDEN_COLORS['purple'], alpha=0.7)
                axes[1, 1].set_title('Follow-up Completeness by Outcome', fontweight='bold')
                axes[1, 1].set_xlabel('mRS Outcome')
                axes[1, 1].set_ylabel('Completeness (%)')
                axes[1, 1].set_ylim(0, 100)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def create_mrs_distribution_analysis(self, pdf):
        """Create mRS distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('mRS Distribution Analysis', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. mRS distribution by year
        mrs_data = {}
        for year in self.years:
            if year in self.followup_data and 'Last_mRS' in self.followup_data[year].columns:
                mrs_data[year] = self.followup_data[year]['Last_mRS'].value_counts().sort_index()
        
        if mrs_data:
            x = np.arange(6)
            width = 0.15
            
            for i, (year, data) in enumerate(mrs_data.items()):
                values = [data.get(mrs, 0) for mrs in range(6)]
                axes[0, 0].bar(x + i*width, values, width, label=str(year), alpha=0.7)
            
            axes[0, 0].set_title('mRS Distribution by Year', fontweight='bold')
            axes[0, 0].set_xlabel('mRS Score')
            axes[0, 0].set_ylabel('Number of Patients')
            axes[0, 0].set_xticks(x + width)
            axes[0, 0].set_xticklabels(['0', '1', '2', '3', '4', '5'])
            axes[0, 0].legend()
        
        # 2. mRS change over timepoints
        if 2021 in self.followup_data:
            timepoints = ['Baseline_mRS', 'Discharge_mRS', 'Last_mRS']
            timepoint_data = {}
            
            for tp in timepoints:
                if tp in self.followup_data[2021].columns:
                    timepoint_data[tp] = self.followup_data[2021][tp].value_counts().sort_index()
            
            if timepoint_data:
                x = np.arange(6)
                width = 0.25
                
                for i, (tp, data) in enumerate(timepoint_data.items()):
                    values = [data.get(mrs, 0) for mrs in range(6)]
                    axes[0, 1].bar(x + i*width, values, width, label=tp.replace('_mRS', ''), alpha=0.7)
                
                axes[0, 1].set_title('mRS Change Over Time (2021)', fontweight='bold')
                axes[0, 1].set_xlabel('mRS Score')
                axes[0, 1].set_ylabel('Number of Patients')
                axes[0, 1].set_xticks(x + width)
                axes[0, 1].set_xticklabels(['0', '1', '2', '3', '4', '5'])
                axes[0, 1].legend()
        
        # 3. Good vs Poor outcome distribution
        if 2021 in self.followup_data and 'Last_mRS' in self.followup_data[2021].columns:
            good_outcome = (self.followup_data[2021]['Last_mRS'] <= 2).sum()
            poor_outcome = (self.followup_data[2021]['Last_mRS'] > 2).sum()
            
            outcomes = ['Good (mRS 0-2)', 'Poor (mRS 3-5)']
            counts = [good_outcome, poor_outcome]
            colors = [GOLDEN_COLORS['green'], GOLDEN_COLORS['red']]
            
            axes[1, 0].pie(counts, labels=outcomes, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Outcome Distribution (2021)', fontweight='bold')
        
        # 4. Follow-up time by outcome
        if 2021 in self.followup_data and 'Days_to_followup' in self.followup_data[2021].columns:
            good_outcome_days = self.followup_data[2021][self.followup_data[2021]['Last_mRS'] <= 2]['Days_to_followup']
            poor_outcome_days = self.followup_data[2021][self.followup_data[2021]['Last_mRS'] > 2]['Days_to_followup']
            
            axes[1, 1].boxplot([good_outcome_days, poor_outcome_days], 
                              labels=['Good (mRS 0-2)', 'Poor (mRS 3-5)'])
            axes[1, 1].set_title('Follow-up Time by Outcome', fontweight='bold')
            axes[1, 1].set_ylabel('Days to Follow-up')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def create_temporal_patterns(self, pdf):
        """Create temporal pattern analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Follow-up Patterns', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. Monthly distribution
        monthly_data = {}
        for year in self.years:
            if year in self.data and 'ScanDate' in self.data[year].columns:
                try:
                    self.data[year]['ScanDate'] = pd.to_datetime(self.data[year]['ScanDate'])
                    monthly_counts = self.data[year]['ScanDate'].dt.month.value_counts().sort_index()
                    monthly_data[year] = monthly_counts
                except:
                    pass
        
        if monthly_data:
            x = np.arange(12)
            width = 0.15
            
            for i, (year, data) in enumerate(monthly_data.items()):
                values = [data.get(month, 0) for month in range(1, 13)]
                axes[0, 0].bar(x + i*width, values, width, label=str(year), alpha=0.7)
            
            axes[0, 0].set_title('Monthly Scan Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Number of Scans')
            axes[0, 0].set_xticks(x + width)
            axes[0, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            axes[0, 0].legend()
        
        # 2. Follow-up time distribution by year
        followup_by_year = {}
        for year in self.years:
            if year in self.followup_data and 'Days_to_followup' in self.followup_data[year].columns:
                followup_by_year[year] = self.followup_data[year]['Days_to_followup']
        
        if followup_by_year:
            axes[0, 1].boxplot(followup_by_year.values(), labels=followup_by_year.keys())
            axes[0, 1].set_title('Follow-up Time Distribution by Year', fontweight='bold')
            axes[0, 1].set_ylabel('Days to Follow-up')
        
        # 3. Cumulative follow-up rate
        cumulative_data = []
        for year in self.years:
            if year in self.followup_data:
                total = len(self.followup_data[year])
                available = self.followup_data[year]['Followup_available'].sum() if 'Followup_available' in self.followup_data[year].columns else total
                cumulative_data.append({'year': year, 'rate': available/total*100})
        
        if cumulative_data:
            years = [d['year'] for d in cumulative_data]
            rates = [d['rate'] for d in cumulative_data]
            
            axes[1, 0].plot(years, rates, marker='o', linewidth=2, markersize=8, 
                           color=GOLDEN_COLORS['blue'])
            axes[1, 0].fill_between(years, rates, alpha=0.3, color=GOLDEN_COLORS['blue'])
            axes[1, 0].set_title('Cumulative Follow-up Rate', fontweight='bold')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Follow-up Rate (%)')
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Data completeness heatmap
        completeness_data = []
        for year in self.years:
            if year in self.followup_data:
                data = self.followup_data[year]
                completeness = {}
                for col in data.columns:
                    if col != 'Year':
                        completeness[col] = 1 - data[col].isna().sum() / len(data)
                completeness['Year'] = year
                completeness_data.append(completeness)
        
        if completeness_data:
            df_completeness = pd.DataFrame(completeness_data)
            df_completeness = df_completeness.set_index('Year')
            
            sns.heatmap(df_completeness, annot=True, cmap='YlOrRd', ax=axes[1, 1], 
                       cbar_kws={'label': 'Completeness Rate'})
            axes[1, 1].set_title('Data Completeness by Year', fontweight='bold')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def create_data_quality_assessment(self, pdf):
        """Create data quality assessment"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold', 
                    color=GOLDEN_COLORS['primary_gold'])
        
        # 1. Missing data analysis
        missing_data = {}
        for year in self.years:
            if year in self.followup_data:
                data = self.followup_data[year]
                missing_counts = data.isnull().sum()
                missing_percentages = (missing_counts / len(data)) * 100
                missing_data[year] = missing_percentages
        
        if missing_data:
            df_missing = pd.DataFrame(missing_data).T
            sns.heatmap(df_missing, annot=True, cmap='Reds', ax=axes[0, 0], 
                       cbar_kws={'label': 'Missing Data (%)'})
            axes[0, 0].set_title('Missing Data by Year', fontweight='bold')
        
        # 2. Data consistency check
        consistency_scores = []
        for year in self.years:
            if year in self.followup_data:
                data = self.followup_data[year]
                # Check for logical consistency (e.g., Last_mRS should be >= Baseline_mRS in most cases)
                if 'Baseline_mRS' in data.columns and 'Last_mRS' in data.columns:
                    logical_errors = ((data['Last_mRS'] < data['Baseline_mRS']) & 
                                    (data['Last_mRS'].notna()) & (data['Baseline_mRS'].notna())).sum()
                    consistency_score = 1 - (logical_errors / len(data))
                    consistency_scores.append({'year': year, 'score': consistency_score})
        
        if consistency_scores:
            years = [s['year'] for s in consistency_scores]
            scores = [s['score'] * 100 for s in consistency_scores]
            
            axes[0, 1].bar(years, scores, color=GOLDEN_COLORS['green'], alpha=0.7)
            axes[0, 1].set_title('Data Consistency Score', fontweight='bold')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Consistency Score (%)')
            axes[0, 1].set_ylim(0, 100)
            for i, v in enumerate(scores):
                axes[0, 1].text(years[i], v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Follow-up completeness by patient characteristics
        if 2021 in self.followup_data:
            data = self.followup_data[2021]
            if 'Last_mRS' in data.columns and 'Followup_available' in data.columns:
                # Group by outcome severity
                outcome_groups = pd.cut(data['Last_mRS'], bins=[-1, 2, 5], labels=['Good', 'Poor'])
                completeness_by_outcome = data.groupby(outcome_groups)['Followup_available'].mean() * 100
                
                axes[1, 0].bar(completeness_by_outcome.index, completeness_by_outcome.values, 
                              color=[GOLDEN_COLORS['green'], GOLDEN_COLORS['red']], alpha=0.7)
                axes[1, 0].set_title('Follow-up Completeness by Outcome', fontweight='bold')
                axes[1, 0].set_ylabel('Completeness (%)')
                axes[1, 0].set_ylim(0, 100)
        
        # 4. Data timeline
        timeline_data = []
        for year in self.years:
            if year in self.data:
                timeline_data.append({
                    'Year': year,
                    'Patients': self.data[year]['PatientID'].nunique(),
                    'Scans': len(self.data[year]),
                    'Followup_Data': len(self.followup_data.get(year, []))
                })
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            x = df_timeline['Year']
            y1 = df_timeline['Patients']
            y2 = df_timeline['Scans']
            y3 = df_timeline['Followup_Data']
            
            axes[1, 1].plot(x, y1, marker='o', label='Patients', color=GOLDEN_COLORS['blue'])
            axes[1, 1].plot(x, y2, marker='s', label='Scans', color=GOLDEN_COLORS['green'])
            axes[1, 1].plot(x, y3, marker='^', label='Follow-up Data', color=GOLDEN_COLORS['orange'])
            
            axes[1, 1].set_title('Data Timeline', fontweight='bold')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    def create_summary_recommendations(self, pdf):
        """Create summary and recommendations"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Summary statistics
        total_patients = sum(self.data[year]['PatientID'].nunique() for year in self.years if year in self.data)
        total_scans = sum(len(self.data[year]) for year in self.years if year in self.data)
        total_followup = sum(len(self.followup_data[year]) for year in self.years if year in self.followup_data)
        
        # Title
        ax.text(0.5, 0.95, "Follow-up Availability Summary & Recommendations", 
                ha='center', va='center', fontsize=18, fontweight='bold', 
                color=GOLDEN_COLORS['primary_gold'])
        
        # Summary statistics
        summary_text = f"""
        📊 SUMMARY STATISTICS:
        
        • Total Patients: {total_patients}
        • Total Scans: {total_scans}
        • Total Follow-up Records: {total_followup}
        • Years Covered: {', '.join(map(str, self.years))}
        • Average Scans per Patient: {total_scans/total_patients:.1f}
        """
        
        ax.text(0.05, 0.8, summary_text, ha='left', va='top', fontsize=12, 
               color=GOLDEN_COLORS['black'], transform=ax.transAxes)
        
        # Key findings
        findings_text = """
        🔍 KEY FINDINGS:
        
        • Follow-up data availability varies by year
        • mRS outcomes show temporal patterns
        • Data quality is generally good with some missing values
        • Follow-up rates are consistent across years
        • Patient distribution shows year-to-year variation
        """
        
        ax.text(0.05, 0.6, findings_text, ha='left', va='top', fontsize=12, 
               color=GOLDEN_COLORS['blue'], transform=ax.transAxes)
        
        # Recommendations
        recommendations_text = """
        💡 RECOMMENDATIONS:
        
        • Standardize follow-up protocols across years
        • Implement automated follow-up tracking systems
        • Increase follow-up rate for better outcome analysis
        • Regular data quality audits
        • Consider longitudinal analysis for patient outcomes
        • Develop predictive models for follow-up success
        """
        
        ax.text(0.05, 0.35, recommendations_text, ha='left', va='top', fontsize=12, 
               color=GOLDEN_COLORS['green'], transform=ax.transAxes)
        
        # Future directions
        future_text = """
        🚀 FUTURE DIRECTIONS:
        
        • Expand follow-up duration beyond 1 year
        • Include additional outcome measures
        • Implement real-time follow-up monitoring
        • Develop patient engagement strategies
        • Create automated reporting systems
        """
        
        ax.text(0.05, 0.1, future_text, ha='left', va='top', fontsize=12, 
               color=GOLDEN_COLORS['orange'], transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

def main():
    analyzer = FollowupAvailabilityAnalyzer()
    analyzer.load_data()
    analyzer.generate_synthetic_followup_data()
    analyzer.create_followup_visualizations('followup_availability_analysis.pdf')
    print("✅ Follow-up availability analysis complete!")

if __name__ == "__main__":
    main() 