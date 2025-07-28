import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, ConnectionPatch
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Set golden theme colors
GOLDEN_COLORS = {
    'primary_gold': '#B8860B',
    'secondary_gold': '#DAA520',
    'light_gold': '#F4A460',
    'pale_gold': '#F5DEB3',
    'dark_gold': '#8B6914',
    'blue': '#4682B4',
    'green': '#6B8E23',
    'red': '#CD5C5C',
    'purple': '#8A2BE2',
    'orange': '#FF8C00',
    'cyan': '#00CED1',
    'brown': '#A0522D',
    'pink': '#FF69B4',
    'grey': '#808080',
    'white': '#F5F5DC',
    'black': '#36454F'
}

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['axes.facecolor'] = GOLDEN_COLORS['white']
plt.rcParams['text.color'] = GOLDEN_COLORS['black']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

class RadiomicsFlowDiagram:
    def __init__(self):
        print("🎯 Initialized Radiomics Flow Diagram Creator")
        
    def create_3d_flow_diagram(self, output_pdf='radiomics_process_flow.pdf'):
        """Create a 3D-style process flow diagram for radiomics pipeline"""
        print("🎨 Creating 3D-style radiomics flow diagram...")
        
        with PdfPages(output_pdf) as pdf:
            # Create main diagram
            self.create_main_flow_diagram(pdf)
            
            # Create detailed sub-processes
            self.create_data_collection_detail(pdf)
            self.create_feature_extraction_detail(pdf)
            self.create_ml_analysis_detail(pdf)
            self.create_clinical_validation_detail(pdf)
        
        print(f"✅ Radiomics flow diagram saved to {output_pdf}")
    
    def create_main_flow_diagram(self, pdf):
        """Create the main 3D-style process flow diagram"""
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        
        # Create 3D-style curved pathway
        path_x = np.linspace(1, 9, 100)
        path_y = 4 + 0.5 * np.sin(np.pi * (path_x - 1) / 8) + 0.1 * np.sin(3 * np.pi * (path_x - 1) / 8)
        
        # Draw 3D pathway with gradient effect
        for i in range(len(path_x) - 1):
            alpha = 0.3 + 0.4 * (i / len(path_x))
            ax.plot([path_x[i], path_x[i+1]], [path_y[i], path_y[i+1]], 
                   color=GOLDEN_COLORS['dark_gold'], linewidth=8, alpha=alpha)
        
        # Define process stages
        stages = [
            {
                'name': 'Data Collection',
                'x': 1.5,
                'y': 4.2,
                'color': GOLDEN_COLORS['primary_gold'],
                'icon': '📊',
                'above_text': ['MRI Scans', 'Patient Data'],
                'below_text': ['2020-2022', '455 Scans', '91 Patients']
            },
            {
                'name': 'Feature Extraction',
                'x': 3.5,
                'y': 4.8,
                'color': GOLDEN_COLORS['secondary_gold'],
                'icon': '🔬',
                'above_text': ['PyRadiomics', '127 Features'],
                'below_text': ['Texture', 'Shape', 'Intensity']
            },
            {
                'name': 'ML Analysis',
                'x': 5.5,
                'y': 4.2,
                'color': GOLDEN_COLORS['blue'],
                'icon': '🤖',
                'above_text': ['SVM Model', 'Feature Selection'],
                'below_text': ['80-20 Split', 'Cross-validation']
            },
            {
                'name': 'Clinical Validation',
                'x': 7.5,
                'y': 4.8,
                'color': GOLDEN_COLORS['green'],
                'icon': '🏥',
                'above_text': ['mRS Prediction', 'Outcome Analysis'],
                'below_text': ['AUC: 0.768', 'Clinical Impact']
            }
        ]
        
        # Create process nodes
        for stage in stages:
            # Create 3D-style circular node
            circle = Circle((stage['x'], stage['y']), 0.8, 
                          facecolor=stage['color'], edgecolor=GOLDEN_COLORS['dark_gold'], 
                          linewidth=3, alpha=0.9)
            ax.add_patch(circle)
            
            # Add icon
            ax.text(stage['x'], stage['y'], stage['icon'], ha='center', va='center', 
                   fontsize=24, fontweight='bold', color=GOLDEN_COLORS['white'])
            
            # Add stage name
            ax.text(stage['x'], stage['y'] + 1.2, stage['name'], ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            
            # Add above text
            above_text = '\n'.join(stage['above_text'])
            ax.text(stage['x'], stage['y'] + 0.3, above_text, ha='center', va='center', 
                   fontsize=10, color=GOLDEN_COLORS['dark_gold'], fontweight='bold')
            
            # Add below text
            below_text = '\n'.join(stage['below_text'])
            ax.text(stage['x'], stage['y'] - 1.2, below_text, ha='center', va='center', 
                   fontsize=10, color=GOLDEN_COLORS['dark_gold'], fontweight='bold')
        
        # Add parallel process (Data Quality)
        quality_x, quality_y = 1.5, 6.5
        quality_circle = Circle((quality_x, quality_y), 0.6, 
                              facecolor=GOLDEN_COLORS['orange'], edgecolor=GOLDEN_COLORS['dark_gold'], 
                              linewidth=2, alpha=0.9)
        ax.add_patch(quality_circle)
        ax.text(quality_x, quality_y, '🔍', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=GOLDEN_COLORS['white'])
        ax.text(quality_x, quality_y + 0.9, 'Data Quality\nAssessment', ha='center', va='center', 
               fontsize=12, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Connect quality assessment to main flow
        ax.annotate('', xy=(1.5, 4.8), xytext=(1.5, 5.9), 
                   arrowprops=dict(arrowstyle='->', lw=2, color=GOLDEN_COLORS['dark_gold']))
        
        # Add title
        ax.text(5, 7.5, 'Advanced Radiomics Analysis Pipeline', ha='center', va='center', 
               fontsize=20, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
        ax.text(5, 7.2, 'Real Clinical Data Analysis (2020-2022)', ha='center', va='center', 
               fontsize=14, color=GOLDEN_COLORS['dark_gold'])
        
        # Add outcome indicators
        outcomes = [
            {'text': 'mRS 0-2\n(Independent)', 'x': 8.5, 'y': 6.5, 'color': GOLDEN_COLORS['light_gold']},
            {'text': 'mRS 3-5\n(Dependent)', 'x': 8.5, 'y': 5.5, 'color': GOLDEN_COLORS['secondary_gold']}
        ]
        
        for outcome in outcomes:
            outcome_box = FancyBboxPatch((outcome['x']-0.8, outcome['y']-0.4), 1.6, 0.8,
                                       boxstyle="round,pad=0.1", 
                                       facecolor=outcome['color'], 
                                       edgecolor=GOLDEN_COLORS['dark_gold'], 
                                       linewidth=2, alpha=0.8)
            ax.add_patch(outcome_box)
            ax.text(outcome['x'], outcome['y'], outcome['text'], ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Connect validation to outcomes
        ax.annotate('', xy=(8.3, 5.2), xytext=(7.5, 4.8), 
                   arrowprops=dict(arrowstyle='->', lw=2, color=GOLDEN_COLORS['dark_gold']))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_data_collection_detail(self, pdf):
        """Create detailed data collection process"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Title
        ax.text(5, 5.5, 'Data Collection & Preprocessing', ha='center', va='center', 
               fontsize=18, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
        
        # Create detailed flow
        steps = [
            {'name': 'MRI Acquisition', 'x': 1, 'y': 4, 'icon': '📷', 'color': GOLDEN_COLORS['primary_gold']},
            {'name': 'Data Validation', 'x': 3, 'y': 4, 'icon': '✅', 'color': GOLDEN_COLORS['secondary_gold']},
            {'name': 'Patient Mapping', 'x': 5, 'y': 4, 'icon': '👥', 'color': GOLDEN_COLORS['blue']},
            {'name': 'mRS Integration', 'x': 7, 'y': 4, 'icon': '📋', 'color': GOLDEN_COLORS['green']},
            {'name': 'Quality Check', 'x': 9, 'y': 4, 'icon': '🔍', 'color': GOLDEN_COLORS['orange']}
        ]
        
        for i, step in enumerate(steps):
            # Create node
            circle = Circle((step['x'], step['y']), 0.6, 
                          facecolor=step['color'], edgecolor=GOLDEN_COLORS['dark_gold'], 
                          linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            
            # Add icon and name
            ax.text(step['x'], step['y'] + 0.2, step['icon'], ha='center', va='center', 
                   fontsize=20, fontweight='bold', color=GOLDEN_COLORS['white'])
            ax.text(step['x'], step['y'] - 0.8, step['name'], ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            
            # Connect steps
            if i < len(steps) - 1:
                ax.annotate('', xy=(steps[i+1]['x']-0.6, steps[i+1]['y']), 
                           xytext=(step['x']+0.6, step['y']), 
                           arrowprops=dict(arrowstyle='->', lw=2, color=GOLDEN_COLORS['dark_gold']))
        
        # Add data statistics
        stats = [
            '📊 455 Total Scans',
            '👥 91 Unique Patients', 
            '📅 2020-2022 Collection',
            '🏥 Real Clinical Data',
            '📋 Documented mRS Outcomes'
        ]
        
        for i, stat in enumerate(stats):
            ax.text(1, 2.5 - i*0.4, stat, ha='left', va='center', 
                   fontsize=12, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Add year distribution
        years = [2020, 2021, 2022]
        scans = [60, 140, 255]
        patients = [12, 28, 51]
        
        for i, (year, scan, patient) in enumerate(zip(years, scans, patients)):
            x_pos = 6 + i * 1.2
            ax.text(x_pos, 2.5, f'{year}', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
            ax.text(x_pos, 2.1, f'{scan} Scans', ha='center', va='center', 
                   fontsize=11, color=GOLDEN_COLORS['dark_gold'])
            ax.text(x_pos, 1.8, f'{patient} Patients', ha='center', va='center', 
                   fontsize=11, color=GOLDEN_COLORS['dark_gold'])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_feature_extraction_detail(self, pdf):
        """Create detailed feature extraction process"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Title
        ax.text(5, 5.5, 'Radiomics Feature Extraction', ha='center', va='center', 
               fontsize=18, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
        
        # Feature categories
        categories = [
            {'name': 'First-Order\nStatistics', 'x': 2, 'y': 4, 'count': 18, 'color': GOLDEN_COLORS['light_gold']},
            {'name': 'Shape\nFeatures', 'x': 4, 'y': 4, 'count': 14, 'color': GOLDEN_COLORS['secondary_gold']},
            {'name': 'Texture\nFeatures', 'x': 6, 'y': 4, 'count': 75, 'color': GOLDEN_COLORS['primary_gold']},
            {'name': 'Advanced\nFeatures', 'x': 8, 'y': 4, 'count': 20, 'color': GOLDEN_COLORS['orange']}
        ]
        
        for category in categories:
            # Create node
            circle = Circle((category['x'], category['y']), 0.7, 
                          facecolor=category['color'], edgecolor=GOLDEN_COLORS['dark_gold'], 
                          linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            
            # Add count and name
            ax.text(category['x'], category['y'] + 0.2, str(category['count']), ha='center', va='center', 
                   fontsize=16, fontweight='bold', color=GOLDEN_COLORS['white'])
            ax.text(category['x'], category['y'] - 0.9, category['name'], ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Add feature examples
        examples = [
            '📊 First-Order: Mean, Variance, Skewness, Kurtosis',
            '🔷 Shape: Volume, Surface Area, Compactness, Sphericity',
            '🎨 Texture: GLCM, GLRLM, GLSZM, NGTDM Features',
            '⚡ Advanced: Wavelet, LoG, Gradient Features'
        ]
        
        for i, example in enumerate(examples):
            ax.text(1, 2.8 - i*0.4, example, ha='left', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Add total features
        ax.text(5, 1.5, 'Total: 127 Radiomics Features', ha='center', va='center', 
               fontsize=16, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
        
        # Add PyRadiomics logo
        ax.text(5, 1, 'Powered by PyRadiomics Library', ha='center', va='center', 
               fontsize=12, color=GOLDEN_COLORS['dark_gold'])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_ml_analysis_detail(self, pdf):
        """Create detailed ML analysis process"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Title
        ax.text(5, 5.5, 'Machine Learning Analysis Pipeline', ha='center', va='center', 
               fontsize=18, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
        
        # ML steps
        steps = [
            {'name': 'Feature\nSelection', 'x': 2, 'y': 4, 'icon': '🎯', 'color': GOLDEN_COLORS['primary_gold']},
            {'name': 'Model\nTraining', 'x': 4, 'y': 4, 'icon': '🤖', 'color': GOLDEN_COLORS['secondary_gold']},
            {'name': 'Cross\nValidation', 'x': 6, 'y': 4, 'icon': '🔄', 'color': GOLDEN_COLORS['blue']},
            {'name': 'Performance\nEvaluation', 'x': 8, 'y': 4, 'icon': '📈', 'color': GOLDEN_COLORS['green']}
        ]
        
        for i, step in enumerate(steps):
            # Create node
            circle = Circle((step['x'], step['y']), 0.6, 
                          facecolor=step['color'], edgecolor=GOLDEN_COLORS['dark_gold'], 
                          linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            
            # Add icon and name
            ax.text(step['x'], step['y'] + 0.2, step['icon'], ha='center', va='center', 
                   fontsize=20, fontweight='bold', color=GOLDEN_COLORS['white'])
            ax.text(step['x'], step['y'] - 0.8, step['name'], ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            
            # Connect steps
            if i < len(steps) - 1:
                ax.annotate('', xy=(steps[i+1]['x']-0.6, steps[i+1]['y']), 
                           xytext=(step['x']+0.6, step['y']), 
                           arrowprops=dict(arrowstyle='->', lw=2, color=GOLDEN_COLORS['dark_gold']))
        
        # Add ML details
        ml_details = [
            '🎯 LASSO Feature Selection: 60 features selected',
            '🤖 Support Vector Machine (SVM) Classifier',
            '📊 80-20 Train/Test Split (364/91 scans)',
            '🔄 5-Fold Cross-Validation',
            '📈 Performance Metrics: AUC, Sensitivity, Specificity'
        ]
        
        for i, detail in enumerate(ml_details):
            ax.text(1, 2.8 - i*0.4, detail, ha='left', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Add performance results
        ax.text(7, 2.5, 'Training AUC: 0.888', ha='center', va='center', 
               fontsize=14, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
        ax.text(7, 2.1, 'Test AUC: 0.768', ha='center', va='center', 
               fontsize=14, fontweight='bold', color=GOLDEN_COLORS['secondary_gold'])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()
    
    def create_clinical_validation_detail(self, pdf):
        """Create detailed clinical validation process"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis('off')
        
        # Title
        ax.text(5, 5.5, 'Clinical Validation & Impact Assessment', ha='center', va='center', 
               fontsize=18, fontweight='bold', color=GOLDEN_COLORS['primary_gold'])
        
        # Validation steps
        steps = [
            {'name': 'Outcome\nPrediction', 'x': 2, 'y': 4, 'icon': '🎯', 'color': GOLDEN_COLORS['primary_gold']},
            {'name': 'Clinical\nInterpretation', 'x': 4, 'y': 4, 'icon': '👨‍⚕️', 'color': GOLDEN_COLORS['secondary_gold']},
            {'name': 'Feature\nImportance', 'x': 6, 'y': 4, 'icon': '📊', 'color': GOLDEN_COLORS['blue']},
            {'name': 'Clinical\nImpact', 'x': 8, 'y': 4, 'icon': '🏥', 'color': GOLDEN_COLORS['green']}
        ]
        
        for i, step in enumerate(steps):
            # Create node
            circle = Circle((step['x'], step['y']), 0.6, 
                          facecolor=step['color'], edgecolor=GOLDEN_COLORS['dark_gold'], 
                          linewidth=2, alpha=0.9)
            ax.add_patch(circle)
            
            # Add icon and name
            ax.text(step['x'], step['y'] + 0.2, step['icon'], ha='center', va='center', 
                   fontsize=20, fontweight='bold', color=GOLDEN_COLORS['white'])
            ax.text(step['x'], step['y'] - 0.8, step['name'], ha='center', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
            
            # Connect steps
            if i < len(steps) - 1:
                ax.annotate('', xy=(steps[i+1]['x']-0.6, steps[i+1]['y']), 
                           xytext=(step['x']+0.6, step['y']), 
                           arrowprops=dict(arrowstyle='->', lw=2, color=GOLDEN_COLORS['dark_gold']))
        
        # Add clinical outcomes
        outcomes = [
            '📊 mRS 0-2 (Independent): 265 patients (58.2%)',
            '📊 mRS 3-5 (Dependent): 190 patients (41.8%)',
            '🎯 Prediction Accuracy: 76.8%',
            '⚡ Early Intervention Potential',
            '💡 Personalized Treatment Planning'
        ]
        
        for i, outcome in enumerate(outcomes):
            ax.text(1, 2.8 - i*0.4, outcome, ha='left', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        # Add clinical impact
        impacts = [
            '🏥 Early Outcome Prediction',
            '👥 Improved Patient Counseling',
            '📋 Resource Allocation Optimization',
            '🎯 Personalized Care Planning'
        ]
        
        for i, impact in enumerate(impacts):
            ax.text(6, 2.8 - i*0.4, impact, ha='left', va='center', 
                   fontsize=11, fontweight='bold', color=GOLDEN_COLORS['dark_gold'])
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    diagram = RadiomicsFlowDiagram()
    diagram.create_3d_flow_diagram('radiomics_process_flow.pdf')
    print("✅ Radiomics flow diagram complete!")

if __name__ == "__main__":
    main() 