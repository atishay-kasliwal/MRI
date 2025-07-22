import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import shap
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# Paths
output_dir = os.path.join('results', 'advanced_radiomics_analysis')
os.makedirs(output_dir, exist_ok=True)
input_csv = os.path.join('results', 'radiomics_lastmrs_mapping.csv')

# Load data
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv)

# Identify radiomic feature columns
radiomic_cols = [col for col in df.columns if col.startswith('original_')]

# Store plot filenames for PDF
plot_files = []

# 1. Distribution of Last mRS
plt.figure(figsize=(8, 5))
sns.histplot(df['Last mRS'], bins=8, kde=False)
plt.title('Distribution of Last mRS')
plt.xlabel('Last mRS')
plt.ylabel('Count')
dist_plot = os.path.join(output_dir, 'lastmrs_distribution.png')
plt.savefig(dist_plot)
plot_files.append(dist_plot)
plt.close()

# 2. Feature Correlation Heatmap
corr = df[radiomic_cols + ['Last mRS']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr[['Last mRS']].sort_values('Last mRS', ascending=False), annot=True, cmap='coolwarm')
plt.title('Correlation of Radiomic Features with Last mRS')
corr_plot = os.path.join(output_dir, 'feature_lastmrs_correlation_heatmap.png')
plt.savefig(corr_plot)
plot_files.append(corr_plot)
plt.close()

# 3. Top Feature Importances (Random Forest)
X = df[radiomic_cols]
y = df['Last mRS']
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=radiomic_cols)
top_features = importances.sort_values(ascending=False).head(15)
plt.figure(figsize=(10, 6))
top_features.plot(kind='barh')
plt.title('Top 15 Radiomic Feature Importances (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
featimp_plot = os.path.join(output_dir, 'top15_rf_feature_importances.png')
plt.savefig(featimp_plot)
plot_files.append(featimp_plot)
plt.close()

# 4. PCA and t-SNE Visualizations
# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Last mRS', data=df, palette='viridis', legend='full')
plt.title('PCA of Radiomic Features (colored by Last mRS)')
pca_plot = os.path.join(output_dir, 'pca_lastmrs.png')
plt.savefig(pca_plot)
plot_files.append(pca_plot)
plt.close()
# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
tsne_result = tsne.fit_transform(X)
df['TSNE1'] = tsne_result[:, 0]
df['TSNE2'] = tsne_result[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Last mRS', data=df, palette='viridis', legend='full')
plt.title('t-SNE of Radiomic Features (colored by Last mRS)')
tsne_plot = os.path.join(output_dir, 'tsne_lastmrs.png')
plt.savefig(tsne_plot)
plot_files.append(tsne_plot)
plt.close()

# 5. Pairplot of Top Features
pairplot_cols = top_features.index[:4].tolist() + ['Last mRS']
pairplot_file = os.path.join(output_dir, 'pairplot_top_features.png')
sns.pairplot(df[pairplot_cols], hue='Last mRS', palette='viridis')
plt.savefig(pairplot_file)
plot_files.append(pairplot_file)
plt.close()

# 6. Predictive Modeling with Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
metrics_file = os.path.join(output_dir, 'rf_model_metrics.txt')
with open(metrics_file, 'w') as f:
    f.write(f"Random Forest MAE: {mae:.3f}\n")
    f.write(f"Random Forest R2: {r2:.3f}\n")

# 7. SHAP Explanations
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
shap_bar = os.path.join(output_dir, 'shap_summary_bar.png')
plt.tight_layout()
plt.savefig(shap_bar)
plot_files.append(shap_bar)
plt.close()
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
shap_dot = os.path.join(output_dir, 'shap_summary_dot.png')
plt.tight_layout()
plt.savefig(shap_dot)
plot_files.append(shap_dot)
plt.close()

# 8. Clustering and Cluster-wise Outcome Analysis
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Last mRS', data=df)
plt.title('Last mRS Distribution by Cluster')
cluster_plot = os.path.join(output_dir, 'cluster_lastmrs_boxplot.png')
plt.savefig(cluster_plot)
plot_files.append(cluster_plot)
plt.close()

# 9. Outlier Detection (simple: z-score on Last mRS)
df['Last mRS_z'] = (df['Last mRS'] - df['Last mRS'].mean()) / df['Last mRS'].std()
outliers = df[np.abs(df['Last mRS_z']) > 3]
outlier_file = os.path.join(output_dir, 'lastmrs_outliers.csv')
outliers.to_csv(outlier_file, index=False)

# 10. Cross-Modality Analysis
if 'Modality' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Modality', y='Last mRS', data=df)
    plt.title('Last mRS by Modality')
    modality_plot = os.path.join(output_dir, 'lastmrs_by_modality.png')
    plt.savefig(modality_plot)
    plot_files.append(modality_plot)
    plt.close()
    # Feature importance by modality (optional, for top 2 modalities)
    modalities = df['Modality'].value_counts().index[:2]
    for mod in modalities:
        mod_df = df[df['Modality'] == mod]
        if len(mod_df) > 10:
            X_mod = mod_df[radiomic_cols]
            y_mod = mod_df['Last mRS']
            rf_mod = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_mod.fit(X_mod, y_mod)
            importances_mod = pd.Series(rf_mod.feature_importances_, index=radiomic_cols)
            top_mod = importances_mod.sort_values(ascending=False).head(10)
            plt.figure(figsize=(8, 5))
            top_mod.plot(kind='barh')
            plt.title(f'Top 10 Feature Importances: {mod}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            mod_plot = os.path.join(output_dir, f'top10_rf_feature_importances_{mod}.png')
            plt.savefig(mod_plot)
            plot_files.append(mod_plot)
            plt.close()

# --- PDF 1: All plots in a single PDF ---
pdf_plots_path = os.path.join(output_dir, 'all_plots.pdf')
with PdfPages(pdf_plots_path) as pdf:
    for plot in plot_files:
        fig = plt.figure()
        img = plt.imread(plot)
        plt.imshow(img)
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)
print(f'All plots saved to {pdf_plots_path}')

# --- PDF 2: Advanced PDF report with text and plots (reportlab) ---
pdf_report_path = os.path.join(output_dir, 'advanced_radiomics_report.pdf')
c = canvas.Canvas(pdf_report_path, pagesize=letter)
width, height = letter

# Title page
c.setFont("Helvetica-Bold", 20)
c.drawCentredString(width/2, height-100, "Advanced Radiomics Analysis Report")
c.setFont("Helvetica", 12)
c.drawCentredString(width/2, height-130, "Generated by automated pipeline")
c.showPage()

# Summary statistics page
c.setFont("Helvetica-Bold", 16)
c.drawString(50, height-50, "Summary Statistics")
c.setFont("Helvetica", 12)
c.drawString(50, height-80, f"Number of samples: {len(df)}")
c.drawString(50, height-100, f"Number of radiomic features: {len(radiomic_cols)}")
c.drawString(50, height-120, f"Last mRS (mean ± std): {df['Last mRS'].mean():.2f} ± {df['Last mRS'].std():.2f}")
c.drawString(50, height-140, f"Last mRS (min, max): {df['Last mRS'].min()} - {df['Last mRS'].max()}")
c.showPage()

# Model metrics page
c.setFont("Helvetica-Bold", 16)
c.drawString(50, height-50, "Random Forest Model Metrics")
c.setFont("Helvetica", 12)
c.drawString(50, height-80, f"MAE: {mae:.3f}")
c.drawString(50, height-100, f"R2: {r2:.3f}")
c.showPage()

# Add key plots as images (one per page)
for plot in plot_files:
    c.setFont("Helvetica", 10)
    c.drawString(50, height-30, os.path.basename(plot))
    img = ImageReader(plot)
    c.drawImage(img, 50, 150, width=500, height=400, preserveAspectRatio=True)
    c.showPage()

c.save()
print(f'Advanced PDF report saved to {pdf_report_path}')

print('All analyses and visualizations complete. Results saved to', output_dir) 