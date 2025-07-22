import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.patches import Patch

# Output directory
output_dir = os.path.join('results', 'visualize_radiomics_2020_2021')
os.makedirs(output_dir, exist_ok=True)
input_csv = os.path.join('data', 'radiomics', 'merged_radiomics_clinical.csv')

# Load data
print(f"Loading data from {input_csv}...")
df = pd.read_csv(input_csv)

# Filter for 2020 and 2021
df = df[df['Year'].isin([2020, 2021])]

# Identify radiomic feature columns
radiomic_cols = [col for col in df.columns if col.startswith('original_')]

# Top 10 features by variance
top_features = df[radiomic_cols].var().sort_values(ascending=False).head(10).index.tolist()

# Prepare PDF
pdf_path = os.path.join(output_dir, 'all_plots_2020_2021.pdf')
pdf = PdfPages(pdf_path)

# --- Summary Page ---
fig, ax = plt.subplots(figsize=(8, 6))
counts = df['Year'].value_counts().sort_index()
summary_text = f"Radiomics Visualization for 2020 & 2021\n\nSample counts:\n"
for year, count in counts.items():
    summary_text += f"{year}: {count}\n"
ax.text(0.1, 0.7, summary_text, fontsize=14, va='top')
ax.axis('off')
pdf.savefig(fig)
plt.close(fig)

# --- Summary statistics page for top features by year ---
sum_stats = []
for feature in top_features:
    for year in [2020, 2021]:
        vals = df[df['Year'] == year][feature].dropna()
        sum_stats.append({
            'Feature': feature,
            'Year': year,
            'Mean': vals.mean(),
            'Std': vals.std(),
            'Min': vals.min(),
            'Max': vals.max(),
        })
sum_stats_df = pd.DataFrame(sum_stats)
fig, ax = plt.subplots(figsize=(12, 0.5 + 0.3 * len(sum_stats_df)))
ax.axis('off')
table = ax.table(cellText=np.round(sum_stats_df[['Mean', 'Std', 'Min', 'Max']].values, 3),
                 colLabels=['Mean', 'Std', 'Min', 'Max'],
                 rowLabels=[f"{row['Feature']} ({int(row['Year'])})" for _, row in sum_stats_df.iterrows()],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
plt.title('Summary Statistics for Top Features by Year', fontsize=14, pad=20)
pdf.savefig(fig)
plt.close(fig)

# --- Correlation heatmap of top features (combined) ---
corr = df[top_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Top 10 Radiomic Features (2020 & 2021 Combined)')
pdf.savefig()
plt.close()

# --- Distribution plots for each feature and year ---
for feature in top_features:
    plt.figure(figsize=(8, 5))
    for year in [2020, 2021]:
        sns.histplot(df[df['Year'] == year][feature], label=str(year), kde=True, stat='density', bins=20, alpha=0.6)
    plt.title(f'Distribution of {feature} (2020 vs 2021)')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    pdf.savefig()
    plt.close()

# --- Boxplots comparing 2020 vs 2021 for each feature ---
for feature in top_features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Year', y=feature, data=df)
    plt.title(f'Boxplot of {feature} by Year')
    pdf.savefig()
    plt.close()

# --- Violin plots comparing 2020 vs 2021 for each feature ---
for feature in top_features:
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='Year', y=feature, data=df, inner='quartile')
    plt.title(f'Violin Plot of {feature} by Year')
    pdf.savefig()
    plt.close()

# --- PCA plot colored by year ---
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[top_features].fillna(0))
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Year', data=df, palette='Set1', alpha=0.8)
plt.title('PCA of Top 10 Radiomic Features (2020 vs 2021)')
pdf.savefig()
plt.close()

# --- t-SNE plot colored by year ---
tsne = TSNE(n_components=2, random_state=42, perplexity=10, max_iter=1000)
tsne_result = tsne.fit_transform(df[top_features].fillna(0))
df['TSNE1'] = tsne_result[:, 0]
df['TSNE2'] = tsne_result[:, 1]
plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Year', data=df, palette='Set1', alpha=0.8)
plt.title('t-SNE of Top 10 Radiomic Features (2020 vs 2021)')
pdf.savefig()
plt.close()

# --- Patient-wise heatmap (all patients, top 10 features) ---
plt.figure(figsize=(min(20, 0.25*len(df)), 8))
heatmap_data = df[top_features].fillna(0)
row_colors = df['Year'].map({2020: 'skyblue', 2021: 'salmon'})
sns.heatmap(heatmap_data.T, cmap='viridis', cbar=True, yticklabels=top_features)
plt.title('Patient-wise Heatmap (Top 10 Features, 2020 & 2021)')
plt.xlabel('Patient Index')
plt.ylabel('Feature')
plt.tight_layout()
pdf.savefig()
plt.close()

# --- Outlier detection (z-score > 3 in any top feature) ---
z_scores = np.abs((heatmap_data - heatmap_data.mean()) / heatmap_data.std())
outlier_patients = z_scores.max(axis=1) > 3

# --- Radar plots for each patient (top 10 features) ---
def radar_factory(num_vars, frame='circle'):
    # From matplotlib radar chart example
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    import matplotlib.patches as mpatches
    import numpy as np
    
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    class RadarAxes(PolarAxes):
        name = 'radar'
        def fill(self, *args, **kwargs):
            closed = kwargs.pop('closed', True)
            return super().fill(closed=closed, *args, **kwargs)
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                line.set_clip_on(False)
            return lines
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        def _gen_axes_patch(self):
            if frame == 'circle':
                return mpatches.Circle((0.5, 0.5), 0.5)
            else:
                return super()._gen_axes_patch()
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            else:
                return super()._gen_axes_spines()
    register_projection(RadarAxes)
    return theta

num_vars = len(top_features)
theta = radar_factory(num_vars)

for idx, row in df.iterrows():
    values = row[top_features].values
    feature_labels = top_features
    year = int(row['Year'])
    patient_id = row['PatientID'] if 'PatientID' in row else str(idx)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection='radar')
    ax.plot(theta, values, color='skyblue' if year == 2020 else 'salmon', linewidth=2)
    ax.fill(theta, values, color='skyblue' if year == 2020 else 'salmon', alpha=0.25)
    ax.set_varlabels(feature_labels)
    ax.set_title(f'Patient {patient_id} ({year})', va='bottom')
    if outlier_patients.iloc[idx]:
        ax.spines['polar'].set_color('red')
        ax.spines['polar'].set_linewidth(3)
        ax.set_title(f'Patient {patient_id} ({year}) - OUTLIER', color='red', va='bottom')
    pdf.savefig(fig)
    plt.close(fig)

# --- Patient clustering (k-means, show cluster assignment by year) ---
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[top_features].fillna(0))
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', style='Year', data=df, palette='tab10', alpha=0.8)
plt.title('Patient Clusters (KMeans, Top 10 Features, PCA space)')
legend_elements = [Patch(facecolor='skyblue', label='2020'), Patch(facecolor='salmon', label='2021')]
plt.legend(handles=legend_elements, title='Year', loc='best')
pdf.savefig()
plt.close()

pdf.close()
print(f'All plots saved to {pdf_path}') 