import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_CSV = 'results/step4_clinical_vs_radiomics_performance.csv'
PLOT_PATH = 'results/step4_clinical_vs_radiomics_performance.png'

# Load results
df = pd.read_csv(RESULTS_CSV)

# Plot MAE and R2 for each feature set
fig, ax1 = plt.subplots(figsize=(7,5))
color = 'tab:blue'
ax1.set_xlabel('Feature Set')
ax1.set_ylabel('MAE', color=color)
ax1.bar(df['feature_set'], df['MAE'], color=color, alpha=0.7, label='MAE')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('R2', color=color)
ax2.plot(df['feature_set'], df['R2'], color=color, marker='o', label='R2')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Clinical vs. Radiomics Model Performance (MAE and R2)')
plt.savefig(PLOT_PATH)
plt.show()
print(f"Plot saved to {PLOT_PATH}") 