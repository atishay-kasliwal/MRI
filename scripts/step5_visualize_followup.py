import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = 'results'

# 1. Histogram of Days f/u
days_fu_img = os.path.join(RESULTS_DIR, 'step5_days_fu_distribution.png')
img = plt.imread(days_fu_img)
plt.figure(figsize=(7,4))
plt.imshow(img)
plt.axis('off')
plt.title('Distribution of Follow-up Days')
plt.tight_layout()
plt.show()

# 2. Bar plot of MAE/R2 with and without Days f/u
summary_path = os.path.join(RESULTS_DIR, 'step5_followup_model_comparison.csv')
sum_df = pd.read_csv(summary_path)
fig, ax1 = plt.subplots(figsize=(6,4))
color = 'tab:blue'
ax1.set_xlabel('Model')
ax1.set_ylabel('MAE', color=color)
ax1.bar(sum_df['Model'], sum_df['MAE'], color=color, alpha=0.7, label='MAE')
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('R2', color=color)
ax2.plot(sum_df['Model'], sum_df['R2'], color=color, marker='o', label='R2')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.title('Model Performance With/Without Days f/u')
plt.savefig(os.path.join(RESULTS_DIR, 'step5_model_comparison.png'))
plt.show()

# 3. Stratified MAE and R2 by follow-up bin
strat_path = os.path.join(RESULTS_DIR, 'step5_stratified_followup_performance.csv')
strat_df = pd.read_csv(strat_path)
plt.figure(figsize=(8,5))
plt.bar(strat_df['fu_bin'], strat_df['MAE'], color='tab:blue', alpha=0.7, label='MAE')
plt.ylabel('MAE')
plt.xlabel('Follow-up Bin')
plt.title('Model MAE by Follow-up Duration Bin')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step5_stratified_mae.png'))
plt.show()

plt.figure(figsize=(8,5))
plt.bar(strat_df['fu_bin'], strat_df['R2'], color='tab:green', alpha=0.7, label='R2')
plt.ylabel('R2')
plt.xlabel('Follow-up Bin')
plt.title('Model R2 by Follow-up Duration Bin')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step5_stratified_r2.png'))
plt.show() 