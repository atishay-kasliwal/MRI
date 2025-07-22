import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = 'results'

# 1. Load feature importances
fi_path = os.path.join(RESULTS_DIR, 'step2_feature_importances.csv')
perm_path = os.path.join(RESULTS_DIR, 'step2_permutation_importances.csv')
shap_path = os.path.join(RESULTS_DIR, 'step2_shap_importances.csv')

fi_df = pd.read_csv(fi_path)
perm_df = pd.read_csv(perm_path)
shap_df = pd.read_csv(shap_path)

# 2. Plot top 15 features from each method
plt.figure(figsize=(8,5))
plt.barh(fi_df['feature'][:15][::-1], fi_df['importance'][:15][::-1], color='tab:blue')
plt.xlabel('Importance')
plt.title('Top 15 Model Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step7_top15_model_importances.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.barh(perm_df['feature'][:15][::-1], perm_df['perm_importance_mean'][:15][::-1], color='tab:orange')
plt.xlabel('Permutation Importance (mean)')
plt.title('Top 15 Permutation Importances')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step7_top15_permutation_importances.png'))
plt.close()

plt.figure(figsize=(8,5))
plt.barh(shap_df['feature'][:15][::-1], shap_df['mean_abs_shap'][:15][::-1], color='tab:green')
plt.xlabel('Mean |SHAP value|')
plt.title('Top 15 SHAP Importances')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step7_top15_shap_importances.png'))
plt.close()

# 3. Compare top features across methods
plt.figure(figsize=(10,7))
plt.plot(fi_df['feature'][:15], fi_df['importance'][:15], 'o-', label='Model FI', color='tab:blue')
plt.plot(perm_df['feature'][:15], perm_df['perm_importance_mean'][:15], 'o-', label='Permutation', color='tab:orange')
plt.plot(shap_df['feature'][:15], shap_df['mean_abs_shap'][:15], 'o-', label='SHAP', color='tab:green')
plt.xticks(rotation=90)
plt.ylabel('Importance (normalized)')
plt.title('Top 15 Features: Model FI vs Permutation vs SHAP')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'step7_compare_top15_importances.png'))
plt.close()

print("Step 7 complete: Interpretation and reporting plots saved to results/.") 