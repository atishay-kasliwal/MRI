import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# Path to merged data
MERGED_CSV = 'data/radiomics/merged_radiomics_clinical.csv'
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load merged data
data = pd.read_csv(MERGED_CSV)
print(f"Loaded merged data: {len(data)} rows")

# Drop rows with missing Last mRS
data = data.dropna(subset=['Last mRS'])
print(f"Rows after dropping missing Last mRS: {len(data)}")

# Prepare features and target
drop_cols = ['PatientID', 'MRN ANON', 'Last mRS', 'Modality']
X = data.drop(columns=[col for col in drop_cols if col in data.columns])
y = data['Last mRS']

# Convert all columns to numeric where possible, coerce errors to NaN
X = X.apply(pd.to_numeric, errors='coerce')

# Fill missing values (if any)
X = X.fillna(X.mean())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
model_path = os.path.join(MODELS_DIR, 'random_forest_last_mrs.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MAE:', mae)
print('R2:', r2)

# Save predictions
pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
pred_path = os.path.join(RESULTS_DIR, 'last_mrs_predictions.csv')
pred_df.to_csv(pred_path, index=False)
print(f"Predictions saved to {pred_path}")

# Save evaluation metrics
metrics_path = os.path.join(RESULTS_DIR, 'last_mrs_metrics.csv')
with open(metrics_path, 'w') as f:
    f.write('MAE,R2\n')
    f.write(f'{mae},{r2}\n')
print(f"Metrics saved to {metrics_path}")

# Feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns

# Save feature importances
fi_df = pd.DataFrame({'feature': feature_names[indices], 'importance': importances[indices]})
fi_path = os.path.join(RESULTS_DIR, 'last_mrs_feature_importances.csv')
fi_df.to_csv(fi_path, index=False)
print(f"Feature importances saved to {fi_path}")

print("Top 10 Features:")
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot feature importance
plt.figure(figsize=(10,6))
plt.title('Feature Importances (Top 20)')
plt.bar(range(20), importances[indices[:20]], align='center')
plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'last_mrs_feature_importances.png'))
plt.show() 