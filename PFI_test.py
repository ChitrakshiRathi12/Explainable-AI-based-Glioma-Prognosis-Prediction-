import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from PFI_training import le, X_train, scaler, model,feature_names

test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Optional: Encode Grade if present
if 'Grade' in test_data.columns:
    if test_data['Grade'].dtype == 'object':
        test_data['Grade'] = le.transform(test_data['Grade'])
    y_test = test_data['Grade']
else:
    y_test = None

X_test = test_data.drop(columns=['Primary_Diagnosis'])
X_test = X_test[X_train.columns]  # Ensure same feature order
X_test_scaled = scaler.transform(X_test)

# ---------------- Predict & Evaluate ----------------
if y_test is not None:
    y_pred = model.predict(X_test_scaled)
    print("Accuracy on actual test data:", accuracy_score(y_test, y_pred))

# ---------------- Compute PFI ----------------
pfi_result = permutation_importance(
    model,
    X_test_scaled,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

# ---------------- Plot PFI ----------------
sorted_idx = pfi_result.importances_mean.argsort()
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), pfi_result.importances_mean[sorted_idx],
         xerr=pfi_result.importances_std[sorted_idx], color='teal')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel("Decrease in Accuracy")
plt.title("Permutation Feature Importance (Test Data)")
plt.tight_layout()
plt.show()

# ---------------- Save Results ----------------
pfi_df = pd.DataFrame({
    'Feature': feature_names[sorted_idx],
    'Mean Importance': pfi_result.importances_mean[sorted_idx],
    'Std Dev': pfi_result.importances_std[sorted_idx]
})
pfi_df_sorted = pfi_df.sort_values(by='Mean Importance', ascending=False)

print("\nTop PFI Features (Test Data):")
print(pfi_df_sorted.head(10))

# Save to CSV
pfi_df_sorted.to_csv(r"C:\Users\Chitrakshi\Downloads\pfi_test_data.csv", index=False)