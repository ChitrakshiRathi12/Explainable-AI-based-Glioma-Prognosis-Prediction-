import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")

# Encode target if needed
if data['Grade'].dtype == 'object':
    le = LabelEncoder()
    data['Grade'] = le.fit_transform(data['Grade'])

X = data.drop(columns=['Grade','Primary_Diagnosis'])
y = data['Grade']
feature_names = X.columns

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (important for valid PFI)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train-test split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale AFTER splitting
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model on scaled training data
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Compute PFI
pfi_result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, scoring='accuracy')

# Plot PFI
sorted_idx = pfi_result.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), pfi_result.importances_mean[sorted_idx],
         xerr=pfi_result.importances_std[sorted_idx], color='salmon')
plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
plt.xlabel("Decrease in Accuracy")
plt.title("Permutation Feature Importance (PFI)")
plt.tight_layout()
plt.show()

# Create DataFrame from PFI results
pfi_df = pd.DataFrame({
    'Feature': feature_names[sorted_idx],
    'Mean Importance': pfi_result.importances_mean[sorted_idx],
    'Std Dev': pfi_result.importances_std[sorted_idx]
})

# Display the top features
print("\nTop PFI Scores:")
print(pfi_df.sort_values(by='Mean Importance', ascending=False).head(10))