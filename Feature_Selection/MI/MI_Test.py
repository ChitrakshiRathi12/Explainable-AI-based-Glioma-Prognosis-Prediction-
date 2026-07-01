import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Import trained MI model and preprocessing objects
from LIME_Training import model, le, scaler, X_scaled, feature_names, y

# ---------------- Load MI Test Dataset ----------------
mi_test = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Encode Grade if it exists and is categorical
if 'Grade' in mi_test.columns and mi_test['Grade'].dtype == 'object':
    mi_test['Grade'] = le.transform(mi_test['Grade'])

# Remove target column if present
if 'Primary_Diagnosis' in mi_test.columns:
    X_test = mi_test.drop(columns=['Primary_Diagnosis'])
else:
    X_test = mi_test.copy()

# Scale the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# ---------------- LIME Explainer ----------------
explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=feature_names,
    class_names=[str(i) for i in sorted(set(y))],
    mode='classification'
)

# Select the sample to explain
test_sample_index = 20   # Change as needed

# Generate explanation
exp_test = explainer.explain_instance(
    data_row=X_test_scaled[test_sample_index],
    predict_fn=model.predict_proba,
    num_features=len(feature_names)   # Show all MI-selected features
)

# ---------------- Plot ----------------
fig = exp_test.as_pyplot_figure()
plt.title(f"LIME Explanation for MI Test Sample #{test_sample_index}")
plt.tight_layout()
plt.show()

# ---------------- Print Explanation ----------------
print(f"\nLIME Explanation for MI Test Sample #{test_sample_index}\n")
for feature, weight in exp_test.as_list():
    print(f"{feature}: {weight:.4f}")