import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")

# Encode target if needed
if data['Grade'].dtype == 'object':
    le = LabelEncoder()
    data['Grade'] = le.fit_transform(data['Grade'])

# Remove 'Primary_Diagnosis' and separate features/target
X = data.drop(columns=['Grade', 'Primary_Diagnosis'])
y = data['Grade']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train CatBoost model
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X, y)

# SHAP with TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP summary plot
shap.summary_plot(shap_values, X)

# Create SHAP DataFrame
shap_df = pd.DataFrame(np.abs(shap_values).mean(axis=0), index=X.columns, columns=['Mean SHAP Value'])
shap_df = shap_df.sort_values(by='Mean SHAP Value', ascending=False)

# Display top SHAP scores
print("Top SHAP Scores:")
print(shap_df.head(10))

# Optional: save SHAP values to CSV
shap_df.to_csv(r"C:\Users\Chitrakshi\Downloads\shap_scores.csv")
