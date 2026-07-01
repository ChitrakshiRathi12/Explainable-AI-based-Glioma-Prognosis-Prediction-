import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
from Shap_training import X, X_scaled,scaler, y, model

# Load test data
test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Prepare test features
X_test = test_data.drop(columns=['Primary_Diagnosis'])
X_test = X_test[X.columns]  # Ensure column alignment
X_test_scaled = scaler.transform(X_test)

# SHAP analysis on test data
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

# Show only SHAP summary bar plot
shap.summary_plot(shap_values, X_test)
plt.title("SHAP Feature Importance on Test Data")
plt.tight_layout()
plt.show()

