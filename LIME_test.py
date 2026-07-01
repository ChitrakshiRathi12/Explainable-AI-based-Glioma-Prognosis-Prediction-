import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import numpy as np
from LIME_Training import model, le, scaler, X_scaled, feature_names, y

test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# If Grade is available in test data and encoded, encode it using the same encoder
if 'Grade' in test_data.columns and test_data['Grade'].dtype == 'object':
    test_data['Grade'] = le.transform(test_data['Grade'])

# Drop 'Primary_Diagnosis' and scale test features
X_test = test_data.drop(columns=['Primary_Diagnosis'])
X_test_scaled = scaler.transform(X_test)

# ----------------- LIME Explainer -----------------
explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=feature_names,
    class_names=[str(i) for i in sorted(set(y))],
    mode='classification'
)

# Choose a test sample to explain
test_sample_index = 20  # change this index as needed

exp_test = explainer.explain_instance(
    data_row=X_test_scaled[test_sample_index],
    predict_fn=model.predict_proba,
    num_features=23
)

# Plot LIME explanation
exp_test.as_pyplot_figure()
plt.title(f"LIME Explanation for Test Sample #{test_sample_index}")
plt.tight_layout()
plt.show()