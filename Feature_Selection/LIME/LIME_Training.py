import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")

# Encode target if needed
if data['Grade'].dtype == 'object':
    le = LabelEncoder()
    data['Grade'] = le.fit_transform(data['Grade'])

# Remove 'Primary_Diagnosis' and separate features/target
X = data.drop(columns=['Grade', 'Primary_Diagnosis'])
y = data['Grade']
feature_names = X.columns

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train CatBoost model
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_scaled, y)

# LIME Explainer
explainer = LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=feature_names,
    class_names=[str(i) for i in sorted(set(y))],
    mode='classification'
)

# Choose a sample to explain
sample_index = 20
exp = explainer.explain_instance(
    data_row=X_scaled[sample_index],
    predict_fn=model.predict_proba,
    num_features=23
)

# Plot the explanation
exp.as_pyplot_figure()
plt.title(f"LIME Explanation for Sample #{sample_index}")
plt.tight_layout()
plt.show()

# Extract explanation into a DataFrame
lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "LIME Score"])
print("\nTop LIME Explanation Scores:")
print(lime_df)

# Optional: Save to CSV
lime_df.to_csv(r"C:\Users\Chitrakshi\Downloads\lime_scores_sample0.csv", index=False)
