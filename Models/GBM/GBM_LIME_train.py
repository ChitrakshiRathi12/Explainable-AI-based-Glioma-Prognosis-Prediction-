import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix
)

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")

# Preview the dataset
print(data.head())

# Step 2: Define features (X) and target (y)
le = LabelEncoder()
data['Grade_encoded'] = le.fit_transform(data['Grade'])

# Step 3: Define features (X) and target (y)
selected_features = ['Age_at_diagnosis', 'Gender', 'TP53']
X = data[selected_features]
y = data['Grade_encoded']

# Step 4: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 5: Train Gradient Boosting Classifier on full dataset
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X_scaled, y)

# Step 6: Make predictions on the same dataset
y_pred = gbm_model.predict(X_scaled)
y_proba = gbm_model.predict_proba(X_scaled)[:, 1]

# Step 7: Evaluate the model
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
roc_auc = roc_auc_score(y, y_proba)
logloss = log_loss(y, y_proba)
mcc = matthews_corrcoef(y, y_pred)
kappa = cohen_kappa_score(y, y_pred)

# Confusion matrix-based specificity calculation
cm = confusion_matrix(y, y_pred)
tn = cm[0, 0]
fp = cm[0, 1]
specificity = tn / (tn + fp)

# Print results
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=le.classes_))
print("\nAdditional Metrics:\n")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Cohen’s Kappa: {kappa:.4f}")
print(f"Specificity: {specificity:.4f}")