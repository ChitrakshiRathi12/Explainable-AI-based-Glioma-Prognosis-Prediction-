import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix
)
from XGBoost_PFI_train import le, scaler, selected_features,xgb_model
# Step 9: Load the test dataset
test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Step 10: Encode the target variable in test set using the same LabelEncoder
test_data['Grade_encoded'] = le.transform(test_data['Grade'])  # Assumes all test labels exist in training set

# Step 11: Prepare features and target
X_test = test_data[selected_features]
y_test = test_data['Grade_encoded']

# Step 12: Scale the test features using the same scaler
X_test_scaled = scaler.transform(X_test)

# Step 13: Make predictions
y_test_pred = xgb_model.predict(X_test_scaled)

# Step 14: Evaluate on test data
precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_test = f1_score(y_test, y_test_pred, average='weighted')
accuracy_test = accuracy_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred, multi_class='ovo')
logloss_test = log_loss(y_test, xgb_model.predict_proba(X_test_scaled))
mcc_test = matthews_corrcoef(y_test, y_test_pred)
kappa_test = cohen_kappa_score(y_test, y_test_pred)

# Specificity for binary classification
if len(le.classes_) == 2:
    cm_test = confusion_matrix(y_test, y_test_pred)
    tn_test = cm_test[0, 0]
    fp_test = cm_test[0, 1]
    specificity_test = tn_test / (tn_test + fp_test)
else:
    specificity_test = "Not applicable for multi-class classification"

# Step 15: Print test set results
print("\nClassification Report (Test Set):\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
print(f"\nTest Set Performance Metrics:")
print(f"Precision: {precision_test:.4f}")
print(f"Recall: {recall_test:.4f}")
print(f"F1-Score: {f1_test:.4f}")
print(f"Accuracy: {accuracy_test:.4f}")
print(f"ROC-AUC Score: {roc_auc_test:.4f}")
print(f"Log Loss: {logloss_test:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_test:.4f}")
print(f"Cohen’s Kappa: {kappa_test:.4f}")
print(f"Specificity: {specificity_test}")
def get_metrics():
    return {
        "Precision": precision_test,
        "Recall": recall_test,
        "F1-Score": f1_test,
        "Accuracy": accuracy_test,
        "Log Loss": log_loss if log_loss is not None else 0.0,
        "MCC": mcc_test,
        "Kappa": kappa_test,
        "ROU_AUC": roc_auc_test,
        "Specificity": specificity_test
    }