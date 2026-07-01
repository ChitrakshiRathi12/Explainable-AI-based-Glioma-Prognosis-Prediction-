import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix
)
from KNN_Correlation_train import selected_features,scaler,le, knn_model
# Step 8: Load the test dataset
test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Step 9: Encode target variable in test data (if available)
if 'Grade' in test_data.columns:
    test_data['Grade_encoded'] = le.transform(test_data['Grade'])
    y_test = test_data['Grade_encoded']
else:
    y_test = None  # Labels not available

# Step 10: Select and normalize features
X_test = test_data[selected_features]
X_test_scaled = scaler.transform(X_test)

# Step 11: Predict on test data
y_test_pred = knn_model.predict(X_test_scaled)
y_test_proba = knn_model.predict_proba(X_test_scaled)[:, 1]

# Step 12: Evaluate if ground truth is available
if y_test is not None:
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_test_proba)
    logloss = log_loss(y_test, y_test_proba)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp)
    else:
        specificity = "Not applicable for multi-class"

    # Print test performance
    print("\n=== Test Set Evaluation ===")
    print("\nClassification Report (Test Set):\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
    print(f"\nTest Performance Metrics:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Cohen’s Kappa: {kappa:.4f}")
    print(f"Specificity: {specificity}")
else:
    print("\nNo labels found in test data. Showing predictions only:")
    print(y_test_pred)
def get_metrics():
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy,
        "Log Loss": log_loss if log_loss is not None else 0.0,
        "MCC": mcc,
        "Kappa": kappa,
        "ROU_AUC": roc_auc,
        "Specificity": specificity
    }