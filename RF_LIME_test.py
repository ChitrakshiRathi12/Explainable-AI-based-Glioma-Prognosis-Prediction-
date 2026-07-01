import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix
)
from RF_LIME_train import le, scaler,rf_model,selected_features
# Step 9: Load the test dataset
test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Step 10: Encode the target variable in test set (if available)
if 'Grade' in test_data.columns:
    test_data['Grade_encoded'] = le.transform(test_data['Grade'])
    y_test = test_data['Grade_encoded']
else:
    y_test = None  # In case test labels are not provided

# Step 11: Select the same features and scale them
X_test = test_data[selected_features]
X_test_scaled = scaler.transform(X_test)

# Step 12: Predict on test data
y_test_pred = rf_model.predict(X_test_scaled)
y_test_proba = rf_model.predict_proba(X_test_scaled)[:, -1]

# Step 13: Evaluate the model if labels are available
if y_test is not None:
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_test_proba)
    logloss = log_loss(y_test, y_test_proba)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)

    # Confusion matrix-based specificity
    cm = confusion_matrix(y_test, y_test_pred)
    if cm.shape == (2, 2):
        tn = cm[0, 0]
        fp = cm[0, 1]
        specificity = tn / (tn + fp)
    else:
        specificity = "Not applicable for multi-class"

    # Print results
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