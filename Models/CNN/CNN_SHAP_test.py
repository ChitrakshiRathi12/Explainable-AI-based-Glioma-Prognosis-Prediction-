import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, log_loss, cohen_kappa_score,
    matthews_corrcoef, confusion_matrix
)
from CNN_SHAP_train import model,le, scaler, selected_features

# Load test dataset
test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Encode target variable (if available in test data for evaluation)
if 'Grade' in test_data.columns:
    test_data['Grade_encoded'] = le.transform(test_data['Grade'])
    y_test = test_data['Grade_encoded']
else:
    y_test = None  # for blind test without labels

# Select and scale features
X_test = test_data[selected_features]
X_test_scaled = scaler.transform(X_test)

# Reshape for Conv1D
X_test_reshaped = np.expand_dims(X_test_scaled, axis=2)

# Predict
y_test_pred_prob = model.predict(X_test_reshaped)
y_test_pred = (y_test_pred_prob > 0.5).astype("int32").flatten()

# If labels are available, evaluate
if y_test is not None:
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred)
    logloss = log_loss(y_test, y_test_pred)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = tn / (tn + fp)

    # Print evaluation metrics
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=le.classes_))
    print("\nTest Set Metrics:\n")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Log Loss: {logloss:.4f}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    print(f"Cohen’s Kappa: {kappa:.4f}")
    print(f"Specificity: {specificity:.4f}")
else:
    print("Test predictions (no ground truth available):")
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