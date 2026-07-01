import pandas as pd
import numpy as np
from keras.src.layers import SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score, matthews_corrcoef, roc_auc_score, confusion_matrix, log_loss
)
from RNN_PCA_train import le, scaler,model, selected_features
# Step 1: Load test dataset
test_data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Step 2: Encode the target variable in test set (if available)
if 'Grade' in test_data.columns:
    test_data['Grade_encoded'] = le.transform(test_data['Grade'])
    y_test = test_data['Grade_encoded']
else:
    y_test = None

# Step 3: Select same features and scale them
X_test = test_data[selected_features]
X_test_scaled = scaler.transform(X_test)

# Step 4: Reshape for RNN input
X_test_scaled = np.expand_dims(X_test_scaled, axis=1)

# Step 5: Make predictions
y_test_proba = model.predict(X_test_scaled)
y_test_pred = np.argmax(y_test_proba, axis=1)

# Step 6: Evaluate if ground truth available
if y_test is not None:
    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    logloss = log_loss(y_test, y_test_proba)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    kappa = cohen_kappa_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp)
    else:
        specificity = "Not applicable for multi-class classification"

    # Print test set evaluation
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
    print("\nTest data has no true labels. Showing predictions only:")
    test_data['Predicted_Grade'] = le.inverse_transform(y_test_pred)
    print(test_data[['Predicted_Grade']])
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