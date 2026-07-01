import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_auc_score, log_loss, confusion_matrix, matthews_corrcoef, \
    cohen_kappa_score, accuracy_score, precision_score, recall_score
import numpy as np

from SVM_PFI_train import f1_score

# Step 1: Load the dataset
print("Loading dataset...")
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")
print("Dataset loaded successfully!")

# Step 2: Check the first few rows of the dataset to verify
print("Dataset preview:")
print(data.head())

# Step 3: Encode the target variable (Grade)
print("Encoding target variable...")
le = LabelEncoder()
data['Grade_encoded'] = le.fit_transform(data['Grade'])

# Step 4: Define features (X) and target (y)
selected_features = ['Age_at_diagnosis', 'IDH1', 'ATRX', 'TP53', 'PTEN', 'EGFR', 'Race', 'Gender']
X = data[selected_features]
y = data['Grade_encoded']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM with calibrated probabilities
base_model = LinearSVC()
svm_model = CalibratedClassifierCV(base_model)
svm_model.fit(X_scaled, y)

# Predict on the full dataset
y_pred = svm_model.predict(X_scaled)
y_proba = svm_model.predict_proba(X_scaled)

# Compute metrics
if len(np.unique(y)) == 2:
    roc_auc = roc_auc_score(y, y_proba[:, 1])
else:
    roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
logloss = log_loss(y, y_proba)
mcc = matthews_corrcoef(y, y_pred)
kappa = cohen_kappa_score(y, y_pred)

# Specificity (only valid for binary classification)
cm = confusion_matrix(y, y_pred)
if cm.shape == (2, 2):
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = tn / (tn + fp)
else:
    specificity = "Not applicable for multi-class classification"

# Print metrics
print("\nClassification Report (Training Set):\n", classification_report(y, y_pred, target_names=le.classes_))
print(f"\nTraining Performance Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Cohen’s Kappa: {kappa:.4f}")
print(f"Specificity: {specificity}")