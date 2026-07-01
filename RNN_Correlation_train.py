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
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")

# Preprocessing
# 1. Encode categorical variables
encoder = LabelEncoder()
data['Grade_encoded'] = encoder.fit_transform(data['Grade'])

# Step 2: Encode the target variable (Grade)
le = LabelEncoder()
data['Grade_encoded'] = le.fit_transform(data['Grade'])

# Step 3: Define features (X) and target (y)
selected_features = ['Age_at_diagnosis', 'IDH1', 'TP53', 'ATRX', 'CIC', 'FUBP1']
X = data[selected_features]
y = data['Grade_encoded']

# Step 4: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for RNN input (samples, time steps, features)
X_scaled = np.expand_dims(X_scaled, axis=1)

# Build the RNN model
model = Sequential([
    SimpleRNN(64, input_shape=(X_scaled.shape[1], X_scaled.shape[2]), activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model on the full dataset
history = model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=0)

# Predict on training data
y_proba = model.predict(X_scaled)
y_pred = np.argmax(y_proba, axis=1)

# Calculate metrics
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

# Specificity calculation (only valid for binary)
cm = confusion_matrix(y, y_pred)
if cm.shape == (2, 2):
    tn, fp = cm[0, 0], cm[0, 1]
    specificity = tn / (tn + fp)
else:
    specificity = "Not applicable for multi-class classification"

# Print the metrics
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