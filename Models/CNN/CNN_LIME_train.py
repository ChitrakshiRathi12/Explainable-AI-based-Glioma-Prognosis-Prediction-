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

# Load dataset
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")

# Encode the target variable (Grade)
le = LabelEncoder()
data['Grade_encoded'] = le.fit_transform(data['Grade'])

# Define selected features based on Pearson's Correlation, Mutual Information, and PCA
selected_features = ['Age_at_diagnosis', 'Gender', 'TP53']
X = data[selected_features]
y = data['Grade_encoded']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Reshape for Conv1D input
X_reshaped = np.expand_dims(X_scaled, axis=2)

# Build CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_reshaped.shape[1], 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model on full dataset
model.fit(X_reshaped, y, epochs=20, batch_size=32)

# Predict on same training data
y_pred = (model.predict(X_reshaped) > 0.5).astype("int32").flatten()

# Evaluate model
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')
accuracy = accuracy_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred)
logloss = log_loss(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)
kappa = cohen_kappa_score(y, y_pred)

# Specificity
cm = confusion_matrix(y, y_pred)
tn, fp = cm[0, 0], cm[0, 1]
specificity = tn / (tn + fp)

# Print metrics
print("\nClassification Report:\n", classification_report(y, y_pred, target_names=le.classes_))
print("\nAdditional Metrics:\n")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
print(f"Cohen’s Kappa: {kappa:.4f}")
print(f"Specificity: {specificity:.4f}")