import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load dataset
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv")

# Drop 'Primary_Diagnosis' and separate features and target
X = data.drop(columns=['Grade', 'Primary_Diagnosis'])
feature_names = X.columns

# Apply PCA
pca = PCA(n_components=10)
principal_components = pca.fit_transform(X)

# Explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot cumulative explained variance
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), explained_variance, marker='o', linestyle='--', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA - Explained Variance')
plt.grid(True)
plt.show()

# Create DataFrame for PCA component loadings
pca_loadings_df = pd.DataFrame(pca.components_.T, index=feature_names, columns=[f'PC{i+1}' for i in range(10)])

# Save to CSV
pca_loadings_df.to_csv(r"C:\Users\Chitrakshi\Downloads\pca_feature_importance.csv", index=True)

# Print top features contributing to each component
print("Top Contributing Features for Each Principal Component:")
for i in range(10):
    top_features = pca_loadings_df.iloc[:, i].abs().sort_values(ascending=False).head(5)
    print(f"\nPC{i+1}:")
    print(top_features)