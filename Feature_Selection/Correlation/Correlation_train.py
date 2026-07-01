import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")
# Compute Pearson's Correlation Matrix
correlation_matrix = data.drop(columns=['Grade','Primary_Diagnosis']).corr()


sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, linewidths=0.5)
plt.title("Pearson's Correlation Heatmap")
plt.show()

# Find highly correlated feature pairs (excluding self-correlations)
threshold = 0.4 # Adjusted threshold for strong correlation
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

high_corr = upper.stack().reset_index()
high_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
high_corr = high_corr[abs(high_corr['Correlation']) > threshold]

# Display highly correlated features
print("Highly correlated feature pairs (|correlation| > {:.1f}):\n".format(threshold))
print(high_corr.sort_values(by='Correlation', ascending=False))
high_corr.to_csv("highly_correlated_features.csv", index=False)