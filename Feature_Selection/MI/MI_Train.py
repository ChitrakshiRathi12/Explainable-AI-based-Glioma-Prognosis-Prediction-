import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# Load data
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv")

# Convert target variable 'Grade' to binary
data['Grade'] = data['Grade'].map({'LGG': 0, 'GBM': 1})

# Drop 'Primary_Diagnosis'
X = data.drop(columns=['Grade', 'Primary_Diagnosis'])
y = data['Grade']

# Compute Mutual Information Scores
mi_scores = mutual_info_classif(X, y)

# Create a DataFrame to display scores
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='MI Score', ascending=False)

# Plot Mutual Information Scores
plt.figure(figsize=(10, 6))
sns.barplot(x='MI Score', y='Feature', hue='Feature', data=mi_scores_df, palette='viridis', legend=False)
plt.title("Mutual Information Scores for Feature Selection")
plt.xlabel("Mutual Information Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# Display top MI scores
print(mi_scores_df.head(10))
