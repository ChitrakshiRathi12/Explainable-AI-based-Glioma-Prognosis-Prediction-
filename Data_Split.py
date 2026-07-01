import pandas as pd
from sklearn.model_selection import train_test_split
# Load dataset
data = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\augmented_TCGA_GBM_LGG_Mutations_5x.csv")
# Split the data into training (70%) and testing (30%) sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Save training data to CSV
train_data.to_csv(r"C:\Users\Chitrakshi\Downloads\train_data.csv", index=False)

# Save testing data to CSV
test_data.to_csv(r"C:\Users\Chitrakshi\Downloads\test_data.csv", index=False)
# Optional: check the sizes
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)