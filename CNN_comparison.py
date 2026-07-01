import matplotlib.pyplot as plt
import pandas as pd

# Import your model performance modules
import CNN_Correlation_test
import CNN_MI_test
import CNN_LIME_test
import CNN_PCA_test
import CNN_PFI_test
import CNN_SHAP_test

# Get metrics from each module
results = {
    "Correlation": CNN_Correlation_test.get_metrics(),
    "MI": CNN_MI_test.get_metrics(),
    "LIME": CNN_LIME_test.get_metrics(),
    "PCA": CNN_PCA_test.get_metrics(),
    "PFI": CNN_PFI_test.get_metrics(),
    "SHAP": CNN_SHAP_test.get_metrics()
}


# Convert the results to a DataFrame for easy plotting
df = pd.DataFrame(results)

# # Transpose to have models on x-axis and metrics as bars
# df = df.T
# Convert all values to numeric (if they are strings)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(how='all')

# Plot with models as bar series, and metrics (parameters) on the x-axis
df.plot(kind='bar', figsize=(12, 6))
plt.title("CNN Model Comparison Across Feature Selection Techniques")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y')
plt.show()
