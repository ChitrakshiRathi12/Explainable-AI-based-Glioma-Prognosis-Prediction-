import matplotlib.pyplot as plt
import pandas as pd

# Import your model performance modules
import GBM_Correlation_test
import GBM_MI_test
import GBM_LIME_test
import GBM_PCA_test
import GBM_PFI_test
import GBM_SHAP_test

# Get metrics from each module
results = {
    "Correlation": GBM_Correlation_test.get_metrics(),
    "MI": GBM_MI_test.get_metrics(),
    "LIME": GBM_LIME_test.get_metrics(),
    "PCA": GBM_PCA_test.get_metrics(),
    "PFI": GBM_PFI_test.get_metrics(),
    "SHAP": GBM_SHAP_test.get_metrics()
}

# Convert the results to a DataFrame for easy plotting
df = pd.DataFrame(results)

# # Transpose to have models on x-axis and metrics as bars
# df = df.T
# Convert all values to numeric (if they are strings)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(how='all')
# df = df.drop(columns=["Log Loss"])
# Plot with models as bar series, and metrics (parameters) on the x-axis
df.plot(kind='bar', figsize=(12, 6))
plt.title("GBM Model Comparison Across Feature Selection Techniques")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y')
plt.show()
