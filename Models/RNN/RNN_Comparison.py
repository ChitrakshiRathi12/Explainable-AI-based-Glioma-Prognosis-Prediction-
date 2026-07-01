import matplotlib.pyplot as plt
import pandas as pd

# Import your model performance modules
import RNN_Correlation_test
import RNN_MI_test
import RNN_LIME_test
import RNN_PCA_test
import RNN_PFI_test
import RNN_SHAP_test

print("=== Debug: Checking all get_metrics() outputs ===")
for name, module in {
    "Correlation": RNN_Correlation_test,
    "MI": RNN_MI_test,
    "LIME": RNN_LIME_test,
    "PCA": RNN_PCA_test,
    "PFI": RNN_PFI_test,
    "SHAP": RNN_SHAP_test
}.items():
    metrics = module.get_metrics()
    print(f"{name}: {metrics}")
# Get metrics from each module
results = {
    "Correlation": RNN_Correlation_test.get_metrics(),
    "MI": RNN_MI_test.get_metrics(),
    "LIME": RNN_LIME_test.get_metrics(),
    "PCA": RNN_PCA_test.get_metrics(),
    "PFI": RNN_PFI_test.get_metrics(),
    "SHAP": RNN_SHAP_test.get_metrics()
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
plt.title("RNN Model Comparison Across Feature Selection Techniques")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y')
plt.show()