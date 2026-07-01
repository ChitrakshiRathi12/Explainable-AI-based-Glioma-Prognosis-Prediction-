import CNN_MI_test
import CNN_PCA_test
import CNN_PFI_test
import CNN_LIME_test
import CNN_SHAP_test
import CNN_Correlation_test
import GBM_MI_test
import GBM_PCA_test
import GBM_PFI_test
import GBM_PFI_test
import GBM_Correlation_test
import GBM_SHAP_test
import RNN_MI_test
import RNN_PCA_test
import RNN_PFI_test
import RNN_SHAP_test
import RNN_Correlation_test
import RNN_LIME_test
import SVM_MI_test
import SVM_PFI_test
import SVM_PCA_test
import SVM_SHAP_test
import SVM_LIME_test
import SVM_Correlation_test
import XGBoost_MI_test
import XGBoost_PFI_test
import XGBoost_LIME_test
import XGBoost_SHAP_test
import XGBoost_PCA_test
import XGBoost_Correlation_test
import pandas as pd

# All model-feature selection modules
model_fs_modules = {
    "CNN": {
        "MI": CNN_MI_test,
        "PCA": CNN_PCA_test,
        "PFI": CNN_PFI_test,
        "LIME": CNN_LIME_test,
        "SHAP": CNN_SHAP_test,
        "Correlation": CNN_Correlation_test
    },
    "GBM": {
        "MI": GBM_MI_test,
        "PCA": GBM_PCA_test,
        "PFI": GBM_PFI_test,
        "SHAP": GBM_SHAP_test,
        "Correlation": GBM_Correlation_test
    },
    "RNN": {
        "MI": RNN_MI_test,
        "PCA": RNN_PCA_test,
        "PFI": RNN_PFI_test,
        "SHAP": RNN_SHAP_test,
        "Correlation": RNN_Correlation_test,
        "LIME": RNN_LIME_test
    },
    "SVM": {
        "MI": SVM_MI_test,
        "PCA": SVM_PCA_test,
        "PFI": SVM_PFI_test,
        "SHAP": SVM_SHAP_test,
        "LIME": SVM_LIME_test,
        "Correlation": SVM_Correlation_test
    },
    "XGBoost": {
        "MI": XGBoost_MI_test,
        "PCA": XGBoost_PCA_test,
        "PFI": XGBoost_PFI_test,
        "SHAP": XGBoost_SHAP_test,
        "LIME": XGBoost_LIME_test,
        "Correlation": XGBoost_Correlation_test
    }
}

results = []

# Loop through and extract accuracy
for model, fs_methods in model_fs_modules.items():
    for fs_method, module in fs_methods.items():
        try:
            acc = module.accuracy  # access variable 'accuracy'
            results.append({
                "Model": model,
                "Feature_Selection": fs_method,
                "Accuracy": acc
            })
        except AttributeError:
            print(f"Accuracy not found in {module.__name__}")
        except Exception as e:
            print(f"Error in {module.__name__}: {e}")

# Create and display comparison table
df = pd.DataFrame(results)
pivot_table = df.pivot(index="Model", columns="Feature_Selection", values="Accuracy")
print("\n=== Accuracy Table ===")
print(pivot_table)

# Best method per model
print("\n=== Best Feature Selection Method per Model ===")
print(pivot_table.idxmax(axis=1))

# # Optionally save
# pivot_table.to_excel("feature_selection_accuracy_comparison.xlsx")
