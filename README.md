# Explainable AI-based Glioma Prognosis Prediction

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Paper](https://img.shields.io/badge/Paper-Under%20Review-yellow)

> An AI-driven framework for glioma prognosis prediction integrating machine learning, deep learning, and Explainable AI techniques on clinical and molecular biomarker datasets.

---

## Overview

Gliomas account for 30–33% of all brain tumors and nearly 78% of malignant brain tumors in adults. Accurate prognosis prediction is critical for personalized treatment planning, yet conventional diagnostic methods rely on invasive procedures and manual interpretation. This research addresses that gap by building a comprehensive ML/DL pipeline over the TCGA and CGGA datasets, enhanced with six feature selection and explainability techniques to identify the most prognostically significant biomarkers.

The best-performing configuration — **XGBoost with PCA-based feature selection** — achieves an ROC-AUC score above **0.98**, demonstrating strong discriminative capability between Glioblastoma Multiforme (GBM) and Lower Grade Glioma (LGG).

---

## Research Paper

**Title:** Explainable AI-based Glioma Prognosis Prediction

**Authors:** Chitrakshi Surendra Rathi (Capgemini, India) · Valentina Rani Basker (St. Francis Institute of Technology)

**Status:** 🔬 Under preparation for submission to a peer-reviewed journal (IEEE / Springer)

---

## Key Results

| Model | Best Feature Selection | Highlight Metric |
|---|---|---|
| XGBoost | PCA | ROC-AUC > 0.98, MCC > 0.98 |
| CNN | PCA | High precision and recall |
| RNN | PCA | MCC and Kappa > 0.85 |
| GBM | PFI | Stable across all metrics |
| SVM | Correlation | ROC-AUC > 0.85 |

**Top prognostic biomarkers identified:** IDH1, Age at Diagnosis, TP53, EGFR, ATRX, PTEN

---

## Methodology

### Datasets
- **TCGA** — 839 records, 23 features after preprocessing
- **CGGA** — 286 complete patient records
- Features include clinical data (age, gender, race) and genetic biomarkers (IDH1, TP53, ATRX, EGFR, PTEN, CIC, and others)
- Target classes: Glioblastoma Multiforme (GBM) and Lower Grade Glioma (LGG)

### Pipeline

```
Dataset → Preprocessing → Augmentation → Train/Test Split (70:30)
       → Feature Extraction → Model Training → Evaluation
```

### Data Preprocessing
- Missing value imputation using statistical methods
- Label encoding and one-hot encoding for categorical features
- Numerical feature normalization
- Data augmentation to prevent overfitting

### Feature Selection & XAI Methods

| Method | Purpose |
|---|---|
| Mutual Information (MI) | Captures nonlinear dependencies between features and target |
| Pearson Correlation (PCC) | Measures linear correlation; identified IDH1, TP53, ATRX as key |
| Principal Component Analysis (PCA) | Dimensionality reduction while retaining maximum variance |
| SHAP | Shapley-value-based global feature importance explanation |
| LIME | Local surrogate model for per-instance prediction explanation |
| Permutation Feature Importance (PFI) | Importance via accuracy drop on feature shuffling |

### Models Implemented
- **SVM** — Hyperplane-based binary classification
- **XGBoost** — Regularized gradient boosted decision trees
- **GBM** — Sequential weak learner ensemble
- **CNN** — Spatial feature learning via convolution layers
- **RNN** — Sequential dependency modelling via hidden states

### Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, ROC-AUC, Log Loss, MCC, Kappa Score, Specificity

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ChitrakshiRathi12/Explainable-AI-based-Glioma-Prognosis-Prediction-.git
cd Explainable-AI-based-Glioma-Prognosis-Prediction-

# Create virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Requirements

```txt
tensorflow>=2.10
scikit-learn>=1.1
xgboost>=1.7
autogluon
catboost
shap
lime
pandas
numpy
matplotlib
seaborn
scipy
```

---

## Usage

```bash
# Run feature selection
python Feature_selection.py

# Train CNN with PCA features (best performing)
python CNN_PCA_train.py

# Evaluate on test set
python CNN_PCA_test.py

# Compare all models across feature selection methods
python CNN_comparison.py
```

---

## Findings Summary

- **PCA** was the best-performing feature selection technique for CNN, RNN, and XGBoost — achieving ROC-AUC scores consistently above 0.98 for XGBoost
- **PFI** delivered the best results for GBM by identifying the most impactful features
- **Correlation-based** selection worked best for SVM by improving class separation
- **LIME** underperformed relative to other methods, particularly in MCC and Kappa metrics
- SHAP analysis confirmed **IDH1, Age at Diagnosis, TP53, and PTEN** as the most critical prognostic features
- XGBoost + PCA achieved the best overall results across all evaluation metrics, with minimal log loss

---

## Clinical Significance

This framework assists neuro-oncologists by:
- Providing **transparent, explainable predictions** via SHAP and LIME visualisations
- Identifying **key genetic biomarkers** (IDH1, TP53, EGFR, ATRX) that drive prognosis
- Enabling **personalized treatment planning** through patient-level prediction explanations
- Supporting early, non-invasive prognosis assessment from clinical and molecular data

---

## Future Work

- Incorporate MRI imaging, histology, and genomics data for multimodal prediction
- Explore Transformer, GNN, and advanced ensemble architectures
- Clinical validation with real-world patient cohorts
- Develop an online decision-support tool for neuro-oncologists

---

## Authors

**Chitrakshi Surendra Rathi**
Software Engineer, Capgemini India
GenAI Team — Agentic Systems & Automation Testing
[GitHub](https://github.com/ChitrakshiRathi12) · [LinkedIn](https://linkedin.com/in/chitrakshirathi)

**Valentina Rani Basker**
Assistant Professor, St. Francis Institute of Technology, Mumbai

---

## Citation

If you use this work, please cite:

```bibtex
@article{rathi2025glioma,
  title     = {Explainable AI-based Glioma Prognosis Prediction},
  author    = {Rathi, Chitrakshi Surendra and Basker, Valentina Rani},
  journal   = {Under Review},
  year      = {2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.(PCC)Measures linear correlation; identified IDH1, TP53, ATRX as keyPrincipal Component Analysis (PCA)Dimensionality reduction while retaining maximum varianceSHAPShapley-value-based global feature importance explanationLIMELocal surrogate model for per-instance prediction explanationPermutation Feature Importance (PFI)Importance via accuracy drop on feature shuffling

Models Implemented


SVM — Hyperplane-based binary classification
XGBoost — Regularized gradient boosted decision trees
GBM — Sequential weak learner ensemble
CNN — Spatial feature learning via convolution layers
RNN — Sequential dependency modelling via hidden states


Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, ROC-AUC, Log Loss, MCC, Kappa Score, Specificity
