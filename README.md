# Explainable-AI-based-Glioma-Prognosis-Prediction-
Overview

Gliomas account for 30–33% of all brain tumors and nearly 78% of malignant brain tumors in adults. Accurate prognosis prediction is critical for personalized treatment planning, yet conventional diagnostic methods rely on invasive procedures and manual interpretation. This research addresses that gap by building a comprehensive ML/DL pipeline over the TCGA and CGGA datasets, enhanced with six feature selection and explainability techniques to identify the most prognostically significant biomarkers.

The best-performing configuration — XGBoost with PCA-based feature selection — achieves an ROC-AUC score above 0.98, demonstrating strong discriminative capability between Glioblastoma Multiforme (GBM) and Lower Grade Glioma (LGG).

Key Results

ModelBest Feature SelectionHighlight MetricXGBoostPCAROC-AUC > 0.98, MCC > 0.98CNNPCAHigh precision and recallRNNPCAMCC and Kappa > 0.85GBMPFIStable across all metricsSVMCorrelationROC-AUC > 0.85

Top prognostic biomarkers identified: IDH1, Age at Diagnosis, TP53, EGFR, ATRX, PTEN

Methodology

Datasets


TCGA — 839 records, 23 features after preprocessing
CGGA — 286 complete patient records
Features include clinical data (age, gender, race) and genetic biomarkers (IDH1, TP53, ATRX, EGFR, PTEN, CIC, and others)
Target classes: Glioblastoma Multiforme (GBM) and Lower Grade Glioma (LGG)


Pipeline

Dataset → Preprocessing → Augmentation → Train/Test Split (70:30)
       → Feature Extraction → Model Training → Evaluation

Data Preprocessing


Missing value imputation using statistical methods
Label encoding and one-hot encoding for categorical features
Numerical feature normalization
Data augmentation to prevent overfitting


Feature Selection & XAI Methods

MethodPurposeMutual Information (MI)Captures nonlinear dependencies between features and targetPearson Correlation (PCC)Measures linear correlation; identified IDH1, TP53, ATRX as keyPrincipal Component Analysis (PCA)Dimensionality reduction while retaining maximum varianceSHAPShapley-value-based global feature importance explanationLIMELocal surrogate model for per-instance prediction explanationPermutation Feature Importance (PFI)Importance via accuracy drop on feature shuffling

Models Implemented


SVM — Hyperplane-based binary classification
XGBoost — Regularized gradient boosted decision trees
GBM — Sequential weak learner ensemble
CNN — Spatial feature learning via convolution layers
RNN — Sequential dependency modelling via hidden states


Evaluation Metrics

Accuracy, Precision, Recall, F1-Score, ROC-AUC, Log Loss, MCC, Kappa Score, Specificity
