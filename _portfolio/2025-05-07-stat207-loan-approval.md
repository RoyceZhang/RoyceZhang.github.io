---
title: "Predicting Loan Approval with LASSO Logistic Regression"
collection: portfolio
permalink: /portfolio/stat207-loan-approval
date: 2025-05-07
venue: "STAT 207 Final Project (UIUC)"
excerpt: "End-to-end pipeline for loan-approval prediction: EDA, interaction diagnostics, LASSO feature selection with 5-fold CV, thresholding to control FPR."
author_profile: true
links:
  - url: "{{ '/files/STAT207%20Final%20Project.pdf' | relative_url }}"
    name: Download Paper (PDF)
  - url: "{{ '/files/STAT207_Presentation.pdf' | relative_url }}"
    name: Download Poster (PDF)
---

### Abstract
We build a predictive workflow for **loan approval** using a Kaggle-sourced synthetic dataset. After cleaning and exploratory analysis, we fit a **LASSO logistic regression** with 5-fold cross-validation to select informative features and reduce overfitting risk. The final model achieves **test AUC ≈ 0.935**, and we select an operating threshold to **minimize false positives** while keeping **TPR ≥ 0.60** (FPR ≈ 0.053), balancing risk control with access to credit.

### Methodology
1. **EDA & Interactions** — age/gender weak; **interest rate** & **previous defaults** strong; notable **age × defaults**、**amount × intent**.  
2. **Preprocessing** — standardize numerics, one-hot categoricals, remove age outliers (>100).  
3. **Model** — LASSO (liblinear), λ grid with **5-fold CV**; metric: ROC-AUC; check multicollinearity.  
4. **Thresholding** — pick threshold with **TPR > 0.60** minimizing FPR (~0.053 at ~0.597).

### Key Results
- **AUC (test): ~0.935**; robust generalization.  
- **Top signals**: previous default (−), interest rate (+), loan amount (+); intent varies by category.  
- **K-Means (k=3)**: segments align with risk profiles; one group shows higher approvals despite lower income but moderate amounts/high rates.

### Discussion & Limitations
Synthetic data limits external validity; future work: add **DTI, LTV, utilization**, compare forward/backward CV selection, test higher-order interactions.

### Downloads
- 📄 **Paper**: [STAT207 Final Project]({{ '/files/STAT207%20Final%20Project.pdf' | relative_url }})  
- 🖼️ **Poster**: [STAT207 Presentation]({{ '/files/STAT207_Presentation.pdf' | relative_url }})
