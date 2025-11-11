---
title: "Predicting Loan Approval with LASSO Logistic Regression"
collection: portfolio
permalink: /portfolio/Stat207-loan-approval
date: 2025-05-07
venue: "STAT 207 Final Project (UIUC)"
excerpt: "End-to-end pipeline for loan-approval prediction: EDA, interaction diagnostics, LASSO feature selection with 5-fold CV, thresholding to control FPR."
author_profile: true
links:
  - url: /files/stat207-final-project.pdf
    name: Download Paper (PDF)
  - url: /files/stat207-presentation.pdf
    name: Download Poster (PDF)
---

### Abstract
We build a predictive workflow for **loan approval** using a Kaggle-sourced synthetic dataset. After cleaning and exploratory analysis, we fit a **LASSO logistic regression** with 5-fold cross-validation to select informative features and reduce overfitting risk. The final model achieves a **test AUC ≈ 0.935**, and we select an operating threshold to **minimize false positives** while keeping **TPR ≥ 0.60** (FPR ≈ 0.053). *This balances risk control (avoiding approvals for likely defaulters) with access to credit.* :contentReference[oaicite:4]{index=4}

### Methodology
1. **EDA & Interactions** — Box/violin and bar charts indicated weak effects for age and gender, but strong signals for **interest rate** and **previous defaults**; interaction checks highlighted **age × defaults** and **amount × intent**. :contentReference[oaicite:5]{index=5}  
2. **Preprocessing** — Standardize numeric features; one-hot encode categoricals; remove age outliers (>100). :contentReference[oaicite:6]{index=6}  
3. **Model** — LASSO (liblinear) with λ grid via **5-fold CV**; evaluate by ROC-AUC; confirm no severe multicollinearity. :contentReference[oaicite:7]{index=7}  
4. **Thresholding** — Choose threshold with **TPR > 0.60** minimizing FPR (≈0.053 at threshold ≈0.597) for conservative approvals. :contentReference[oaicite:8]{index=8}

### Key Results
- **AUC (test): ~0.935**; robust generalization.  
- **Top signals**: previous default (strong negative), loan interest rate (+), loan amount (+); loan intent effects vary by category. :contentReference[oaicite:9]{index=9}  
- **Cluster insight (K-Means, k=3)**: segments align with risk profiles; one group shows higher approvals despite lower income but moderate amounts/high rates—suggesting unobserved factors. :contentReference[oaicite:10]{index=10}

### Discussion & Limitations
Synthetic data limits external validity; feature space could include **DTI, LTV, utilization**; future work should compare alternative models and selection (e.g., forward/backward CV), and test higher-order interactions. :contentReference[oaicite:11]{index=11}

### Downloads
- 📄 **Paper**: [STAT207 Final Project](/files/stat207-final-project.pdf)  
- 🖼️ **Poster**: [STAT207 Presentation](/files/stat207-presentation.pdf)
