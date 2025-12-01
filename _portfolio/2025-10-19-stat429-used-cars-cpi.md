---
title: "Estimating Used Cars & Trucks CPI with Macroeconomic Predictors"
collection: portfolio
permalink: /portfolio/stat429-used-cars-cpi
date: 2025-10-19
venue: "STAT 429 Project Proposal (UIUC)"
type: "Project Proposal"
excerpt: "Proposal to model the CPI for used cars and trucks using new-vehicle CPI, motor-fuel CPI, and the federal funds rate, with a COVID-19 breakpoint."
author_profile: true
links:
  - url: "{{ '/files/STAT429_Project%20Proposal.pdf' | relative_url }}"
    name: Download Proposal (PDF)
---

### Abstract
We propose a monthly time-series model for the **CPI of used cars and trucks** in the U.S. The response will be explained by three macro predictors: **new-vehicle CPI**, **motor-fuel CPI**, and the **federal funds effective rate**. To capture structural change due to the **COVID-19** period, we plan to include a **breakpoint at January 2020** via a dummy (and interactions as needed). The goal is an interpretable regression that can **explain historical movements** and **produce short-horizon forecasts** of the used-vehicle CPI. :contentReference[oaicite:0]{index=0}

### Data
Monthly, seasonally adjusted U.S. series from **FRED** (21st century through **Aug 2025**):
- **Used cars & trucks CPI** (BLS, series **CUSR0000SETA02**) — response  
- **New vehicles CPI** (BLS, **CUSR0000SETA01**) — predictor  
- **Motor fuel CPI** (BLS, **CUSR0000SETB**) — predictor  
- **Federal funds effective rate** (Board of Governors, **FEDFUNDS**) — predictor  
All series share monthly frequency and will be aligned by date. :contentReference[oaicite:1]{index=1}

### Methodology (Plan)
1. **Download & align** the four series; handle missing values and confirm seasonal adjustment status.  
2. **Baseline OLS** on levels (or appropriate transforms) with contemporaneous predictors.  
3. **COVID breakpoint**: add a **post-2020.01 dummy** and, if needed, **interactions** with predictors to allow slope shifts.  
4. **Diagnostics**: residual autocorrelation (ACF/PACF), heteroskedasticity, multicollinearity (VIF), stability checks.  
5. **Model comparison**: plain OLS vs. dummy/interaction model using lack-of-fit, adjusted \(R^2\), information criteria, and forecast performance on a hold-out period.  
6. **Forecasting**: generate **5-month-ahead** point forecasts (and CIs) using the preferred model. :contentReference[oaicite:2]{index=2}

### Downloads
- 📄 **Proposal**: [STAT429 Project Proposal]({{ '/files/STAT429_Project%20Proposal.pdf' | relative_url }})
- 📄 **Report**: [STAT429 Final Project Report]({{ '/files/STAT-429-Final-Project.pdf' | relative_url }})
