---
title: "Forecasting Used Cars & Trucks CPI: Regression vs. ARIMA"
collection: portfolio
permalink: /portfolio/stat429-used-cars-cpi
date: 2025-12-10
venue: "STAT 429 Final Project (UIUC)"
type: "Final Project"
excerpt: "Comparative analysis of OLS, WLS, and ARIMA models to forecast U.S. Used Vehicle CPI. The study identifies ARIMA(0,1,2) as the optimal model for short-term prediction, outperforming macroeconomic regression approaches."
author_profile: true
links:
  - url: "{{ '/files/STAT429-Final-Project.pdf' | relative_url }}"
    name: Final Report (PDF)
  - url: "{{ '/files/STAT-429-Presentation_Slides.pdf' | relative_url }}"
    name: Presentation Slides (PDF)
---

### Abstract
This project analyzes the **CPI of used cars and trucks** in the U.S. to identify the most accurate model for short-term forecasting. We compared four modeling approaches: **OLS regression** with a COVID-19 breakpoint, **Weighted Least Squares (WLS)** to address heteroscedasticity, **Lagged-Predictor Regression** based on CCF analysis, and an **ARIMA model**. While macroeconomic predictors like **New Vehicle CPI**, **Motor Fuel CPI**, and the **Federal Funds Rate** showed significant relationships, linear models failed to fully resolve residual autocorrelation. The study concluded that an **ARIMA(0,1,2)** model provided the best stability and predictive accuracy (MAPE ≈ 0.67%) for 5-month-ahead forecasts.

### Data
We utilized monthly U.S. time-series data from **FRED** spanning **January 2000 to April 2025**:
- **Used Cars & Trucks CPI** (Response, `CUSR0000SETA02`)
- **New Vehicles CPI** (Predictor, `CUSR0000SETA01`)
- **Motor Fuel CPI** (Predictor, `CUSR0000SETB`)
- **Federal Funds Effective Rate** (Predictor, `FEDFUNDS`)

A **COVID-19 dummy variable** was introduced for the post-January 2020 period to capture structural market shifts.

### Methodology
1.  **Exploratory Analysis**: Confirmed non-stationarity in all series; applied differencing ($d=1$) to achieve stationarity.
2.  **Regression Modeling**:
    *   **Model 1 (OLS)**: Baseline with contemporaneous predictors and COVID dummy ($R^2 \approx 0.75$).
    *   **Model 2 (WLS)**: Applied to fix non-constant variance found in OLS.
    *   **Model 3 (Lagged)**: Incorporated specific lags (New CPI $t-1$, Fuel $t-3$, Fed Funds $t-12$) identified via Cross-Correlation Function (CCF) analysis ($R^2 \approx 0.79$).
3.  **Time Series Modeling (ARIMA)**: Fitted an ARIMA(0,1,2) model based on ACF/PACF plots and AIC/BIC minimization.
4.  **Evaluation**: Models were compared using residual diagnostics (Ljung-Box, Normality) and out-of-sample forecast accuracy. The ARIMA model successfully eliminated autocorrelation artifacts present in the regression approaches.

### Downloads
- 📄 **Report**: [STAT429 Final Project Report]({{ 'files/STAT429 Final Project.pdf' | relative_url }})
- 📊 **Slides**: [Presentation Slides]({{ 'files/STAT 429 Presentation_Slides.pdf' | relative_url }})
