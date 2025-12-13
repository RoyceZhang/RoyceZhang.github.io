---
title: "Bayesian Hierarchical Modeling of Serious Crime Rates (LAPD 2024–2025)"
collection: portfolio
permalink: /portfolio/stat431-bayesian-lapd
date: 2025-12-12
venue: "STAT 431 Final Project (UIUC)"
type: "Final Project Report"
excerpt: "Hierarchical Beta-Binomial model implemented in JAGS to estimate serious crime probabilities across 21 LAPD areas, using Gibbs sampling to stabilize estimates for high-variance districts."
author_profile: true
links:
  - url: "{{ '/files/STAT431_Final_Project_Report.pdf' | relative_url }}"
    name: Download Paper (PDF)
---

### Abstract
This project investigates spatial heterogeneity in serious crime rates across the 21 geographic areas of the Los Angeles Police Department (LAPD). Using data from **January 2024 to October 2025**, we modeled the count of serious offenses (homicide, robbery, aggravated assault) as a Binomial outcome relative to the total reported crimes in each area. A **Bayesian hierarchical Beta-Binomial model** was implemented to borrow strength across districts, stabilizing estimates for areas with smaller sample sizes. Posterior inference performed via **Gibbs sampling** revealed significant geographic disparities, with the **77th Street** area exhibiting the highest risk ($\approx 0.15$) and **West LA** the lowest ($\approx 0.02$).

### Data
**Source**: LAPD "Crime Data from 2020 to Present" (Data.gov).  
**Study Period**: January 1, 2024 – October 31, 2025.  
**Variables**:
- **Unit of Analysis**: 21 LAPD Geographic Areas.
- **Serious Crime**: Incidents with `Primary Crime Code 1` < 300 (e.g., homicide, rape, robbery, aggravated assault).
- **Exposure**: Total number of crime incidents reported in the area ($N_i$).
- **Response**: Count of serious crimes ($y_i$).

### Model
- **Likelihood**: The number of serious crimes $y_i$ in area $i$ follows a Binomial distribution:  
  $y_i | p_i, N_i \sim \text{Binomial}(N_i, p_i)$.
- **Prior**: Area-specific rates $p_i$ share a conjugate Beta prior:  
  $p_i \sim \text{Beta}(\alpha, \beta)$.
- **Hyperprior**: The shape parameters $\alpha$ and $\beta$ are assigned independent weakly informative Gamma priors:  
  $\alpha, \beta \sim \text{Gamma}(2, 0.5)$.
- **Computation**: Implemented in **JAGS** using Gibbs sampling. We ran 3 parallel chains with 100,000 iterations each (after 1,500 burn-in) to approximate the posterior distribution.

### Methodology
1.  **Data Wrangling**: Filtered raw incident logs for the 2024–2025 period and aggregated counts by police division to calculate empirical proportions.
2.  **MCMC Simulation**: Executed the JAGS sampler to generate posterior draws for $\alpha$, $\beta$, and all area-specific rates $p_i$.
3.  **Diagnostics**: Confirmed convergence using **Trace Plots** and the **Gelman-Rubin statistic** (PSRF $\approx 1.0$). Validated model fit via **Posterior Predictive Checks** (Bayesian p-value $\approx 0.52$).
4.  **Sensitivity Analysis**: Compared results against diffuse priors ($\text{Gamma}(1, 0.25)$) to ensure robustness of the posterior estimates. 

### Downloads
- 📄 **Report**: [STAT431 Final Project Report]({{ 'files/STAT431_Final_Project_Report.pdf' | relative_url }})
