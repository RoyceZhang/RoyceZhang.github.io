---
title: "Bayesian Hierarchical Modeling of Crime Severity and Weapon Involvement (LAPD 2024–2025)"
collection: portfolio
permalink: /portfolio/stat431-bayesian-lapd
date: 2025-11-01
venue: "STAT 431 Final Project (UIUC)"
type: "Final Project Report"
excerpt: "Hierarchical Poisson model with an Exponential–Gamma shrinkage prior; Gibbs sampling to estimate area-level crime and weapon-involved offense rates across 21 LAPD areas."
author_profile: true
links:
  - url: "{{ '/files/STAT431_Final_Project.pdf' | relative_url }}"
    name: Download Paper (PDF)
---

### Abstract
We analyze serious crime and weapon-involved offenses across the 21 LAPD geographic areas for 2024–2025.  
Each area’s count of a target offense is modeled as a Poisson outcome with exposure equal to the total number of crimes in that area.  
To borrow strength across areas, we place a common shrinkage prior on the area-level rates and put a Gamma hyperprior on the shrinkage parameter.  
Posterior inference is performed with Gibbs sampling. We report area-level rate estimates, overall mean rate, uncertainty intervals, posterior predictive checks, and prior-sensitivity analysis.

### Data
Source: LAPD “Crime Data from 2020 to Present” (Data.gov).  
From the 2024–2025 subset, for each area we compute:
- Serious-crime proportion (Primary Crime Code 1 below 300).
- Weapon-involved proportion (weapon code present).
Key variables include date of occurrence, police area code, primary crime code 1, and weapon code.

### Model
- **Likelihood:** For each area, the observed count is treated as coming from a Poisson distribution with mean equal to exposure times the area’s underlying rate.  
- **Prior:** Each area’s rate has an Exponential prior with a shared shrinkage parameter; that parameter has a Gamma hyperprior.  
- **Gibbs sampling:**  
  - Update each area’s rate from its Gamma full conditional given the data and the current shrinkage parameter.  
  - Update the shrinkage parameter from its Gamma full conditional given the current rates.  
This yields posterior draws for all rates and hyperparameters.

### Methodology
1. **Wrangle & aggregate** the raw data; compute exposure (total crimes) and target counts per area.  
2. **Set weakly informative hyperparameters** for the prior and run the Gibbs sampler in R.  
3. **Diagnostics**: trace plots, effective sample size, R-hat; posterior predictive checks; prior sensitivity.  
4. **Reporting**: posterior means and 95% credible intervals for each area’s rate; estimate the overall mean rate; visualize spatial variation across areas.

### Proposal Preview
- 📄 **Proposal**: [STAT431 Final Project Proposal]({{ '/files/STAT431_Final_Project.pdf' | relative_url }})

