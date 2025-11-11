---
title: "Bayesian Hierarchical Modeling of Crime Severity and Weapon Involvement (LAPD 2024–2025)"
collection: portfolio
permalink: /portfolio/stat431-bayesian-lapd
date: 2025-11-01
venue: "STAT 431 Final Project (UIUC)"
type: "Final Project Report"
excerpt: "Hierarchical Poisson model with Exponential–Gamma shrinkage prior; Gibbs sampling to estimate area-level crime rates and weapon-involved offense rates across 21 LAPD areas."
author_profile: true
use_math: true
links:
  - url: "{{ '/files/STAT431_Final_Project.pdf' | relative_url }}"
    name: Download Paper (PDF)
---

### Abstract
We study spatial variation in **serious crime** and **weapon-involved offenses** across the 21 LAPD geographic areas (2024–2025).
Let \(Y_i\) be the count of a specific offense type in area \(i\) with exposure \(N_i\) (total crimes).
We model \(Y_i \mid \lambda_i \sim \mathrm{Poisson}(N_i \lambda_i)\) and place a shared **shrinkage prior**
\(\lambda_i \mid \gamma \stackrel{iid}{\sim} \mathrm{Exponential}(\gamma),\ \gamma \sim \mathrm{Gamma}(a,b)\).
Posterior inference uses **Gibbs sampling** with conjugate full conditionals to estimate area-level rates,
overall mean rate \(E[1/\gamma]\), and to run posterior predictive checks and prior-sensitivity analysis.

### Data
Open-access **LAPD Crime Data (2020–Present)** from Data.gov.
We extract 2024–2025 records and compute, for each area:
- **Serious-crime proportion** (Primary Crime Code 1 < 300)
- **Weapon-involved proportion** (non-missing Weapon Involved Code)

Key fields include *Date of Occurrence*, *Police Area Code*, *Primary Crime Code 1*, *Weapon Involved Code*.

### Model
- **Likelihood:** \(Y_i \mid \lambda_i \sim \mathrm{Poisson}(N_i \lambda_i)\)
- **Prior:** \(\lambda_i \mid \gamma \sim \mathrm{Exponential}(\gamma),\ \gamma \sim \mathrm{Gamma}(a,b)\)
- **Full conditionals (Gibbs):**
  \[
  \lambda_i \mid \gamma,\mathbf{y} \sim \mathrm{Gamma}(y_i+1,\ N_i+\gamma),\qquad
  \gamma \mid \boldsymbol{\lambda},\mathbf{y} \sim \mathrm{Gamma}\!\Big(K+a,\ \sum_{i=1}^K \lambda_i + b\Big).
  \]

### Plan / Methodology
1. **Wrangle & aggregate** dataset; compute area-level exposures \(N_i\), serious and weapon-involved counts \(Y_i\).
2. **Specify priors** and implement **Gibbs sampler** in R.
3. **Diagnostics:** trace plots, ESS, \(\hat{R}\); posterior predictive checks; **prior sensitivity**.
4. **Reporting:** posterior means and 95% credible intervals for \(\lambda_i\); estimate overall mean rate \(E[1/\gamma]\); visualize spatial variation.

### Proposal Preview
<object data="{{ '/files/STAT431_Final_Project.pdf' | relative_url }}" type="application/pdf" width="100%" height="800">
  <p>PDF preview unavailable. <a href="{{ '/files/STAT431_Final_Project.pdf' | relative_url }}">Download Paper (PDF)</a></p>
</object>
