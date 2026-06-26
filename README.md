# Influential Outlier Metric (IOM)

## Overview
This repository accompanies the paper introducing the **Influential Outlier Metric (IOM)** — a general framework for detecting influential outliers in statistical and machine learning (ML) models.

The method defines influence using **SHapley Additive exPlanations (SHAP)** values and applies **normalizing flows** to transform model-based diagnostics into a known reference distribution. The resulting test statistic follows the product of two independent chi-square distributions, enabling principled statistical inference for outlier detection.

---

## Key Contributions
✔ **Model-agnostic influence** using SHAP values  
✔ **Normalizing flows** to Gaussianize SHAP values and residuals  
✔ **Closed-form reference distribution**: Product of χ² random variables  
✔ **Statistical hypothesis testing** for influential outlier detection  
✔ Demonstrated across ML models:
- Linear regression
- Neural networks
- Random forests
- Gradient-boosted trees

---

## Method Summary
1. Compute **SHAP values** for each observation to quantify influence on predictions.  
2. Apply **normalizing flows** to transform:
   - SHAP values → Gaussian
   - Residuals → Gaussian
3. Compute the **Influential Outlier Metric (IOM)** as:

$$
\text{IOM} = \chi^2_{\text{transformed SHAP}} \times \chi^2_{\text{transformed residuals}}
$$

4. Compare IOM values to critical values derived from the **product chi-square distribution**.  
5. Flag observations as influential outliers at significance level **α**.

---

## Why IOM?
Traditional influence measures (e.g., Cook’s Distance) are tied to specific statistical models and may not generalize to nonlinear ML methods.  
IOM provides:

| Feature | Benefit |
|--------|---------|
| SHAP-based influence | Works with any model |
| Flow-based distribution alignment | Valid inference even with non-Gaussian diagnostics |
| Product χ² reference | Transparent, interpretable thresholding |

---

## Notebooks
- `BART.ipynb` compares the IOM to the methodology in Pratola 2022
- `deep_learning_cats_dogs.ipynb` compares the IOM to influence functions in deep learning using the Cats and Dogs dataset
- `Countries.ipynb` investigates the IOM in a real world data set (From the World Development Indicators) using a boosted-tree algorithm
- `Comparison_ols_iom.ipynb` compares the IOM to OLS and a random forest using the Prestige dataset from the `R` package `carData`
- `nf_full.ipynb`, `nf_full_architecture.ipynb`, `nf_full_moons.ipynb` evaluate the normalizing flow change-of-measure technique

---

## Data sets
- BART: https://bitbucket.org/mpratola/openbt/wiki/Home
- Cats and Dogs: https://www.kaggle.com/code/raqhea/cats-vs-dogs-pytorch-resnet18
- Countries: https://www.worldbank.org/ext/en/home
- Prestige: https://cran.r-project.org/web/packages/carData/index.html
