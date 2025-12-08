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

## Limitations
- Requires reliable SHAP estimates  
- Normalizing flows introduce computational cost  
- Residual structure matters: inappropriate residuals may affect detection power

---

## Citation
If you use IOM or its implementation, please cite:

> *[Authors]* (2025). **Detecting Influential Outliers in Machine Learning Models Using Normalizing Flows and SHAP Values.**

BibTeX:
```bibtex
@article{your_iom_paper,
  title={{Detecting Influential Outliers in Machine Learning Models Using Normalizing Flows and SHAP Values}},
  author={Author1 and Author2},
  year={2025},
  journal={...}
}
```

---

## References
- Lundberg, S. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.*
- Papamakarios, G. et al. (2021). *Normalizing Flows for Probabilistic Modeling and Inference.*

