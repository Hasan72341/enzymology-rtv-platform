# Enzyme Selection & Bioprocess Optimization Report
## Dataset: GST
**Generated:** 2026-04-19 13:44:50
**Real-World Application:** Detoxification & bioremediation

---

## 1. Dataset Summary
- **EC Number:** 2.5.1.18
- **Total Samples:** 160
- **Unique Enzymes:** 5
- **Unique Organisms:** 32
- **log(kcat) Range:** -1.61 to 2.81


## 2. Model Performance (Model A: Enzyme Selection)

### Mean Performance Metrics
| Model | R² Score | RMSE | MSE | MAE | Spearman |
|-------|----------|------|-----|-----|----------|
| random_forest | 0.756 | 0.499 | 0.252 | 0.398 | 0.874 |
| xgboost ⭐ (Best) | 0.823 | 0.425 | 0.181 | 0.337 | 0.903 |
| hybrid_ensemble | 0.622 | 0.619 | 0.383 | 0.461 | 0.886 |
| linear_probe | 0.029 | 0.993 | 0.987 | 0.762 | 0.255 |

### Standard Deviations across Folds
| Model | R² Std | RMSE Std | MSE Std | MAE Std | Spearman Std |
|-------|--------|----------|---------|---------|--------------|
| random_forest | 0.029 | 0.054 | 0.054 | 0.038 | 0.029 |
| xgboost | 0.006 | 0.028 | 0.024 | 0.006 | 0.030 |
| hybrid_ensemble | 0.022 | 0.012 | 0.015 | 0.015 | 0.057 |
| linear_probe | 0.037 | 0.029 | 0.057 | 0.019 | 0.011 |

### Validation Summary
The best performing model is **xgboost** with a Spearman rank correlation of 0.903.
Its MAE of 0.337 means the average prediction error is approximately 0.337 log-units of kcat.


## 3. Top-3 Enzyme Variants

| Rank | UniProt ID | Organism | Predicted log(kcat) | Actual log(kcat) | Δ |
|------|------------|----------|---------------------|------------------|---|
| 1 | Q03013 | Atactodea striata | 2.821 | 2.814 | +0.006 |
| 2 | P10620 | Atactodea striata | 2.817 | 2.814 | +0.003 |
| 3 | O60760 | Atactodea striata | 2.810 | 2.814 | -0.005 |


## 4. Bioprocess Optimization (Model B)
- **Optimal pH:** 6.97
- **Optimal Temperature:** 29.8°C
- **Experimental pH range:** 2.0 units
- **Experimental temperature range:** 30.0°C

The optimization surface was learned from diverse experimental conditions.
## 5. Industrial Interpretation
### Detoxification & Bioremediation Application

The selected GST variant (Q03013) from *Atactodea striata* exhibits high intrinsic catalytic efficiency, as identified by sequence-informed machine learning.

Sensitivity analysis suggests that standard neutral pH and ambient temperature conditions already align with the enzyme's optimal operating window. Therefore, enzyme selection provides greater performance gains than bioprocess condition tuning for this reaction.


## 6. Visualizations
- Enzyme Ranking: `outputs/plots/gst_ranking.png`
- pH-Temperature Contour: `outputs/plots/gst_optimization_contour.png`
- Feature Importance: `outputs/plots/gst_feature_importance.png`

