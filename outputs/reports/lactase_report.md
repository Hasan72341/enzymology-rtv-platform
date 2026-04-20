# Enzyme Selection & Bioprocess Optimization Report
## Dataset: LACTASE
**Generated:** 2026-04-19 13:46:29
**Real-World Application:** Lactose-free dairy processing

---

## 1. Dataset Summary
- **EC Number:** 3.2.1.23
- **Total Samples:** 145
- **Unique Enzymes:** 5
- **Unique Organisms:** 29
- **log(kcat) Range:** -1.23 to 3.93


## 2. Model Performance (Model A: Enzyme Selection)

### Mean Performance Metrics
| Model | R² Score | RMSE | MSE | MAE | Spearman |
|-------|----------|------|-----|-----|----------|
| random_forest | 0.734 | 0.467 | 0.218 | 0.357 | 0.582 |
| xgboost ⭐ (Best) | 0.754 | 0.450 | 0.203 | 0.308 | 0.716 |
| hybrid_ensemble | 0.633 | 0.550 | 0.303 | 0.420 | 0.663 |
| linear_probe | -0.206 | 0.997 | 0.996 | 0.803 | 0.181 |

### Standard Deviations across Folds
| Model | R² Std | RMSE Std | MSE Std | MAE Std | Spearman Std |
|-------|--------|----------|---------|---------|--------------|
| random_forest | 0.042 | 0.005 | 0.004 | 0.008 | 0.099 |
| xgboost | 0.031 | 0.011 | 0.010 | 0.020 | 0.142 |
| hybrid_ensemble | 0.040 | 0.019 | 0.021 | 0.021 | 0.074 |
| linear_probe | 0.129 | 0.035 | 0.070 | 0.019 | 0.021 |

### Validation Summary
The best performing model is **xgboost** with a Spearman rank correlation of 0.716.
Its MAE of 0.308 means the average prediction error is approximately 0.308 log-units of kcat.


## 3. Top-3 Enzyme Variants

| Rank | UniProt ID | Organism | Predicted log(kcat) | Actual log(kcat) | Δ |
|------|------------|----------|---------------------|------------------|---|
| 1 | O07012 | Kluyveromyces marxianus | 3.937 | 3.932 | +0.004 |
| 2 | P23780 | Kluyveromyces marxianus | 3.932 | 3.932 | -0.000 |
| 3 | P00722 | Kluyveromyces marxianus | 3.932 | 3.932 | -0.000 |


## 4. Bioprocess Optimization (Model B)
- **Optimal pH:** 6.52
- **Optimal Temperature:** 48.8°C
- **Experimental pH range:** 4.9 units
- **Experimental temperature range:** 75.0°C

The optimization surface was learned from diverse experimental conditions.
## 5. Industrial Interpretation
### Lactose-Free Dairy Processing Application

The selected lactase enzyme (O07012) from *Kluyveromyces marxianus* exhibits high intrinsic catalytic efficiency, as identified by sequence-informed machine learning.

Temperature optimization is particularly important for dairy processing, with optimal activity near 49°C. This aligns well with typical pasteurization temperatures.


## 6. Visualizations
- Enzyme Ranking: `outputs/plots/lactase_ranking.png`
- pH-Temperature Contour: `outputs/plots/lactase_optimization_contour.png`
- Feature Importance: `outputs/plots/lactase_feature_importance.png`

