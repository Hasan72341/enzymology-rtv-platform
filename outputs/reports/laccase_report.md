# Enzyme Selection & Bioprocess Optimization Report
## Dataset: LACCASE
**Generated:** 2026-04-19 13:46:06
**Real-World Application:** Wastewater dye degradation

---

## 1. Dataset Summary
- **EC Number:** 1.10.3.2
- **Total Samples:** 290
- **Unique Enzymes:** 5
- **Unique Organisms:** 58
- **log(kcat) Range:** -1.81 to 3.07


## 2. Model Performance (Model A: Enzyme Selection)

### Mean Performance Metrics
| Model | R² Score | RMSE | MSE | MAE | Spearman |
|-------|----------|------|-----|-----|----------|
| random_forest | 0.735 | 0.550 | 0.302 | 0.418 | 0.837 |
| xgboost ⭐ (Best) | 0.785 | 0.481 | 0.236 | 0.337 | 0.857 |
| hybrid_ensemble | 0.631 | 0.660 | 0.441 | 0.529 | 0.827 |
| linear_probe | 0.181 | 0.978 | 0.958 | 0.784 | 0.265 |

### Standard Deviations across Folds
| Model | R² Std | RMSE Std | MSE Std | MAE Std | Spearman Std |
|-------|--------|----------|---------|---------|--------------|
| random_forest | 0.057 | 0.010 | 0.011 | 0.006 | 0.073 |
| xgboost | 0.091 | 0.065 | 0.063 | 0.026 | 0.070 |
| hybrid_ensemble | 0.015 | 0.073 | 0.096 | 0.051 | 0.049 |
| linear_probe | 0.064 | 0.051 | 0.099 | 0.071 | 0.035 |

### Validation Summary
The best performing model is **xgboost** with a Spearman rank correlation of 0.857.
Its MAE of 0.337 means the average prediction error is approximately 0.337 log-units of kcat.


## 3. Top-3 Enzyme Variants

| Rank | UniProt ID | Organism | Predicted log(kcat) | Actual log(kcat) | Δ |
|------|------------|----------|---------------------|------------------|---|
| 1 | C0HLV7 | Shiraia sp. SUPER-H168 | 3.081 | 3.071 | +0.010 |
| 2 | Q84J37 | Shiraia sp. SUPER-H168 | 3.052 | 3.071 | -0.018 |
| 3 | P07788 | Shiraia sp. SUPER-H168 | 3.038 | 3.071 | -0.033 |


## 4. Bioprocess Optimization (Model B)
- **Optimal pH:** 4.72
- **Optimal Temperature:** 54.8°C
- **Experimental pH range:** 5.8 units
- **Experimental temperature range:** 66.5°C

The optimization surface was learned from diverse experimental conditions.
## 5. Industrial Interpretation
### Wastewater Dye Degradation Application

The selected laccase variant (C0HLV7) from *Shiraia sp. SUPER-H168* exhibits high intrinsic catalytic efficiency, as identified by sequence-informed machine learning.

pH optimization analysis indicates operation between pH 4.7–5.2 for maximum activity in industrial wastewater treatment.


## 6. Visualizations
- Enzyme Ranking: `outputs/plots/laccase_ranking.png`
- pH-Temperature Contour: `outputs/plots/laccase_optimization_contour.png`
- Feature Importance: `outputs/plots/laccase_feature_importance.png`

