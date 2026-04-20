# Machine Learning Architecture

The **Enzymology 3** platform relies on a dual-model architecture to handle both variant selection (ranking) and environmental bioprocess optimization.

## Model A: Enzyme Selection (Variant Ranking)
**Objective:** Predict the catalytic efficiency ($\log(k_{cat})$) of a given enzyme sequence to rank variants for high-throughput screening.

### Architectures
The pipeline tests multiple models and automatically selects the best performer based on the **Spearman Rank Correlation**.
- **Hybrid Neural-Classical Ensemble:** The flagship model. A custom PyTorch feed-forward network (FFN) with residual blocks processes the ESM-2 embeddings. Its outputs are concatenated with the original features and fed into an XGBoost regressor. Hyperparameters (learning rates, tree depths, blocks) are tuned automatically via **Optuna**.
- **Classical Baselines:** Random Forest, standard XGBoost, and Ridge Regression (Linear Probing).

### Cross-Validation Strategy
To ensure models generalize to novel enzyme families, we utilize **EC-Aware Cross-Validation**. Using `GroupKFold` on the EC classes prevents data leakage between closely related sequence clusters, offering a realistic estimation of real-world predictive precision.

---

## Model B: Bioprocess Optimization
**Objective:** Determine the optimal thermodynamic conditions (pH and Temperature) that maximize the predicted activity of a selected variant.

### Gaussian Process Regression
We utilize **Gaussian Process (GP) Regression** paired with a Radial Basis Function (RBF) kernel. The GP regressor maps the non-linear relationship between process conditions and catalytic efficiency.
- **Uncertainty & Grids:** A 50x50 search grid (2500 points) evaluates the pH and temperature bounds to construct a continuous 2D stability manifold.
- **Stability Penalty:** A thermodynamic constraint ($\lambda$) penalizes predictions that stray too far from known optimums, ensuring the suggested conditions remain biologically viable and preventing the model from predicting extreme, unrealistic peaks in regions lacking training support.