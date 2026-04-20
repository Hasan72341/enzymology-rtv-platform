# Data Pipeline Architecture

## Overview
The **Enzymology 3** data pipeline is designed to process raw empirical enzyme data into high-fidelity, machine-learning-ready feature matrices. It bridges the gap between raw FASTA sequences and the downstream predictive models.

## Data Ingestion & Preprocessing
The pipeline reads empirical datasets (e.g., `gst.csv`, `laccase.csv`, `lactase.csv`). Each dataset is validated against a strict schema requiring core features such as `sequence`, `ec`, `organism`, `log_kcat` (the target variable), and `n_measurements`.

- **Missing Value Imputation:** Optional parameters like `molecularWeight` and `kmValue` are safely imputed using dataset medians if absent.
- **Dynamic Inference Alignment:** During live inference, the pipeline pads missing scalar inputs with global fallback values to ensure exact dimensionality matches the trained tensors.

## Feature Engineering
The pipeline relies on a dual-stream feature engineering approach, combining deep learning sequence embeddings with classical biochemical scalars.

### 1. Structural Sequence Representations (ESM-2)
We utilize the **Evolutionary Scale Modeling 2 (ESM-2)** transformer (`esm2_t12_35M_UR50D`) from Meta/HuggingFace. 
- Sequences are passed through the 35-million parameter language model.
- **Mean Pooling** is applied to the final hidden state to generate a fixed **480-dimensional** embedding vector for each enzyme, capturing complex structural and evolutionary contexts without requiring explicit 3D folding.

### 2. Thermodynamic & Metadata Scalars
The scalar pipeline extracts and encodes crucial metadata:
- **Intrinsic Features:** Log-transformed $K_m$ (`log_kmValue`), Molecular Weight.
- **Taxonomic & Functional Encoding:** EC numbers are hierarchically split into 4 integer levels. Organism origins are mapped to simple domain indicators (e.g., `is_bacteria`, `is_fungi`).
- **Bioprocess Features:** `ph_opt` and `temp_opt` are tracked for downstream stability manifolds.

## Data Augmentation
To improve the robustness of the regression models—especially on smaller, highly-curated industrial datasets—a **SMOTE-like continuous interpolation** technique is applied exclusively during the training folds. This creates synthetic variants by interpolating between known data points, augmenting the dataset size and smoothing the decision boundaries.