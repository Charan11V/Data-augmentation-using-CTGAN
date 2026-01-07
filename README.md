# ~~~~~~~~"Each notebooks in each folder have their own set of documentations in '.md' format that can guide you with those notebooks."~~~~~~~~
# ~~~~~~~~"You may not find that many visualizations in the notebooks as these are presented as is when i worked, focussing on conducting max no of experiments. You may refer to a small sample project i built to demonstarte the visualization, MLOPS skills. For this and others you can rely on the documentation files to understand my work. "~~~~~~~~
# Synthetic Data Augmentation using CTGAN – Experiments, Results, and Diagnosis

This repository documents an extensive set of experiments to generate synthetic data for the **Telco Customer Churn dataset** using **CTGAN**.  
The work focused particularly on handling **numeric features** such as `MonthlyCharges`, `TotalCharges`, and `tenure`, which repeatedly emerged as problematic.  
Despite applying a wide range of transformations, scaling techniques, segmentation strategies, and evaluation methods, results show that **CTGAN is not able to faithfully reproduce these variables**.

---

## Objectives

1. **Synthetic data generation** with CTGAN.  
2. **Evaluation of realism** using:
   - **SDV QualityReport** (Column Shapes / Pair Trends).  
   - **Discriminator test**: LightGBM classifier distinguishing real from synthetic rows.  
   - **TSTR/TSFR utility tests**: Train-on-Real → Test-on-Synthetic and vice versa.  
   - **Distribution metrics**: KS tests, Wasserstein distances, skewness, kurtosis, normality tests.  
   - **Conditional analyses**: checking distributions within tenure bins, contracts, etc.  
3. **Diagnosis of numeric modeling failures**, with special focus on `MonthlyCharges`.

---

## Data and Initial Checks

- Dataset: `dataset1_openml_7043x21.csv`  
- Cleaning: `TotalCharges` coerced to numeric, NA rows removed.  
- Final dataset: ≈ **7032 rows, 20 features**.  
- EDA: Confirmed categorical levels, numeric ranges, unique values (e.g. `tenure` 0–72).

---

## Experiments Conducted

### 1. Baseline CTGAN with Optuna Tuning
- Multiple trials varying epochs, batch size, embedding dim, generator/discriminator architectures, learning rates.  
- **QualityReport outcomes**:
  - Example: Shapes **95.78%**, Pair Trends **92.71%**, Overall **94.25%**.  
  - Other runs lower: Shapes **91.09%**, Pair Trends **81.52%**.  
- At face value, reports suggested acceptable column-level similarity.

### 2. Real vs Synthetic Discriminator (LightGBM)
- **Critical evaluation step**.  
- A classifier was trained to label rows as real vs synthetic.  
- Results:
  - In one run: **Accuracy, Precision, Recall, F1 = 1.0** (perfect separation).  
  - Another run: Accuracy ≈ **0.91**, ROC AUC ≈ **0.78**.  
- **Top separating features (consistently):**
  - `MonthlyCharges`, `TotalCharges`, `tenure`.  
- This indicates that while SDV reports looked positive, **conditional and joint distributions were not realistic**.

### 3. Distribution Diagnostics
- KS tests and Wasserstein distances revealed large deviations:
  - `MonthlyCharges`: KS **0.0748 (p≈0)**, Wasserstein **2.50**.  
  - `TotalCharges`: KS **0.1129 (p≈0)**, Wasserstein **368**.  
- Descriptives highlighted step-like, tariff-structured distributions for `MonthlyCharges` (18–119 range, with sharp quantization).

### 4. Gaussian Mixture Modeling (GMM)
- Explored multimodality explicitly with `GaussianMixture`.  
- Best-fit component counts:  
  - `tenure`: 73  
  - `MonthlyCharges`: **998**  
  - `TotalCharges`: 14  
- Even within clusters, Shapiro-Wilk tests usually rejected normality.  
- **Conclusion**: `MonthlyCharges` is effectively “hyper-multimodal”, reflecting plan steps and add-ons. CTGAN’s internal variational Gaussian mixture is not capable of capturing this complexity.

### 5. Transformations Toward Normality
- Tested `log1p`, Box-Cox, Yeo-Johnson, QuantileTransformer.  
- Outcomes:
  - Best overall: `MonthlyCharges` → Quantile (normal), `TotalCharges` → Quantile (normal), `tenure` → Yeo-Johnson.  
  - Example post-transform stats:
    - `MonthlyCharges`: skew **0.0054**, kurtosis **3.15**.  
    - `TotalCharges`: skew **−0.0022**, kurtosis **3.14**.  
- While marginals became Gaussian-like, discriminators still detected differences easily → **marginal fixes alone insufficient**.

### 6. Per-Tenure CTGAN
- Trained **72 separate CTGANs**, one per unique tenure value.  
- Each model generated synthetic rows for its tenure subset.  
- This approach severely fragmented the data, led to data scarcity in slices, and recombined synthetic data was still unrealistic.

### 7. Binning and Discretization
- Applied `optbinning.OptimalBinning` on `MonthlyCharges`, `TotalCharges`, and `tenure`.  
- Produced bins with WoE/IV/JS values.  
- Converted features to categorical bins.  
- Useful for churn modeling but **not sufficient when continuous realism is required**.

### 8. Scaling Baselines
- StandardScaler and MinMaxScaler compared.  
- Neither resolved the numeric fidelity issues.

### 9. Train-on-Real, Test-on-Synthetic (TSTR/TSFR)
- Evaluated churn models across domains.  
- Example results: AUC ≈ **0.817**, Accuracy ≈ **0.77**.  
- Mixed outcomes. Less decisive than discriminator tests.

### 10. Post-Processing
- Rounding numeric predictions to 2 decimals, clipping to realistic ranges.  
- Necessary step, but does not solve conditional/joint distribution gaps.

---

## Consolidated Diagnosis

1. **`MonthlyCharges` is hyper-multimodal.**  
   - GMM analysis suggests nearly **1000 underlying modes**.  
   - CTGAN’s mode-specific normalization is not designed for this degree of multimodality.

2. **`TotalCharges` is formula-driven.**  
   - Essentially `TotalCharges ≈ tenure × MonthlyCharges (+ residual)`.  
   - CTGAN does not enforce this arithmetic relationship, producing unrealistic values.

3. **Conditional heterogeneity is lost.**  
   - Charges vary by contract, service, and add-ons.  
   - Without strong conditioning, CTGAN averages across contexts, blurring distributions.

4. **Segmentation is impractical.**  
   - Training per-tenure models (72 splits) fragments data, creating over/underfit issues.

5. **Marginal transformations are inadequate.**  
   - Quantile transforms normalized marginals, but joint and conditional fidelity remained poor.

**Overall conclusion:**  
After extensive testing with scaling, transforms, GMM diagnostics, binning, segmentation, and post-processing, it is very unlikely that CTGAN can model `MonthlyCharges` (and by extension `TotalCharges`) faithfully. The discriminator’s repeated ability to achieve near-perfect separation is the clearest evidence.

---

## Representative Results

- **QualityReport (best)**: Shapes 95.78%, Pair Trends 92.71%, Overall 94.25%.  
- **Discriminator**: Accuracy = 1.0 (perfect), another run Accuracy = 0.91, ROC AUC = 0.78.  
- **Top separating features**: `MonthlyCharges`, `TotalCharges`, `tenure`.  
- **KS/Wasserstein**:
  - `MonthlyCharges`: KS 0.0748, W 2.50.  
  - `TotalCharges`: KS 0.1129, W 368.  
- **GMM best component counts**: tenure 73, `MonthlyCharges` 998, `TotalCharges` 14.  

---

## Future Work – Alternative Models

Given the extensive number of experiments and transformations already tested, it is **very unlikely that further tuning of CTGAN will resolve these issues**. The numeric features, particularly `MonthlyCharges`, expose structural challenges that CTGAN is not designed to address.  

For this reason, I will shift focus to **alternative generative models** that may better capture numeric fidelity:

- **GaussianCopula**: strong at modeling continuous marginals and dependencies.  
- **TVAE (Tabular VAE)**: often superior to GANs for continuous-valued columns.  
- **Diffusion-based models** for tabular data: emerging approaches that handle multimodality more naturally.  
- **Hybrid pipelines** combining different models for categorical vs numeric features.  

The next stage of work will involve testing these alternatives.

---

## Reproduction Guide

Key cells for replication:

1. EDA & cleaning: 0–4, 82  
2. Baseline CTGAN: 9, 11, 13, 16, 25, 53, 59, 64, 84  
3. QualityReport: 17, 26, 60, 65  
4. Discriminator: 21–22, 32–33, 43–44, 50, 62, 67  
5. KS/Wasserstein: 48  
6. Transformations: 35, 37, 58–59, 64, 70  
7. GMM analysis: 71–74, 77–79  
8. Per-tenure CTGAN: 53  
9. OptBinning: 80–83, 94–95, second notebook 1–2  
10. TSTR/TSFR: 75, 89–92  

---

## Final Note

The core difficulty is clear: **`MonthlyCharges` cannot be adequately modeled with CTGAN given its hyper-multimodal, conditional, and formula-constrained nature.**  
Future experiments will therefore move to **alternative generative models** better suited to these distributions.

