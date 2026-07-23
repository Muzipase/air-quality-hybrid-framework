# THE COPPERBELT UNIVERSITY
## SCHOOL OF INFORMATION AND COMMUNICATION TECHNOLOGY
### COMPUTER SCIENCE DEPARTMENT

**CS 400: PROJECT SEMINARS**

---

# A Hybrid SMOTE-Tomek and Bayesian-Optimized SVM Framework for Interpretable Air Quality Classification in Urban Zambia

## (System Design Specification)

**Submitted By:**
Muzipase Tembo (Student ID: 22107085)

**Supervised By:**
Dr. George Mufungulwa

**23rd July, 2026**

---

## Contents

| Section | Title | Page |
|---------|-------|------|
| 1 | Introduction | 1 |
| 1.1 | Purpose | 1 |
| 1.2 | Scope | 1 |
| 1.3 | Intended Audience | 2 |
| 2 | System Architecture | 2 |
| 2.1 | Architectural Style Selection | 2 |
| 2.2 | High-Level Architect Diagram | 2 |
| 2.3 | Component Breakdown | 3 |
| 3 | Detailed Design | 5 |
| 3.1 | Sequence Diagram: End-to-End Pipeline Flow | 5 |
| 3.2 | Data Flow | 5 |
| 4 | Data Design | 6 |
| 4.1 | Data Model Rationale | 6 |
| 4.2 | Feature Schema | 7 |
| 4.3 | Air Quality Classification Thresholds | 8 |
| 4.4 | Entity-Relationship Overview | 8 |
| 5 | SMOTE-Tomek Component Design | 9 |
| 5.1 | Design Rationale | 9 |
| 5.2 | Implementation Specification | 9 |
| 5.3 | Class Imbalance Analysis Output | 10 |
| 6 | Model Design | 10 |
| 6.1 | SVM Model Specification | 10 |
| 6.2 | Bayesian Optimization Design | 11 |
| 6.3 | SHAP Interpretability Design | 11 |
| 7 | Evaluation Design | 12 |
| 7.1 | Train/Test Split Strategy | 12 |
| 7.2 | Performance Metrics | 12 |
| 7.3 | Comparative Evaluation Framework | 13 |
| 8 | Technology Stack | 13 |
| 9 | Non-Functional Design Considerations | 14 |
| 9.1 | Performance and Computational Efficiency | 14 |
| 9.2 | Reproducibility | 14 |
| 9.3 | Security and Privacy | 15 |
| 9.4 | Error Handling | 15 |
| 10 | References | 16 |

**List of Tables**

| Table | Title | Page |
|-------|-------|------|
| 1 | Processed Feature Schema | 7 |
| 2 | AQI Classification Thresholds (WHO/EPA) | 8 |
| 3 | SVM Model Configuration | 10 |
| 4 | Bayesian Optimization Configuration | 11 |
| 5 | Evaluation Metrics | 12 |
| 6 | Technology Stack and Justification | 13 |

---

## 1. Introduction

This System Design Specification (SDS) provides a detailed technical blueprint for the Hybrid SMOTE-Tomek and Bayesian-Optimized SVM Air Quality Classification system. It translates the research objectives and requirements established in the project proposal into a concrete design, defining the system's architecture, data pipeline, model components, and evaluation strategy. This document serves as the primary technical guide for implementation and the basis for verifying that the design meets the stated project objectives.

### 1.1 Purpose

The purpose of this document is to define how the system will be constructed. It provides sufficient detail to guide the implementation of the data pipeline, imbalance correction, model training, hyperparameter optimization, interpretability analysis, and evaluation components of the project.

### 1.2 Scope

This specification covers the internal structure of the classification system, including:

- The overall software and data pipeline architecture.
- Detailed descriptions of each processing component and their interactions.
- The data schema and preprocessing design.
- The SMOTE-Tomek imbalance correction implementation.
- The Bayesian-Optimized SVM model design and training strategy.
- The SHAP-based interpretability component.
- The evaluation framework and metrics.
- The technology stack and implementation tools.

This document does not cover project management aspects, physical sensor deployment, IoT integration, mobile application development, or post-submission maintenance.

### 1.3 Intended Audience

- **Software/ML Developers:** Will use this document as the primary guide for implementing the pipeline.
- **Academic Supervisors:** Can review architectural decisions and component interactions.
- **QA/Testing Team:** Will use this document to develop test cases and evaluation strategies.
- **Project Managers:** Will use this document to understand the technical scope and effort required.

---

## 2. System Architecture

### 2.1 Architectural Style Selection

A **Sequential Pipeline Architecture** has been selected for this system. In this model, all processing stages — data ingestion, preprocessing, imbalance correction, model training, optimization, interpretability, and evaluation — are organized as a linear, modular sequence within a single Python execution environment.

**Rationale:**

- **Modularity:** Each stage is implemented as an independent, interchangeable module, enabling iterative refinement of individual components without disrupting the entire pipeline.
- **Reproducibility:** A sequential pipeline with explicit state capture at each stage ensures that all results are fully reproducible, which is a core requirement for academic research.
- **Transparency:** The linear flow makes it straightforward to inspect intermediate outputs (e.g., class distributions before and after SMOTE-Tomek, hyperparameter search history, SHAP values) for debugging and analysis.
- **Compatibility:** All selected tools (scikit-learn, imbalanced-learn, Optuna, SHAP) are designed to operate within this sequential, scikit-learn-compatible pipeline pattern.

### 2.2 High-Level Architect Diagram

The system is organized into five logical layers, each with distinct responsibilities:

*[Insert Figure 1: High-Level Architect Diagram here]*

### 2.3 Component Breakdown

- **Data Ingestion Component (Open-Meteo API + OpenAQ API + pandas):** Interfaces with the Open-Meteo Air Quality API and the OpenAQ REST API to retrieve pollutant measurements (PM2.5, PM10, NO₂, SO₂, CO, O₃) and meteorological data (temperature, humidity, wind speed) for Lusaka, Kitwe, and Ndola. Both APIs are accessed via HTTP with a 30-second timeout and 3-retry exponential backoff strategy. A State Check validates the structure and completeness of the API response — verifying the `hourly` key exists and the timestamp array is non-empty — before the data is handed off to the Preprocessing Component. When the Open-Meteo API is unavailable, the system falls back to the OpenAQ API. When both APIs fail, a locally cached CSV dataset is used as the final fallback.

- **Preprocessing Component (pandas + scikit-learn):** Handles column normalisation, duplicate removal, median-based missing value imputation, z-score standardisation, and AQI labelling. A Variance Inflation Factor (VIF) multicollinearity analysis step is also applied to the pollutant features to identify highly correlated pairs (e.g., PM2.5 and PM10) and flag features that may degrade SVM performance under strong multicollinearity.

- **Imbalance Analysis Component (pandas):** Computes and stores the class distribution across the five air quality categories (Good, Moderate, Unhealthy, Very Unhealthy, Hazardous), producing empirical evidence for SMOTE-Tomek configuration. Logs class frequency counts, percentages, and the majority-to-minority imbalance ratio.

- **SMOTE-Tomek Component (imbalanced-learn):** Applies SMOTE oversampling followed by Tomek Links undersampling exclusively on the training subset, producing a balanced training dataset for SVM input. A hard cap (`max_synthetic_ratio = 3.0`) limits the maximum size of minority classes relative to the majority class, preventing the dataset from growing so large that Bayesian optimization becomes computationally intractable.

- **SVM Model Component (scikit-learn):** Implements the RBF-kernel and polynomial-kernel SVM classifier via `SVC`. Includes both the hybrid (SMOTE-Tomek + Bayesian-Optimized) variant and a standard baseline variant for comparison.

- **Bayesian Optimization Component (Optuna):** Systematically tunes the SVM hyperparameters (kernel, C, gamma, degree) using Optuna's Tree-structured Parzen Estimator (TPE) surrogate model. At least 10 random startup trials are evaluated before the TPE surrogate model begins guiding the search, reducing the risk of the optimizer converging on a local minimum.

- **SHAP Interpretability Component (SHAP library):** Computes SHapley Additive exPlanations for the trained SVM model using the model-agnostic KernelExplainer. Background data is summarised via k-means clustering (k=50) for computational efficiency. A beeswarm plot is saved as a high-resolution PNG (200 DPI) for stakeholders without a Python environment.

- **Evaluation Component (scikit-learn metrics):** Computes accuracy, weighted precision, weighted recall, weighted F1-score, and per-class classification reports for the optimised model, with specific attention to minority class recall.

---

## 3. Detailed Design

### 3.1 Sequence Diagram: End-to-End Pipeline Flow

The following sequence describes the interaction between components during a full pipeline execution:

*[Insert Figure 2: Sequence Diagram here]*

### 3.2 Data Flow

The Data Flow Diagram below shows the movement of data through the system at the top level:

*[Insert Figure 3: Data Flow Diagram here]*

---

## 4. Data Design

### 4.1 Data Model Rationale

The system operates on tabular, structured data stored in-memory as pandas DataFrames throughout execution. There is no relational database, as the application is a batch analytics pipeline rather than a transactional system. Intermediate states (cleaned data, balanced training data, model artifacts) are persisted to disk as CSV files and serialized Python objects (pickle/joblib) to support reproducibility and iterative development.

The primary data source is the **Open-Meteo Air Quality API** (`air-quality-api.open-meteo.com`), which provides hourly pollutant forecasts from the CAMS European and Global atmospheric composition models. Meteorological data (temperature, humidity, wind speed) is sourced from the **Open-Meteo Weather API** (`api.open-meteo.com`). When the Open-Meteo API is unavailable, the system falls back to the **OpenAQ REST API** (`api.openaq.org`). Historical data for Lusaka, Kitwe, and Ndola is retrieved programmatically and stored locally as raw CSV files to avoid repeated API calls during development. Each record represents a single hourly measurement at a given location and timestamp.

### 4.2 Feature Schema

After preprocessing, each data instance contains the following features:

**Table 1: Processed Feature Schema:**

| Feature | Type | Description |
|---------|------|-------------|
| pm25 | float | PM2.5 particulate matter concentration (μg/m³), z-score normalised |
| pm10 | float | PM10 particulate matter concentration (μg/m³), z-score normalised |
| no2 | float | Nitrogen Dioxide concentration (μg/m³), z-score normalised |
| so2 | float | Sulphur Dioxide concentration (μg/m³), z-score normalised |
| co | float | Carbon Monoxide concentration (μg/m³), z-score normalised |
| o3 | float | Ozone concentration (μg/m³), z-score normalised |
| temperature | float | Ambient temperature (°C), z-score normalised |
| humidity | float | Relative humidity (%), z-score normalised |
| wind_speed | float | Wind speed (km/h), z-score normalised |
| pm25_pm10_ratio | float | Ratio of PM2.5 to PM10, capturing fine-to-coarse particulate fraction |
| pollution_load | float | Sum of PM2.5 and O₃, representing combined pollution burden |
| is_high_pm25 | bool | Binary flag: 1 if PM2.5 > 35.4 μg/m³ (Unhealthy threshold), 0 otherwise |
| is_high_pm10 | bool | Binary flag: 1 if PM10 > 150.0 μg/m³, 0 otherwise |

Meteorological variables (temperature, humidity, wind_speed) are included to improve the predictive performance of the air quality classification model by capturing atmospheric conditions that influence pollutant dispersion and concentration. Unlike pollutant data, these features are not available through the OpenAQ API. Instead, they are sourced from the **Open-Meteo Weather API**, which provides globally consistent weather forecast data based on geographic coordinates. The weather data is spatially matched to the three study locations using their latitude and longitude values, and temporally aligned with the corresponding pollutant measurement timestamps. This ensures consistency between environmental conditions and air quality observations.

### 4.3 Air Quality Classification Thresholds

Labels are assigned based on WHO and US EPA Air Quality Index (AQI) thresholds. The PM2.5 AQI breakpoints are used as the primary labeling criterion. The system uses a 5-class scheme (excluding the "Unhealthy for Sensitive Groups" category for simplicity):

**Table 2: AQI Classification Thresholds (WHO/EPA):**

| Class Label | PM2.5 Range (μg/m³) | Category | Health Implication |
|-------------|---------------------|----------|-------------------|
| 0 | 0 – 12.0 | Good | Air quality is satisfactory; little or no risk |
| 1 | 12.1 – 35.4 | Moderate | Acceptable; sensitive individuals may experience minor effects |
| 2 | 35.5 – 55.4 | Unhealthy | General public may experience health effects |
| 3 | 55.5 – 150.4 | Very Unhealthy | Health alert; serious effects for all |
| 4 | 150.5+ | Hazardous | Emergency conditions; entire population at risk |

When PM2.5 data is unavailable, the system falls back to PM10 divided by 2.0 as an approximation before applying the same breakpoints.

### 4.4 Entity-Relationship Overview

As a pipeline system, the conceptual data relationships are:

- **One City has many Measurements (1:N).** Each measurement is linked to one city (Lusaka, Kitwe, or Ndola).
- **One Measurement has one Label (1:1).** Each processed measurement record is assigned a single AQI class label.
- **One Measurement may generate many SHAP Values (1:N).** Each feature of a measurement contributes one SHAP value per prediction.

*[Insert Figure 4: Entity-Relationship Diagram here]*

---

## 5. SMOTE-Tomek Component Design

### 5.1 Design Rationale

SMOTE-Tomek is a hybrid resampling technique that combines SMOTE oversampling of minority classes with Tomek Links undersampling to clean borderline majority-class instances. This two-phase approach is selected over SMOTE alone because Tomek Links removal produces a cleaner decision boundary, reducing noise near class boundaries in the SVM feature space — a known cause of misclassification in imbalanced settings.

### 5.2 Implementation Specification

- **Library:** imbalanced-learn (`SMOTETomek` class).
- **SMOTE k_neighbors:** Dynamically computed as `max(1, min(5, min_class_count - 1))`, where `min_class_count` is the smallest class in the training set. This ensures that classes with fewer than 6 samples are handled without errors. The maximum value is capped at 5 (the default).
- **Synthetic Sample Cap:** A hard limit is enforced via `max_synthetic_ratio = 3.0`. The target count for any minority class is `min(int(majority_count × 3.0), majority_count)`, ensuring no class exceeds the majority class size and the total dataset remains computationally tractable for subsequent Bayesian optimization.
- **Tomek Links:** Pairs of samples from opposing classes that are each other's nearest neighbours. The majority-class sample from each pair is removed.
- **Application Scope:** Applied exclusively to `X_train` and `y_train`. `X_test` and `y_test` are never resampled to ensure that evaluation reflects the true real-world class distribution.
- **Output:** `X_balanced` (n_balanced × 13 features), `y_balanced` (n_balanced labels) with balanced representation across all AQI classes.
- **Random Seeds:** `random_state=42` for both SMOTE and SMOTETomek for reproducibility.

### 5.3 Class Imbalance Analysis Output

Prior to SMOTE-Tomek, a class distribution analysis is computed and logged. Metrics include:

- Class frequency counts and percentages for each of the 5 AQI categories.
- Imbalance ratio: ratio of the majority class count to the minority class count.
- Post-SMOTE-Tomek distribution comparison to confirm balance correction.
- If all classes already meet or exceed the target count, SMOTE oversampling is skipped (no-op).

---

## 6. Model Design

### 6.1 SVM Model Specification

The core classifier is an SVM implemented using scikit-learn's `SVC` class. Two variants are trained:

**Table 3: SVM Model Configuration:**

| Property | Hybrid Model (Primary) | Baseline Model (Comparison) |
|----------|----------------------|---------------------------|
| Training Data | SMOTE-Tomek balanced X_train | Original imbalanced X_train |
| Kernel | RBF or Polynomial (optimised) | RBF (fixed) |
| C (Regularization) | Bayesian Optimised (0.1 – 10.0, log-uniform) | 1.0 (default) |
| gamma (Kernel width) | Optimised: "scale" or "auto" | "scale" (default) |
| degree (Poly only) | Optimised (2–4) | N/A |
| Decision Function | One-vs-One (OvO) multi-class | One-vs-One (OvO) multi-class |
| probability | Disabled (False) | Disabled (False) |

Both models use `random_state=42` for reproducibility.

### 6.2 Bayesian Optimization Design

Bayesian Optimization is implemented using **Optuna** with its **Tree-structured Parzen Estimator (TPE)** surrogate model. TPE is a sequential model-based optimisation algorithm that models the objective function using kernel density estimation, replacing grid search with a principled probabilistic approach.

At least **10 random startup trials** (`n_startup_trials=10`) are evaluated before the TPE surrogate model begins guiding the search, reducing the risk of the optimizer converging on a local minimum.

**Table 4: Bayesian Optimization Configuration:**

| Parameter | Search Space | Description |
|-----------|-------------|-------------|
| kernel | Categorical: ["rbf", "poly"] | Kernel function type |
| C (Regularization) | Float: 0.1 – 10.0 (log-uniform) | Controls the trade-off between margin width and training error |
| gamma (Kernel width) | Categorical: ["scale", "auto"] | Controls the influence radius of each support vector |
| degree | Integer: 2 – 4 | Polynomial kernel degree (only when kernel="poly", else 3) |
| n_trials | 20 | Number of Bayesian search evaluations |
| cv | 3-fold (dynamic: clamped to smallest viable fold count) | Stratified cross-validation within each search iteration |
| scoring | recall_macro | Macro-averaged recall — emphasises minority class performance |
| pruner | MedianPruner (n_startup_trials=5) | Prunes underperforming trials early |
| direction | maximize | Maximise the recall_macro score |

### 6.3 SHAP Interpretability Design

SHAP (SHapley Additive exPlanations) values are computed for the trained hybrid SVM model using the SHAP library's `KernelExplainer`, which is model-agnostic and compatible with SVM.

- **Explainer:** `shap.KernelExplainer(model.predict_proba, background_data)` where `background_data` is a k-means summary of `X_train` with `k=50` clusters for computational efficiency.
- **SHAP values** are computed for instances in `X_test`.
- **Summary bar chart:** Global feature importance showing mean absolute SHAP values per feature across all classes.
- **Beeswarm plot:** Distribution of SHAP values per feature showing directionality of feature effects. Saved as a high-resolution PNG (200 DPI, 10×6 inches) to `artifacts/shap_plots/shap_beeswarm.png`. For the final report, a beeswarm plot specifically compares which pollutants (e.g., SO₂) contribute most strongly to "Hazardous" predictions in the Copperbelt cities (Kitwe, Ndola) versus Lusaka.
- **Persistence Strategy:** The beeswarm plot is saved as a static high-resolution PNG file so that stakeholders without a Python environment can view feature-importance results directly.

---

## 7. Evaluation Design

### 7.1 Train/Test Split Strategy

- **Split ratio:** 80% training / 20% testing.
- **Method:** Stratified random split (`train_test_split` with `stratify=y`) to preserve class distribution in both subsets. Stratification is disabled when a class has fewer than 2 samples or the total dataset has fewer than 4 records.
- **Random seed:** Fixed (`random_state=42`) for reproducibility.
- **SMOTE-Tomek** is applied only after the split, on `X_train` and `y_train` exclusively.

### 7.2 Performance Metrics

Although multiple evaluation metrics are computed, accuracy is not considered a reliable primary performance measure for this system due to the highly imbalanced nature of the dataset. In imbalanced classification problems, accuracy can be misleading because a model may achieve high accuracy by favouring majority classes while performing poorly on critical minority classes such as "Very Unhealthy" and "Hazardous". Therefore, the primary evaluation focus of this system is the **weighted F1-score**, which provides a balanced measure of precision and recall across all classes. In addition, recall for minority classes is closely monitored to ensure the system effectively identifies high-risk air pollution events that are most important for public health and decision-making.

**Table 5: Evaluation Metrics:**

| Metric | Scope | Primary Purpose |
|--------|-------|----------------|
| Accuracy | Overall | Proportion of correctly classified instances |
| Precision (weighted) | Per-class + weighted average | Fraction of predicted positives that are true positives |
| Recall (weighted) | Per-class + weighted average | Fraction of actual positives correctly identified |
| F1-Score (weighted) | Per-class + weighted average | Harmonic mean of precision and recall — primary comparison metric |
| Classification Report | All classes | Per-class precision, recall, F1-score, and support counts |

### 7.3 Comparative Evaluation Framework

Both the hybrid model and the baseline model are evaluated on the same original (imbalanced) `X_test`/`y_test`. Results are reported side-by-side to quantify the improvement achieved by SMOTE-Tomek and Bayesian Optimization. The primary measure of contribution is the improvement in per-class recall for the "Very Unhealthy" (class 3) and "Hazardous" (class 4) categories.

---

## 8. Technology Stack

Technology choices are based on open-source availability, compatibility, performance, and alignment with academic reproducibility requirements.

**Table 6: Technology Stack and Justification:**

| Component | Technology | Justification |
|-----------|-----------|---------------|
| Programming Language | Python 3.13 | Industry standard for ML/data science; extensive ecosystem; free and open-source |
| Data Manipulation | pandas | Efficient DataFrame operations for tabular data; seamless API integration |
| ML Framework | scikit-learn | Standardised SVM implementation (SVC); pipeline integration; stratified splitting; evaluation metrics |
| Imbalance Correction | imbalanced-learn | Official SMOTETomek implementation; scikit-learn compatible; well-validated |
| Bayesian Optimization | Optuna | TPE surrogate model for hyperparameter search; lightweight; fast convergence |
| Interpretability | SHAP | Model-agnostic KernelExplainer for SVM; production-grade visualizations; widely cited |
| HTTP Client | requests | HTTP client for Open-Meteo and OpenAQ REST APIs; JSON response parsing to DataFrame |
| Numerical Computing | NumPy | Array operations underlying all ML computations |
| Visualization | matplotlib | SHAP beeswarm plot generation; static PNG export at 200 DPI |
| Serialization | joblib | Efficient model and pipeline persistence (save/load trained SVM and scaler) |
| Backend API | FastAPI + Uvicorn | REST API for serving predictions, forecasts, and SHAP explanations |
| Frontend | Next.js 14 (React) | Dashboard for real-time air quality visualisation, forecasts, and SHAP analysis |
| Version Control | Git + GitHub | Source control and reproducibility; Git-LFS for large file management |
| Deployment | Vercel (frontend) + Render (backend) | Free-tier hosting for academic project; auto-deploy from GitHub |

---

## 9. Non-Functional Design Considerations

### 9.1 Performance and Computational Efficiency

- SVM training complexity is O(n²) to O(n³) in the number of training samples. To manage this, SMOTE-Tomek is configured with a `max_synthetic_ratio` of 3.0, capping minority class size at no more than the majority class, preventing the balanced training set from growing so large that subsequent Bayesian Optimisation iterations become computationally intractable.
- KernelExplainer for SHAP is computationally expensive on large test sets. The background data is summarised to 50 k-means clusters (rather than using all training samples) to maintain tractability. The summary computation uses 25 k-means clusters.
- Bayesian Optimisation is limited to 20 trials with 3-fold CV, balancing search thoroughness against compute time.

### 9.2 Reproducibility

- All random seeds are fixed (`random_state=42`) across train/test splitting, SMOTE-Tomek, Bayesian Optimisation (Optuna sampler seed), and SHAP background sampling to ensure fully reproducible results.
- Raw data is stored locally as CSV after initial API retrieval. All preprocessing steps are logged and documented.
- Trained models are serialized with joblib and version-stamped for exact replication.
- Git Large File Storage (Git-LFS) is configured to manage large local CSV datasets, serialized model artifacts (`.pkl`), and generated plot images (`.png`), preventing the GitHub repository from becoming bloated.

### 9.3 Security and Privacy

- The system uses only publicly available, non-proprietary environmental monitoring data from Open-Meteo and OpenAQ. No personally identifiable information (PII) is collected or processed.
- API access is performed using public endpoints requiring no authentication token. All data is accessed in compliance with Open-Meteo and OpenAQ platform terms of use.
- All data is stored locally on the researcher's personal machine within a project-specific directory. No data is transmitted to third-party services beyond the initial API retrieval.
- The web-based deployment (Vercel + Render) serves only aggregated air quality data and model predictions; no raw datasets are exposed through the public API.

### 9.4 Error Handling

- **API Failure:** If the Open-Meteo API is unreachable or returns an error, the pipeline retries 3 times with exponential backoff (2ⁿ seconds). If all retries fail, it falls back to the OpenAQ API. If OpenAQ also fails, the system falls back to a locally cached CSV dataset. An error log entry is generated at each failure stage.
- **Missing Data:** Rows where all six pollutant columns are NaN are dropped. Remaining missing values are filled using median imputation per column. Columns where all values are NaN (e.g., weather data unavailable) are filled with 0.0.
- **SMOTE Failure:** If a minority class has fewer than `k_neighbors` samples, the `k_neighbors` parameter is automatically reduced to `n_samples - 1` for that class. If all classes already meet the target size, SMOTE oversampling is skipped entirely.
- **Convergence:** SVM training uses `max_iter=5000` to allow sufficient convergence time. A convergence warning triggers a logged warning.
- **Data Validation:** A `validate_dataframe()` function checks for required columns (pm25, pm10, no2, so2, co, o3), numeric validity, and non-empty data before any preprocessing occurs. Validation failures raise `ValueError` with specific error details.

---

## 10. References

Brauer, M., Freedman, G., Frostad, J., Cohen, A., & Burnett, R. (2020). Global estimates of ambient fine particulate matter concentrations and health impacts. *The Lancet Planetary Health*, 4(2), e98-e108.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2623-2631.

Kumar, P., Morawska, L., Johnson, W. B., Heald, G., & Ball, R. S. (2020). Air quality monitoring and modelling in developing countries: Challenges and opportunities. *Environment International*, 142, 105832.

Lemaitre, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, 18(17), 1-5.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

Open-Meteo. (2024). Open-Meteo Air Quality API and Weather API documentation. https://open-meteo.com/en/docs

OpenAQ. (2024). OpenAQ platform and REST API documentation. https://docs.openaq.org/

Patel, R., Iyer, S., Nair, M., & Kulkarni, A. (2022). A systematic review of machine learning models for air pollution prediction. *Journal of Cleaner Production*, 357, 131914.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

Rahman, M. M., et al. (2024). Real-time AQI monitoring platform using machine learning. *Environmental Monitoring and Assessment*.

Tirink, S. (2025). Performance comparison of XGBoost, LightGBM, and SVM for AQI forecasting. *Expert Systems with Applications*, 235, 121098.

Wang, J., Liu, P., Zhang, M., & Chen, K. (2020). Air quality classification using support vector machines. *Atmospheric Research*, 242, 104999.

World Health Organization. (2021). WHO global air quality guidelines. https://www.who.int/publications/i/item/9789240034228

Zambia Environmental Management Agency. (2022). State of air quality in Zambia report. ZEMA.
