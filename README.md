# BSSMOTE OpenML Reproducible Experiments

This repository contains the reproducible code, fold-wise preprocessing, BSSMOTE ablation variants, benchmark oversampling results, metric reports, statistical analyses, deep tabular feature-interval comparisons, logs, selected figures, and hyperparameter configurations.

## Main experiment

- Source: OpenML benchmark datasets.
- Results folder: `results/openml/bssmote_update`
- IR is computed after target construction.
- Existing binary targets are standardized as minority class = 1 and majority class = 0.
- Multiclass benchmark sources are converted using dataset-specific minority-vs-rest rules before cross-validation.
- MinMax scaling and one-hot encoding are fitted only on the training fold.
- Oversampling is applied only to the preprocessed training fold.
- The validation fold remains unchanged.

## Main methods

The main experiment includes:

None, SMOTE, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE, Full BSSMOTE Linear, Full BSSMOTE RBF, BSSMOTE w/o Clustering Linear, BSSMOTE w/o Clustering RBF, BSSMOTE w/o ADASYN Linear, BSSMOTE w/o ADASYN RBF, BSSMOTE w/o SVM

These methods, including `None`, are evaluated across all feature intervals: `Features <= 10`, `10 < Features <= 30`, and `Features > 30`.

## Deep tabular additional comparison

CTGAN and TVAE are run only for the `Features > 30` interval and are saved separately under:

`results/openml/bssmote_update/deep_tabular_highd_only`

Deep tabular baselines are additional comparisons only. They are excluded from Friedman, Holm/Wilcoxon, and critical-difference analyses. Critical-difference figures are generated using `aeon.visualisation.plot_critical_difference`.

## Dynamic analysis outputs

The notebook dynamically generates the following additional files from the current run logs and generated datasets:

- `metrics_report_clean_mean_std.csv`
- `summary_numeric_mean_std.csv`
- `before_after_class_distribution_summary_per_dataset_feature_interval_all_os.csv`
- `bssmote_synthetic_generation_time_feature_interval.csv`
- `synthetic_generation_time_feature_interval_dataset_summary.csv`
- `bssmote_noise_sensitivity_dynamic.csv`
- `bssmote_svm_margin_sensitivity_dynamic.csv`

The following figures are generated at 300 dpi:
- average synthetic generation time per sample by feature interval, with log-scaled y-axis;
- BSSMOTE demonstration figures;
- noise-scale and SVM-margin sensitivity figures;
- critical-difference diagrams;
- before/after class-distribution figures, with log-scaled y-axis.

## BSSMOTE hyperparameter configurations

BSSMOTE configuration files are stored under:

`results/openml/bssmote_update/hyperparameter_configurations/bssmote`

Files:

- `bssmote_hyperparameter_configurations.csv`
- `bssmote_hyperparameter_configurations.json`
- `experiment_protocol.json`

## Data availability

The datasets are publicly available from OpenML using the OpenML IDs saved in:

`results/openml/bssmote_update/dataset_metadata_table1.csv`
### Dataset Summary

The benchmark datasets used in this repository are grouped by feature-count interval and imbalance-ratio (IR) category. The IR values are computed from the actual majority- and minority-class counts used in the experiments.

| Dataset | OpenML ID | Instances | Features | IR | IR Category | Feature Interval | Majority Count | Minority Count |
|---|---:|---:|---:|---:|---|---|---:|---:|
| Blood-transfusion | [1464](https://www.openml.org/d/1464) | 748 | 4 | 3.20 | Low | Features <= 10 | 570 | 178 |
| Haberman | [43](https://www.openml.org/d/43) | 306 | 3 | 2.78 | Low | Features <= 10 | 225 | 81 |
| Mammography | [310](https://www.openml.org/d/310) | 11183 | 6 | 42.01 | Extreme | Features <= 10 | 10923 | 260 |
| Pulsar (HTRU2) | [45558](https://www.openml.org/d/45558) | 17898 | 8 | 9.92 | Medium | Features <= 10 | 16259 | 1639 |
| Wholesale-customers | [1511](https://www.openml.org/d/1511) | 440 | 8 | 2.10 | Low | Features <= 10 | 298 | 142 |
| Wilt | [40983](https://www.openml.org/d/40983) | 4839 | 5 | 17.54 | Medium | Features <= 10 | 4578 | 261 |
| CPU small | [735](https://www.openml.org/d/735) | 8192 | 12 | 2.31 | Low | 10 < Features <= 30 | 5715 | 2477 |
| Diabetes (DIS) | [40713](https://www.openml.org/d/40713) | 3772 | 29 | 64.03 | Extreme | 10 < Features <= 30 | 3714 | 58 |
| Elevators | [846](https://www.openml.org/d/846) | 16599 | 18 | 2.24 | Low | 10 < Features <= 30 | 11469 | 5130 |
| Vehicle | [994](https://www.openml.org/d/994) | 846 | 18 | 2.88 | Low | 10 < Features <= 30 | 628 | 218 |
| Mfeat-fourier | [971](https://www.openml.org/d/971) | 2000 | 76 | 9.00 | Medium | Features > 30 | 1800 | 200 |
| Mfeat-karhunen | [1020](https://www.openml.org/d/1020) | 2000 | 64 | 9.00 | Medium | Features > 30 | 1800 | 200 |
| Optdigits | [980](https://www.openml.org/d/980) | 5620 | 64 | 8.83 | Medium | Features > 30 | 5048 | 572 |
| Satellite | [40900](https://www.openml.org/d/40900) | 5100 | 36 | 67.00 | Extreme | Features > 30 | 5025 | 75 |

### Notes

- All datasets are publicly available from OpenML.
- OpenML IDs in the table link to the original dataset pages.
- IR denotes the imbalance ratio between the majority and minority classes.
- Majority Count and Minority Count are the actual class counts used to compute IR.
- Feature Interval is defined using the number of input features.
- These datasets are used for evaluating BSSMOTE, its ablation variants, traditional oversampling baselines, and deep tabular oversampling methods under a fold-wise preprocessing protocol.

## Data Preprocessing Protocol

The experiment uses a leakage-safe, fold-wise preprocessing pipeline. All preprocessing operations are fitted only on the training fold and then applied to the corresponding validation fold.

### 1. Dataset loading

- Benchmark datasets are loaded from OpenML using the predefined OpenML dataset IDs.
- Each dataset is converted into a feature matrix `X` and target vector `y`.
- Feature column names are converted to strings and stripped of extra spaces.
- Dataset metadata are recorded, including:
  - dataset name;
  - OpenML ID;
  - number of instances;
  - number of original features;
  - imbalance ratio;
  - imbalance-ratio category;
  - feature-count interval.

### 2. Target construction and label standardization

- All datasets are converted to binary classification tasks.
- If the original target is already binary:
  - the minority class is encoded as `1`;
  - the majority class is encoded as `0`.
- If the dataset is multiclass:
  - a predefined minority-vs-rest rule is used where specified;
  - otherwise, the rarest class is treated as the minority class and all remaining classes are grouped as the majority class.
- For numeric regression-style targets in selected datasets:
  - the mean target value is used as the threshold;
  - values below or equal to the mean and values above the mean are converted into two classes;
  - the final minority class is encoded as `1` and the majority class as `0`.
- The imbalance ratio is computed after final binary target construction.

### 3. Feature-count and imbalance grouping

- Datasets are grouped by feature-count interval:
  - `Features <= 10`;
  - `10 < Features <= 30`;
  - `Features > 30`.
- Imbalance-ratio categories are assigned as:
  - Low: `IR <= 5`;
  - Medium: `5 < IR <= 20`;
  - Extreme: `IR > 20`.

### 4. Cross-validation split

- Stratified 5-fold cross-validation is used.
- Stratification preserves the class distribution in each fold.
- The random seed is fixed at `42` for reproducibility.
- For each fold:
  - the training fold is used for fitting preprocessing and oversampling;
  - the validation fold is only transformed and evaluated;
  - no validation-fold information is used during preprocessing or oversampling.

### 5. Dynamic feature-type detection

- Feature types are detected separately within each training fold.
- A feature is treated as numerical if:
  - it already has a numeric dtype; or
  - at least 98% of its values can be converted to numeric.
- Remaining features are treated as categorical.

### 6. Missing-value handling

- Numerical missing values are imputed using the median value from the training fold only.
- If the training-fold median is unavailable, the fallback value `0.0` is used.
- Categorical missing values are replaced with the string `"missing"`.
- The same training-fold imputation values are applied to the validation fold.

### 7. Numerical feature scaling

- Numerical features are scaled using MinMax scaling.
- The scaler is fitted only on the training fold.
- The learned minimum and maximum values from the training fold are used to transform the validation fold.
- The transformed numerical values are scaled to the range `[0, 1]`.

### 8. Categorical feature encoding

- Categorical features are encoded using one-hot encoding.
- The encoder is fitted only on the training fold.
- Unknown categories in the validation fold are ignored using `handle_unknown="ignore"`.
- This prevents validation-fold categories from influencing the training process.

### 9. Column transformation

- Numerical and categorical transformations are combined using a `ColumnTransformer`.
- Only selected numerical and categorical columns are retained.
- Transformed feature names are stored for reproducibility.
- The transformed feature matrix is converted to dense `float64` format.

### 10. Oversampling protocol

- Oversampling is applied only after fold-wise preprocessing.
- Oversamplers are fitted only on the preprocessed training fold.
- The validation fold is never oversampled.
- The no-oversampling baseline keeps the original preprocessed training fold unchanged.
- For standard oversamplers, the neighborhood size is adjusted dynamically based on the number of minority samples.
- If a sampler fails or there are too few minority samples, the original training fold is retained and the failure reason is logged.

### 11. Deep tabular baseline preprocessing

- CTGAN and TVAE are used only for the `Features > 30` interval.
- They are treated as additional comparisons and are excluded from Friedman, Holm/Wilcoxon, and critical-difference analyses.
- CTGAN and TVAE are trained only on minority-class samples from the training fold.
- Synthetic samples are generated only for the minority class.
- Generated values are converted to numeric form, missing generated values are replaced using minority-fold medians, and final synthetic values are clipped to `[0, 1]`.
- The validation fold remains unchanged.

### 12. Reproducibility and leakage prevention

- All random seeds are fixed using `SEED = 42`.
- Preprocessing objects are never fitted on validation data.
- Oversampling is never applied before cross-validation.
- Oversampling is never applied to validation folds.
- Fold-wise preprocessing logs are saved for audit and reproducibility.

