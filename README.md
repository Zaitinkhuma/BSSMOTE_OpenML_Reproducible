# BSSMOTE OpenML Reproducible Experiments

This repository contains the reproducible code, fold-wise preprocessing, BSSMOTE ablation variants, benchmark oversampling results, clean metric reports, statistical analyses, deep tabular feature-interval comparisons, logs, selected figures, and hyperparameter configurations.

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

Only the requested figures are generated at 300 dpi:
- average synthetic generation time per sample by feature interval, with log-scaled y-axis;
- BSSMOTE demonstration figures;
- noise-scale and SVM-margin sensitivity figures;
- critical-difference diagrams;
- before/after class-distribution figures, with log-scaled y-axis.

Metric boxplots, mean metric bar plots, deep metric comparison plots, and synthetic-count plots are removed/disabled.

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

| Dataset | OpenML ID | Instances | Features | IR | IR Category | Feature Interval | Majority Count | Minority Count |
|---|---:|---:|---:|---:|---|---|---:|---:|
| Blood-transfusion | [1464](https://www.openml.org/d/1464) | 748 | 4 | 3.2022 | Low | Features <= 10 | 570 | 178 |
| Haberman | [43](https://www.openml.org/d/43) | 306 | 3 | 2.7778 | Low | Features <= 10 | 225 | 81 |
| Mammography | [310](https://www.openml.org/d/310) | 11,183 | 6 | 42.0115 | Extreme | Features <= 10 | 10,923 | 260 |
| Pulsar (HTRU2) | [45558](https://www.openml.org/d/45558) | 17,898 | 8 | 9.9201 | Medium | Features <= 10 | 16,259 | 1,639 |
| Wholesale-customers | [1511](https://www.openml.org/d/1511) | 440 | 8 | 2.0986 | Low | Features <= 10 | 298 | 142 |
| Wilt | [40983](https://www.openml.org/d/40983) | 4,839 | 5 | 17.5402 | Medium | Features <= 10 | 4,578 | 261 |
| CPU small | [735](https://www.openml.org/d/735) | 8,192 | 12 | 2.3072 | Low | 10 < Features <= 30 | 5,715 | 2,477 |
| Diabetes (DIS) | [40713](https://www.openml.org/d/40713) | 3,772 | 29 | 64.0345 | Extreme | 10 < Features <= 30 | 3,714 | 58 |
| Elevators | [846](https://www.openml.org/d/846) | 16,599 | 18 | 2.2357 | Low | 10 < Features <= 30 | 11,469 | 5,130 |
| Vehicle | [994](https://www.openml.org/d/994) | 846 | 18 | 2.8807 | Low | 10 < Features <= 30 | 628 | 218 |
| Mfeat-fourier | [971](https://www.openml.org/d/971) | 2,000 | 76 | 9.0000 | Medium | Features > 30 | 1,800 | 200 |
| Mfeat-karhunen | [1020](https://www.openml.org/d/1020) | 2,000 | 64 | 9.0000 | Medium | Features > 30 | 1,800 | 200 |
| Optdigits | [980](https://www.openml.org/d/980) | 5,620 | 64 | 8.8252 | Medium | Features > 30 | 5,048 | 572 |
| Satellite | [40900](https://www.openml.org/d/40900) | 5,100 | 36 | 67.0000 | Extreme | Features > 30 | 5,025 | 75 |

### Dataset Grouping Criteria

The datasets are grouped using the number of input features:

| Feature Interval | Definition |
|---|---|
| Features <= 10 | Datasets with 10 or fewer input features |
| 10 < Features <= 30 | Datasets with more than 10 and up to 30 input features |
| Features > 30 | Datasets with more than 30 input features |

The imbalance-ratio categories are defined as follows:

| IR Category | Definition |
|---|---|
| Low | IR <= 5:1 |
| Medium | 5:1 < IR <= 20:1 |
| Extreme | IR > 20:1 |

### Notes

- All datasets are publicly available from OpenML.
- OpenML IDs in the table are clickable links to the original dataset sources.
- IR denotes the imbalance ratio between the majority and minority classes.
- Majority Count and Minority Count refer to the class distribution used for computing IR.
- These datasets are used for evaluating BSSMOTE, its ablation variants, traditional oversampling baselines, and deep tabular oversampling methods under a fold-wise preprocessing protocol.

## Code availability

The source code and experimental resources are intended for repository release at:

https://github.com/Zaitinkhuma/BSSMOTE_OpenML_Reproducible
