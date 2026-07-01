# Machine Learning Pipeline for Binary Endpoint Prediction

This repository contains a three-step R workflow for building and evaluating binary classification models from molecular descriptor or fingerprint data. The pipeline creates repeated train/test splits, performs feature selection on each training set, trains multiple machine learning models using the selected features, and aggregates model performance across repeats.

## Pipeline scripts

| Order | Script | Purpose |
|---|---|---|
| Step 1 | `split_training_test.R` | Create 20 stratified train/test splits for each input CSV. |
| Step 2 | `feature_selection_on_training_data.R` | Run feature selection on each training set using Fisher exact test, single-feature AUC, XGBoost importance, and Random Forest importance. |
| Step 3 | `machine_learning_models.R` | Train models on each selected training set, predict the matching test set, calculate performance metrics, and aggregate all results. |

## Input data requirements

Each raw input file should be a CSV file with the following structure:

```text
Mapping.ID, feature_1, feature_2, ..., feature_n, endpoint_column
```

Requirements:

- The first column is expected to be an ID column, usually `Mapping.ID`.
- The last column in the raw CSV is treated as the endpoint by the split script and is renamed to `endpoint`.
- The endpoint must represent a binary outcome and should be coded as `0` and `1` before model training.
- Feature columns must be numeric or convertible to numeric.
- Fisher exact test feature selection assumes binary `0/1` features. Non-binary features may still be processed by other methods, but Fisher results may not be reliable.

## Required R packages

Install the required packages before running the pipeline:

```r
install.packages(c(
  "splitstackshape",
  "ROCR",
  "xgboost",
  "randomForest",
  "e1071",
  "nnet",
  "pROC",
  "ROSE",
  "mltools"
))
```

## Step 1: Create repeated train/test splits

Script:

```bash
Rscript split_training_test.R.R
```

Before running, edit `dir_path` so it points to the folder containing the raw CSV files:

```r
dir_path <- "C:\\path\\to\\raw_csv_files\\"
```

What the script does:

1. Reads CSV files from `dir_path`.
2. Renames the last column to `endpoint`.
3. Converts all columns except the first ID column to numeric.
4. Removes duplicated rows.
5. Creates 20 repeat folders: `repeat_1` through `repeat_20`.
6. Uses stratified sampling by `endpoint` to assign approximately 30% of compounds to the test set.
7. Splits data by `Mapping.ID` so the same compound ID should not appear in both train and test sets.
8. Writes one train CSV and one test CSV per repeat.

Expected output:

```text
<dataset_name>/
  repeat_1/
    YYYY-MM-DD_repeat_1-testdata.csv
    YYYY-MM-DD_repeat_1-traindata.csv
  repeat_2/
    YYYY-MM-DD_repeat_2-testdata.csv
    YYYY-MM-DD_repeat_2-traindata.csv
  ...
  repeat_20/
    YYYY-MM-DD_repeat_20-testdata.csv
    YYYY-MM-DD_repeat_20-traindata.csv
```

Important note:

The current script contains this line inside the file loop:

```r
i = 1
```

This forces the script to process only the first CSV file, even if multiple CSV files are found in `dir_path`. To process all CSV files, remove or comment out this line.

## Step 2: Run feature selection

Script:

```bash
Rscript feature_selection_on_training_data.R
```

Before running, edit `dir_path` so it points to the dataset folder created by Step 1. This folder should contain `repeat_1`, `repeat_2`, ..., `repeat_20`.

```r
dir_path <- "C:\\path\\to\\split_dataset\\"
```

Key configuration options:

```r
CONFIG <- list(
  train_pattern = "traindata\\.csv$",
  id_cols = c("Mapping.ID"),
  endpoint_col = "endpoint"
)
```

Use `id_cols = c("Mapping.ID", "SMILES")` if the data contains two ID columns that should be preserved.

Feature selection methods:

| Method | Output folder | Selection rule |
|---|---|---|
| Fisher exact test | `fisher_exact_test` | Select features with p-value below each configured threshold. |
| Single-feature AUC | `auc_feature_selection` | Select features with AUC above each configured threshold. |
| XGBoost importance | `xg_feature_selection` | Select top N features by XGBoost Gain. |
| Random Forest importance | `RF_feature_selection` | Select top N features by Random Forest importance, usually `MeanDecreaseGini`. |

Default thresholds:

```r
fisher thresholds: seq(0.01, 0.05, by = 0.01)
auc thresholds:    seq(0.52, 0.60, by = 0.02)
xgboost top_n:     seq(10, 50, by = 10)
rf top_n:          seq(10, 50, by = 10)
```

To run only selected feature-selection methods, edit:

```r
RUN_METHODS <- c("fisher", "auc", "xgboost", "rf")
```

Expected output inside each repeat folder:

```text
repeat_1/
  fisher_exact_test/
    YYYY-MM-DD_fisher_feature_pvalue_all.csv
    YYYY-MM-DD_fisher_feature_pvalue_0.05.csv
    selected_data_for_model/
      pvalue_0.01/
        YYYY-MM-DD_fisher_pvalue_0.01_selected_data.csv
      pvalue_0.02/
        YYYY-MM-DD_fisher_pvalue_0.02_selected_data.csv
      ...
  auc_feature_selection/
    YYYY-MM-DD_auc_feature_score_all.csv
    YYYY-MM-DD_auc_feature_score_0.55.csv
    selected_data_for_model/
      auc_0.52/
      auc_0.54/
      ...
  xg_feature_selection/
    YYYY-MM-DD_xgboost_feature_importance_all.csv
    selected_data_for_model/
      xg_10/
      xg_20/
      ...
  RF_feature_selection/
    YYYY-MM-DD_rf_feature_importance_all.csv
    selected_data_for_model/
      rf_10/
      rf_20/
      ...
```

Each selected-data CSV contains:

```text
ID column(s), selected feature columns, endpoint
```

## Step 3: Train models, predict test sets, and aggregate metrics

Script:

```bash
Rscript machine_learning_models.R
```

Before running, edit `dir_path` so it points to the same dataset folder that contains the repeat folders and feature-selection outputs:

```r
dir_path <- "C:\\path\\to\\split_dataset\\"
```

Key configuration options:

```r
CONFIG <- list(
  id_cols = c("Mapping.ID"),
  endpoint_col = "endpoint",
  test_pattern = "testdata\\.csv$",
  selected_train_pattern = "selected_data\\.csv$",
  prediction_output_dir = "model_prediction_results",
  reset_prediction_output = TRUE
)
```

Models run by default:

```r
RUN_MODELS <- c("nb", "svm", "rf", "nnet", "xgboost")
```

These correspond to:

- `nb`: Naive Bayes
- `svm`: Support Vector Machine
- `rf`: Random Forest
- `nnet`: Neural Network
- `xgboost`: XGBoost

Sampling variants applied to the training set:

| Variant | Sampler | Description |
|---|---|---|
| `original` | `none` | Use the original training set. |
| `down` | `under` | Downsample the majority class. |
| `rose` | `ROSE` | Generate a balanced set using ROSE. |
| `up` | `over` | Oversample the minority class. |

For every selected training CSV, the script:

1. Finds the matching test CSV in the same repeat folder.
2. Validates that endpoint values are binary `0/1`.
3. Aligns test-set features to the selected training features.
4. Trains each model under each sampling variant.
5. Predicts the probability of class `1` on the test set.
6. Calculates performance metrics.
7. Writes model-level metrics and prediction files.
8. Aggregates all per-folder metrics into a top-level summary CSV.

Metrics reported:

- AUC
- Best threshold from ROC analysis
- Sensitivity
- Specificity
- Accuracy
- Balanced accuracy
- Matthews correlation coefficient (MCC)
- Number of test samples

Expected output for each selected feature set:

```text
selected_data_for_model/<selection_setting>/
  model_prediction_results/
    original/
      YYYY-MM-DD-nb_metrics.csv
      YYYY-MM-DD-nb_predictions.csv
      YYYY-MM-DD-svm_metrics.csv
      YYYY-MM-DD-svm_predictions.csv
      ...
    down/
    rose/
    up/
    YYYY-MM-DD-metrics_all_models.csv
```

Top-level aggregated output:

```text
<split_dataset>/YYYY-MM-DD-ALL_repeats_metrics_summary.csv
```

The aggregated summary includes the repeat folder, feature-selection path, model name, sampling variant, number of selected features, and test-set performance metrics.

## Recommended project structure

```text
project/
  README.md
  2026-06-25-split_train_test_step_2.R
  1_feature_selection_idcols_universal_improved_step_3.R
  ml_predict_selected_train_to_test_clean_fixed_sampling_step_4.R
  raw_data/
    dataset_1.csv
  outputs/
    dataset_1/
      repeat_1/
      repeat_2/
      ...
      repeat_20/
```

## Common settings to modify

### Input/output root path

Each script has its own hard-coded `dir_path`. Update it before running.

### ID columns

For one ID column:

```r
id_cols = c("Mapping.ID")
```

For two ID columns:

```r
id_cols = c("Mapping.ID", "SMILES")
```

### Endpoint column

If the endpoint column is not already named `endpoint`, set:

```r
endpoint_col = "your_endpoint_column_name"
```

The feature-selection and prediction scripts internally standardize this column to `endpoint`.

### Feature-selection methods

Edit `RUN_METHODS` in the feature-selection script:

```r
RUN_METHODS <- c("fisher", "auc", "xgboost", "rf")
```

### Models

Edit `RUN_MODELS` in the prediction script:

```r
RUN_MODELS <- c("nb", "svm", "rf", "nnet", "xgboost")
```

### Sampling strategy

Edit `sampler_configs` in the prediction script to add, remove, or change sampling variants.

### Output overwrite behavior

By default, prediction outputs are reset before rerunning:

```r
reset_prediction_output = TRUE
```

Set this to `FALSE` if you do not want existing `model_prediction_results` folders to be deleted.

## Troubleshooting

### Only the first raw CSV is processed

Remove or comment out `i = 1` inside the loop in the split script.

### Endpoint contains non-numeric labels

Recode endpoint labels to `0` and `1` before running feature selection or prediction. Labels such as `active/inactive`, `yes/no`, or `positive/negative` will trigger validation errors.

### More than one train or test file is found

The scripts expect exactly one `traindata.csv` and one `testdata.csv` per repeat folder based on their filename patterns. Remove old duplicates or adjust the filename patterns in `CONFIG`.

### Test data is missing selected features

The prediction script requires all selected training features to exist in the test CSV. Make sure the train and test files were generated from the same original dataset and use the same descriptor columns.

### `View(combined)` fails in command-line R

The final line of the prediction script opens the combined summary in RStudio. If running with `Rscript`, comment out:

```r
View(combined)
```

## Reproducibility notes

- Train/test split repeats use seeds `1` through `20`.
- Sampling variants use fixed seeds defined in `sampler_configs`.
- Random Forest, neural network, and XGBoost training also use seeds where configured in the prediction functions.
- Output filenames include `Sys.Date()`, so rerunning the same script on a different date will create files with a new date prefix.

## Suggested run order

```bash
Rscript split_training_test.R
Rscript feature_selection_on_training_data.R
Rscript machine_learning_models.R
```

After completion, use the top-level `YYYY-MM-DD-ALL_repeats_metrics_summary.csv` file to compare feature-selection methods, selected-feature counts, sampling variants, and machine learning models.
