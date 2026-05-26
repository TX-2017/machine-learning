# Machine Learning Modeling and Feature Selection Pipeline

This project contains two R scripts for binary-classification datasets. The workflow first performs feature selection, then batch-trains and evaluates multiple machine learning models using the selected feature sets.

## Files

|File|Purpose|
|-|-|
|`Feature selection.R`|Performs feature selection on raw CSV files. Supported methods include Fisher's exact test, single-feature ROC/AUC screening, XGBoost feature importance, and Random Forest feature importance.|
|`machine learning.R`|Reads the feature-selected CSV files, repeatedly splits data into training and test sets, and evaluates Naive Bayes, SVM, Random Forest, NNET, and XGBoost models.|

## Workflow

Recommended execution order:

1. Update the input paths and parameters in `Feature selection.R`.
2. Run `Feature selection.R` to generate feature-selected datasets.
3. Update the input path in `machine learning.R`.
4. Run `machine learning.R` to train and evaluate models in batch.

## Environment Requirements

R 4.x or later is recommended. The scripts require the following R packages:

```r
install.packages(c(
  "e1071",
  "randomForest",
  "nnet",
  "ROCR",
  "pROC",
  "mltools",
  "ROSE",
  "splitstackshape",
  "xgboost"
))
```

Load the packages with:

```r
library(e1071)
library(randomForest)
library(nnet)
library(ROCR)
library(pROC)
library(mltools)
library(ROSE)
library(splitstackshape)
library(xgboost)
```

## Input Data Format

Input files should be CSV files located in the directory specified by `CONFIG$dir\_path` or `dir\_path\_ori`.

The scripts assume that each CSV file has the following structure:

|Column position|Description|
|-|-|
|Column 1|Sample ID, for example `Mapping.ID`.|
|Column 2|Sample description, for example `Structure\_SMILES\_2D.QSAR`.|
|Column 3 to the second-to-last column|Feature columns. The Fisher exact test workflow selects feature columns whose names start with `V` followed by digits, such as `V1`, `V2`, or `V100`.|
|Last column|Binary class label. Recommended values are `0` and `1`. In the modeling script, this column is renamed to `num`.|

The scripts convert columns from the third column through the last column to numeric values. Make sure that all feature columns and the label column can be safely converted to `numeric`.

## 1\. Feature Selection Script

Script: `Feature selection.R`

### Main Configuration

Modify `CONFIG` near the top of the script:

```r
CONFIG <- list(
  dir\_path = "D:\\\\2025\\\\20250808\_wnt\\\\2\_ML\\\\",
  csv\_pattern = "\\\\.csv$",
  ...
)
```

Common configuration items:

|Parameter|Description|
|-|-|
|`CONFIG$dir\_path`|Directory containing the raw CSV input files.|
|`CONFIG$csv\_pattern`|Pattern used to identify input files. The default matches `.csv` files.|
|`CONFIG$fisher$thresholds`|Fisher exact test p-value thresholds. The default range is `0.01` to `0.05`.|
|`CONFIG$auc$thresholds`|Single-feature AUC thresholds. The default range is `0.52` to `0.60`.|
|`CONFIG$xgboost$top\_n`|Numbers of top-ranked features to retain based on XGBoost importance. Defaults include 20, 40, 60, 80, and 100.|
|`CONFIG$rf$top\_n`|Numbers of top-ranked features to retain based on Random Forest importance. Defaults include 20, 40, 60, 80, and 100.|

### Supported Feature Selection Methods

By default, all four methods are executed:

```r
RUN\_METHODS <- c("fisher", "auc", "xgboost", "rf")
```

You can modify this vector to run only selected methods. For example, to run Fisher and AUC only:

```r
RUN\_METHODS <- c("fisher", "auc")
```

### Output Directories

Feature selection results are written under `CONFIG$dir\_path`.

|Method|Output directory|Description|
|-|-|-|
|Fisher exact test|`fisher exact test/`|Outputs p-values for each feature and generates selected datasets based on p-value thresholds.|
|ROC/AUC|`auc\_feature\_selection/`|Outputs AUC values for each feature and generates selected datasets based on AUC thresholds.|
|XGBoost|`xg\_featue\_selection/`|Ranks features by XGBoost Gain and generates top-N feature datasets.|
|Random Forest|`RF\_featue\_selection/`|Ranks features by MeanDecreaseGini and generates top-N feature datasets.|

The selected datasets used for modeling are stored in the following subdirectory under each method:

```text
selected\_data\_for\_model/
```

Examples:

```text
auc\_feature\_selection/selected\_data\_for\_model/pvalue\_0.56/
xg\_featue\_selection/selected\_data\_for\_model/xg\_60/
RF\_featue\_selection/selected\_data\_for\_model/rf\_100/
```

## 2\. Machine Learning Modeling Script

Script: `machine learning.R`

### Main Configuration

Update the root path in the script:

```r
dir\_path\_ori <- "D:\\\\2025\\\\20250808\_wnt\\\\2\_ML\\\\"
```

This path should match `CONFIG$dir\_path` in `Feature selection.R`.

By default, the modeling script reads feature-selected datasets from the following four directories:

```r
fs\_dir <- c(
  "auc\_feature\_selection",
  "fisher exact test",
  "xg\_featue\_selection",
  "RF\_featue\_selection"
)
```

Each model is repeated 20 times by default:

```r
loop <- 20
```

### Data Splitting and Sampling Strategy

In each loop, the script:

1. Performs stratified sampling based on the `num` class label.
2. Uses approximately 30% of samples as the test set and the remaining samples as the training set.
3. Applies the specified sampling strategy to the training set.
4. Trains the model and calculates performance metrics.

Supported sampling configurations:

|Variant|Sampling method|Output suffix|
|-|-|-|
|`test`|No sampling|`test`|
|`down`|Down-sampling|`down`|
|`ROSE` / `rose`|ROSE sampling|`ROSE` or `rose`|
|`up`|Up-sampling|`up`|

### Models

Classic models:

* Naive Bayes
* Support Vector Machine, SVM
* Random Forest
* Neural Network using `nnet`

XGBoost model:

* Trained with `xgb.train()`
* Binary-classification objective: `binary:logistic`
* Default `nrounds = 1000`

### Evaluation Metrics

For each model, the script reports the mean and standard deviation of the following metrics:

|Metric|Description|
|-|-|
|`auc`|ROC AUC.|
|`accuracy`|Classification accuracy.|
|`Balanced\_accuracy`|Balanced accuracy.|
|`mcc\_result`|Matthews correlation coefficient.|

The output table is organized as:

```text
mean\_auc, sd\_auc,
mean\_accuracy, sd\_accuracy,
mean\_Balanced\_accuracy, sd\_Balanced\_accuracy,
mean\_mcc\_result, sd\_mcc\_result
```

### Modeling Output Directories

For each selected dataset subdirectory, the script automatically creates model-result directories such as:

```text
nb\_test/
svm\_test/
rf\_test/
nnet\_test/
xgboost\_test/

nb\_down/
svm\_down/
rf\_down/
nnet\_down/
xgboost\_down/

nb\_up/
svm\_up/
rf\_up/
nnet\_up/
xgboost\_up/

nb\_ROSE/
svm\_ROSE/
rf\_ROSE/
nnet\_ROSE/
xgboost\_rose/
```

Each output CSV file corresponds to one input dataset and one model configuration.

## Quick Start

### 1\. Run Feature Selection

```r
source("Feature selection.R")
```

### 2\. Run Machine Learning Models

```r
source("machine learning.R")
```

## Common Modifications

### Run Only Selected Feature Selection Methods

```r
RUN\_METHODS <- c("xgboost", "rf")
```

### Run Classic Models Only, Excluding XGBoost

In `machine learning.R`, modify:

```r
run\_classic\_model\_family <- TRUE
run\_xgboost\_family <- FALSE
```

### Run XGBoost Only

```r
run\_classic\_model\_family <- FALSE
run\_xgboost\_family <- TRUE
```

### Change the Number of Repeats

```r
loop <- 50
```

### Modify XGBoost Parameters

In the feature selection script:

```r
CONFIG$xgboost$params <- list(
  objective = "binary:logistic",
  eval\_metric = "error",
  max\_depth = 3,
  eta = 0.01,
  gamma = 1,
  colsample\_bytree = 0.5,
  min\_child\_weight = 1
)
```

Note: In the current script, the parameter name appears as `gammma`. To ensure that XGBoost recognizes the parameter, it is recommended to correct it to `gamma`.

## Notes

1. The paths currently use Windows-style syntax, for example `D:\\\\2025\\\\20250808\_wnt\\\\2\_ML\\\\`. On macOS or Linux, use `/` paths or build paths with `file.path()`.
2. `machine learning.R` deletes and recreates the corresponding model output directories before writing results. Back up existing results before rerunning the script.
3. The binary class label should contain both `0` and `1`. If a train/test split or prediction result contains only one class, some metric calculations may fail. The script uses `tryCatch()` to catch errors and continue execution.
4. The Fisher exact test method only processes feature columns matching `^V\[\[:digit:]]` by default. If your feature columns use different names, update the `grep()` rule.
5. Output file names include `Sys.Date()`. Running the same workflow multiple times on the same day may overwrite files with the same name.
6. Some directory names preserve the original spelling used in the scripts, such as `xg\_featue\_selection` and `RF\_featue\_selection`. If you rename them, update the directory settings in both scripts accordingly.

## Recommended Project Structure

```text
project/
├── Feature selection.R
├── machine learning.R
├── README.md
└── data/
    ├── input\_1.csv
    ├── input\_2.csv
    └── ...
```

In practice, set `CONFIG$dir\_path` and `dir\_path\_ori` to the working directory that contains the CSV files and generated output folders.

