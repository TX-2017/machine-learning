rm(list = ls())

################################################################################
# 0. Input folders
################################################################################

# Root folder that contains repeat folders / split folders.
dir_path <- "C:\\Users\\NCATSCompTox\\Desktop\\Tuan\\1_data_zeo\\2026-06-25-ECFP4_only\\"

folders_list <- list.dirs(dir_path, full.names = TRUE, recursive = FALSE)

################################################################################
# 1. Configuration
################################################################################

CONFIG <- list(
  train_pattern = "traindata\\.csv$",

  # ---------------------------------------------------------------------------
  # Only change these lines when using new data.
  # One ID column:
  # id_cols = c("Mapping.ID")
  #
  # Two ID columns:
  # id_cols = c("Mapping.ID", "SMILES")
  #
  # Endpoint column must be specified by exact column name.
  # ---------------------------------------------------------------------------
  id_cols = c("Mapping.ID"),
  endpoint_col = "endpoint",

  fisher = list(
    output_dir = "fisher_exact_test",
    selected_dir = "selected_data_for_model",
    stat_cutoff = 0.05,
    thresholds = seq(0.01, 0.05, by = 0.01),
    subdir_prefix = "pvalue_"
  ),

  auc = list(
    output_dir = "auc_feature_selection",
    selected_dir = "selected_data_for_model",
    stat_cutoff = 0.55,
    thresholds = seq(0.52, 0.60, by = 0.02),
    subdir_prefix = "auc_"
  ),

  xgboost = list(
    output_dir = "xg_feature_selection",
    selected_dir = "selected_data_for_model",
    top_n = seq(10, 50, by = 10),
    subdir_prefix = "xg_",
    params = list(
      objective = "binary:logistic",
      eval_metric = "error",
      max_depth = 3,
      eta = 0.01,
      gamma = 1,
      colsample_bytree = 0.5,
      min_child_weight = 1
    ),
    nrounds = 1000
  ),

  rf = list(
    output_dir = "RF_feature_selection",
    selected_dir = "selected_data_for_model",
    top_n = seq(10, 50, by = 10),
    subdir_prefix = "rf_"
  )
)

RUN_METHODS <- c("fisher", "auc", "xgboost", "rf")

################################################################################
# 2. Shared functions
################################################################################

get_feature_cols <- function(data, config = CONFIG) {
  setdiff(colnames(data), c(config$id_cols, "endpoint"))
}

write_selected_data <- function(data, selected_features, output_path, config = CONFIG) {
  selected_features <- selected_features[selected_features %in% colnames(data)]

  if (length(selected_features) == 0) {
    return(invisible(FALSE))
  }

  output_data <- data[, c(config$id_cols, selected_features, "endpoint"), drop = FALSE]
  write.csv(output_data, output_path, row.names = FALSE)

  invisible(TRUE)
}

load_train_data <- function(train_file, config = CONFIG) {
  data <- read.csv(
    train_file,
    header = TRUE,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  if (ncol(data) < 3) {
    stop("Input train data must have at least 3 columns. File: ", train_file)
  }

  id_cols <- config$id_cols
  endpoint_col <- config$endpoint_col

  if (!all(id_cols %in% colnames(data))) {
    stop(
      "Some ID columns not found: ",
      paste(setdiff(id_cols, colnames(data)), collapse = ", "),
      " in file: ",
      train_file
    )
  }

  if (!endpoint_col %in% colnames(data)) {
    stop(
      "Endpoint column not found: ",
      endpoint_col,
      " in file: ",
      train_file
    )
  }

  # Standardize endpoint name internally.
  if (endpoint_col != "endpoint") {
    colnames(data)[colnames(data) == endpoint_col] <- "endpoint"
  }

  # Endpoint: convert and hard-validate. Do not silently convert labels to NA.
  raw_endpoint <- as.character(data$endpoint)
  data$endpoint <- suppressWarnings(as.numeric(raw_endpoint))

  introduced_na <- is.na(data$endpoint) &
    !is.na(raw_endpoint) &
    trimws(raw_endpoint) != ""

  if (any(introduced_na)) {
    bad <- unique(raw_endpoint[introduced_na])
    stop(
      "Endpoint contains non-numeric labels that cannot become 0/1: ",
      paste(head(bad, 5), collapse = ", "),
      ". Recode to 0/1 first. File: ",
      train_file
    )
  }

  n_na_ep <- sum(is.na(data$endpoint))
  if (n_na_ep > 0) {
    warning(n_na_ep, " row(s) with NA endpoint dropped. File: ", train_file)
    data <- data[!is.na(data$endpoint), , drop = FALSE]
  }

  ep_levels <- sort(unique(data$endpoint))
  if (!all(ep_levels %in% c(0, 1)) || length(ep_levels) != 2) {
    stop(
      "Endpoint must be binary 0/1 and both classes must be present. Found: ",
      paste(ep_levels, collapse = ", "),
      ". File: ",
      train_file
    )
  }

  # Features: convert and validate.
  feature_cols <- get_feature_cols(data, config)

  if (length(feature_cols) == 0) {
    stop("No feature columns found after excluding ID and endpoint columns. File: ", train_file)
  }

  data[feature_cols] <- lapply(
    data[feature_cols],
    function(col) suppressWarnings(as.numeric(col))
  )

  all_na_feats <- feature_cols[vapply(
    data[feature_cols],
    function(col) all(is.na(col)),
    logical(1)
  )]

  if (length(all_na_feats) > 0) {
    stop(
      "Feature column(s) all-NA after numeric conversion: ",
      paste(head(all_na_feats, 5), collapse = ", "),
      ". File: ",
      train_file
    )
  }

  feats_with_na <- feature_cols[vapply(data[feature_cols], anyNA, logical(1))]
  if (length(feats_with_na) > 0) {
    warning(
      length(feats_with_na),
      " feature col(s) contain NA; XGBoost/RF may fail. e.g. ",
      paste(head(feats_with_na, 5), collapse = ", "),
      ". File: ",
      train_file
    )
  }

  # Fisher 2x2 table assumes strictly 0/1 features.
  non_binary <- feature_cols[vapply(data[feature_cols], function(col) {
    u <- unique(col[!is.na(col)])
    !all(u %in% c(0, 1))
  }, logical(1))]

  if (length(non_binary) > 0) {
    warning(
      length(non_binary),
      " feature col(s) not strictly 0/1; Fisher results may be unreliable. e.g. ",
      paste(head(non_binary, 5), collapse = ", "),
      ". File: ",
      train_file
    )
  }

  data
}

################################################################################
# 3. Fisher exact test feature selection
################################################################################

run_fisher_fs_one_folder <- function(data, folder_path, config = CONFIG) {
  feature_cols <- get_feature_cols(data, config)
  output_root <- file.path(folder_path, config$fisher$output_dir)
  dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

  pvalues <- sapply(feature_cols, function(feature) {
    x <- data[[feature]]
    y <- data$endpoint
    
    # Pairwise-complete deletion, same NA handling as the AUC method.
    ok <- complete.cases(x, y)
    x <- x[ok]
    y <- y[ok]
    
    matrix_data <- matrix(
      c(
        sum(x == 1 & y == 1),
        sum(x == 1 & y == 0),
        sum(x == 0 & y == 1),
        sum(x == 0 & y == 0)
      ),
      nrow = 2,
      byrow = TRUE
    )
    
    fisher.test(matrix_data, alternative = "two.sided")$p.value
  })

  stat_data <- data.frame(
    feature = feature_cols,
    pvalue = as.numeric(pvalues)
  )
  stat_data <- stat_data[order(stat_data$pvalue), ]

  write.csv(
    stat_data,
    file.path(output_root, paste0(Sys.Date(), "_fisher_feature_pvalue_all.csv")),
    row.names = FALSE
  )

  selected_stat <- stat_data[stat_data$pvalue <= config$fisher$stat_cutoff, ]
  write.csv(
    selected_stat,
    file.path(output_root, paste0(Sys.Date(), "_fisher_feature_pvalue_", config$fisher$stat_cutoff, ".csv")),
    row.names = FALSE
  )

  selected_root <- file.path(output_root, config$fisher$selected_dir)
  dir.create(selected_root, recursive = TRUE, showWarnings = FALSE)

  for (threshold in config$fisher$thresholds) {
    selected_features <- stat_data$feature[stat_data$pvalue <= threshold]
    out_dir <- file.path(selected_root, paste0(config$fisher$subdir_prefix, threshold))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    write_selected_data(
      data,
      selected_features,
      file.path(out_dir, paste0(Sys.Date(), "_fisher_pvalue_", threshold, "_selected_data.csv")),
      config
    )
  }

  invisible(stat_data)
}

################################################################################
# 4. ROC/AUC feature selection
################################################################################

safe_auc <- function(feature_value, endpoint) {
  ok <- complete.cases(feature_value, endpoint)
  feature_value <- feature_value[ok]
  endpoint <- endpoint[ok]

  if (length(unique(endpoint)) < 2 || length(unique(feature_value)) < 2) {
    return(NA_real_)
  }

  pred <- ROCR::prediction(feature_value, endpoint)
  auc_value <- ROCR::performance(pred, "auc")@y.values[[1]]

  if (is.na(auc_value)) {
    return(NA_real_)
  }

  if (auc_value < 0.5) {
    auc_value <- 1 - auc_value
  }

  auc_value
}

run_auc_fs_one_folder <- function(data, folder_path, config = CONFIG) {
  if (!requireNamespace("ROCR", quietly = TRUE)) {
    stop("Package 'ROCR' is required for AUC feature selection.")
  }

  feature_cols <- get_feature_cols(data, config)
  output_root <- file.path(folder_path, config$auc$output_dir)
  dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

  auc_values <- sapply(feature_cols, function(feature) {
    safe_auc(data[[feature]], data$endpoint)
  })

  stat_data <- data.frame(
    feature = feature_cols,
    auc = as.numeric(auc_values)
  )
  stat_data <- stat_data[order(stat_data$auc, decreasing = TRUE, na.last = TRUE), ]

  write.csv(
    stat_data,
    file.path(output_root, paste0(Sys.Date(), "_auc_feature_score_all.csv")),
    row.names = FALSE
  )

  selected_stat <- stat_data[!is.na(stat_data$auc) & stat_data$auc >= config$auc$stat_cutoff, ]
  write.csv(
    selected_stat,
    file.path(output_root, paste0(Sys.Date(), "_auc_feature_score_", config$auc$stat_cutoff, ".csv")),
    row.names = FALSE
  )

  selected_root <- file.path(output_root, config$auc$selected_dir)
  dir.create(selected_root, recursive = TRUE, showWarnings = FALSE)

  for (threshold in config$auc$thresholds) {
    selected_features <- stat_data$feature[!is.na(stat_data$auc) & stat_data$auc >= threshold]
    out_dir <- file.path(selected_root, paste0(config$auc$subdir_prefix, threshold))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    write_selected_data(
      data,
      selected_features,
      file.path(out_dir, paste0(Sys.Date(), "_auc_", threshold, "_selected_data.csv")),
      config
    )
  }

  invisible(stat_data)
}

################################################################################
# 5. XGBoost feature importance selection
################################################################################

run_xgboost_fs_one_folder <- function(data, folder_path, config = CONFIG) {
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("Package 'xgboost' is required for XGBoost feature selection.")
  }

  feature_cols <- get_feature_cols(data, config)
  output_root <- file.path(folder_path, config$xgboost$output_dir)
  dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

  train_data <- as.matrix(data[, feature_cols, drop = FALSE])
  train_label <- data$endpoint
  train_matrix <- xgboost::xgb.DMatrix(data = train_data, label = train_label)

  bst_model <- xgboost::xgb.train(
    params = config$xgboost$params,
    data = train_matrix,
    nrounds = config$xgboost$nrounds,
    verbose = 0
  )

  stat_data <- xgboost::xgb.importance(feature_names = colnames(train_data), model = bst_model)
  stat_data <- stat_data[order(stat_data$Gain, decreasing = TRUE), ]

  write.csv(
    stat_data,
    file.path(output_root, paste0(Sys.Date(), "_xgboost_feature_importance_all.csv")),
    row.names = FALSE
  )

  selected_root <- file.path(output_root, config$xgboost$selected_dir)
  dir.create(selected_root, recursive = TRUE, showWarnings = FALSE)

  for (top_n in config$xgboost$top_n) {
    selected_features <- head(stat_data$Feature, top_n)
    out_dir <- file.path(selected_root, paste0(config$xgboost$subdir_prefix, top_n))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    write_selected_data(
      data,
      selected_features,
      file.path(out_dir, paste0(Sys.Date(), "_xgboost_top_", top_n, "_selected_data.csv")),
      config
    )
  }

  invisible(stat_data)
}

################################################################################
# 6. Random Forest feature importance selection
################################################################################

run_rf_fs_one_folder <- function(data, folder_path, config = CONFIG) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Package 'randomForest' is required for RF feature selection.")
  }

  feature_cols <- get_feature_cols(data, config)
  output_root <- file.path(folder_path, config$rf$output_dir)
  dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

  fit_rf <- randomForest::randomForest(
    x = data[, feature_cols, drop = FALSE],
    y = as.factor(data$endpoint),
    importance = FALSE
  )

  importance_data <- as.data.frame(randomForest::importance(fit_rf))
  importance_data$feature <- rownames(importance_data)

  if ("MeanDecreaseGini" %in% colnames(importance_data)) {
    stat_data <- importance_data[order(importance_data$MeanDecreaseGini, decreasing = TRUE), ]
  } else {
    stat_data <- importance_data
  }

  rownames(stat_data) <- NULL
  stat_data <- stat_data[, c("feature", setdiff(colnames(stat_data), "feature")), drop = FALSE]

  write.csv(
    stat_data,
    file.path(output_root, paste0(Sys.Date(), "_rf_feature_importance_all.csv")),
    row.names = FALSE
  )

  selected_root <- file.path(output_root, config$rf$selected_dir)
  dir.create(selected_root, recursive = TRUE, showWarnings = FALSE)

  for (top_n in config$rf$top_n) {
    selected_features <- head(stat_data$feature, top_n)
    out_dir <- file.path(selected_root, paste0(config$rf$subdir_prefix, top_n))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    write_selected_data(
      data,
      selected_features,
      file.path(out_dir, paste0(Sys.Date(), "_rf_top_", top_n, "_selected_data.csv")),
      config
    )
  }

  invisible(stat_data)
}

################################################################################
# 7. Run feature selection for one folder
################################################################################

run_feature_selection_one_folder <- function(folder_path, methods = RUN_METHODS, config = CONFIG) {
  traindata_list <- dir(
    folder_path,
    pattern = config$train_pattern,
    full.names = TRUE,
    recursive = TRUE
  )

  if (length(traindata_list) == 0) {
    return(invisible(NULL))
  }

  if (length(traindata_list) > 1) {
    stop("More than one traindata.csv file found in folder: ", folder_path)
  }

  data <- load_train_data(traindata_list[1], config)

  results <- list()

  if ("fisher" %in% methods) {
    results$fisher <- run_fisher_fs_one_folder(data, folder_path, config)
  }
  if ("auc" %in% methods) {
    results$auc <- run_auc_fs_one_folder(data, folder_path, config)
  }
  if ("xgboost" %in% methods) {
    results$xgboost <- run_xgboost_fs_one_folder(data, folder_path, config)
  }
  if ("rf" %in% methods) {
    results$rf <- run_rf_fs_one_folder(data, folder_path, config)
  }

  invisible(results)
}

################################################################################
# 8. Run all folders
################################################################################

all_results <- vector("list", length(folders_list))
failed_folders <- character(0)

for (i in seq_along(folders_list)) {
  folder <- folders_list[i]

  all_results[[i]] <- tryCatch(
    run_feature_selection_one_folder(
      folder_path = folder,
      methods = RUN_METHODS,
      config = CONFIG
    ),
    error = function(e) {
      message("[FAILED] ", folder, " -> ", conditionMessage(e))
      failed_folders <<- c(failed_folders, folder)
      NULL
    }
  )
}

names(all_results) <- basename(folders_list)

if (length(failed_folders) > 0) {
  message("\n", length(failed_folders), " folder(s) failed:")
  message(paste(" -", failed_folders, collapse = "\n"))
} else {
  message("\nAll ", length(folders_list), " folder(s) completed.")
}
