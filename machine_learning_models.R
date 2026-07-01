rm(list = ls())

################################################################################
# 0. Load packages
################################################################################

library(e1071)
library(randomForest)
library(nnet)
library(pROC)
library(ROSE)
library(xgboost)
library(mltools)

################################################################################
# 1. Input folders
################################################################################

# Root folder that contains repeat_1, repeat_2, ..., repeat_20.
dir_path <- "C:\\Users\\NCATSCompTox\\Desktop\\Tuan\\zeo\\fp4_only\\"

repeat_folders <- list.dirs(dir_path, full.names = TRUE, recursive = FALSE)

################################################################################
# 2. Configuration
################################################################################

CONFIG <- list(
  id_cols = c("Mapping.ID"),
  endpoint_col = "endpoint",
  
  test_pattern = "testdata\\.csv$",
  selected_train_pattern = "selected_data\\.csv$",
  
  prediction_output_dir = "model_prediction_results",
  reset_prediction_output = TRUE,
  
  sampler_configs = list(
    list(variant = "original", sampler = "none",  seed = 2020L),
    list(variant = "down",     sampler = "under", seed = 2022L),
    list(variant = "rose",     sampler = "ROSE",  seed = 2021L),
    list(variant = "up",       sampler = "over",  seed = 2020L)
  ),
  
  xgboost = list(
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
  
  nnet = list(size = 10, MaxNWts = 100000, maxit = 200)
)

RUN_MODELS <- c("nb", "svm", "rf", "nnet", "xgboost")

################################################################################
# 3. Data helpers
################################################################################

try_or_null <- function(expr, prefix) {
  tryCatch(expr, error = function(e) { message(prefix, " -> ", conditionMessage(e)); NULL })
}

find_one_file <- function(folder, pattern, recursive = TRUE, description = "file") {
  files <- dir(folder, pattern = pattern, full.names = TRUE, recursive = recursive)
  files <- files[!grepl("model_prediction_results", files, fixed = TRUE)]
  
  if (length(files) == 0) stop("No ", description, " found in folder: ", folder)
  if (length(files) > 1) {
    stop("More than one ", description, " found in folder: ", folder,
         "\n", paste(files, collapse = "\n"))
  }
  files[1]
}

standardize_data <- function(file_path, config = CONFIG) {
  data <- read.csv(file_path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  
  if (ncol(data) < 3) {
    stop("Data must have at least 3 columns: ID, feature(s), endpoint. File: ", file_path)
  }
  if (!all(config$id_cols %in% colnames(data))) {
    stop("Some ID columns not found: ",
         paste(setdiff(config$id_cols, colnames(data)), collapse = ", "), ". File: ", file_path)
  }
  
  if ("endpoint" %in% colnames(data)) {
    # already named endpoint
  } else if (config$endpoint_col %in% colnames(data)) {
    colnames(data)[colnames(data) == config$endpoint_col] <- "endpoint"
  } else {
    stop("Endpoint column not found: ", config$endpoint_col, ". File: ", file_path)
  }
  
  raw_endpoint <- as.character(data$endpoint)
  data$endpoint <- suppressWarnings(as.numeric(raw_endpoint))
  
  introduced_na <- is.na(data$endpoint) & !is.na(raw_endpoint) & trimws(raw_endpoint) != ""
  if (any(introduced_na)) {
    stop("Endpoint contains non-numeric labels that cannot become 0/1: ",
         paste(head(unique(raw_endpoint[introduced_na]), 5), collapse = ", "),
         ". Recode to 0/1 first. File: ", file_path)
  }
  
  data <- data[!is.na(data$endpoint), , drop = FALSE]
  
  ep_levels <- sort(unique(data$endpoint))
  if (!all(ep_levels %in% c(0, 1)) || length(ep_levels) < 2) {
    stop("Endpoint must be binary 0/1 and both classes must be present. Found: ",
         paste(ep_levels, collapse = ", "), ". File: ", file_path)
  }
  
  feature_cols <- setdiff(colnames(data), c(config$id_cols, "endpoint"))
  if (length(feature_cols) == 0) stop("No feature columns found. File: ", file_path)
  
  data[feature_cols] <- lapply(data[feature_cols], function(col) suppressWarnings(as.numeric(col)))
  
  all_na <- feature_cols[vapply(data[feature_cols], function(x) all(is.na(x)), logical(1))]
  if (length(all_na) > 0) {
    stop("Feature column(s) all-NA after numeric conversion: ",
         paste(head(all_na, 5), collapse = ", "), ". File: ", file_path)
  }
  
  data
}

get_feature_cols <- function(data, config = CONFIG) {
  setdiff(colnames(data), c(config$id_cols, "endpoint"))
}

align_train_test <- function(train_data, test_data, train_file, test_file, config = CONFIG) {
  train_features <- get_feature_cols(train_data, config)
  missing_features <- setdiff(train_features, colnames(test_data))
  
  if (length(missing_features) > 0) {
    stop("Test data is missing selected feature(s): ",
         paste(head(missing_features, 10), collapse = ", "),
         ". Train file: ", train_file, ". Test file: ", test_file)
  }
  
  list(
    train = train_data[, c(config$id_cols, train_features, "endpoint"), drop = FALSE],
    test  = test_data[,  c(config$id_cols, train_features, "endpoint"), drop = FALSE],
    features = train_features
  )
}

remove_id_cols <- function(data, config = CONFIG) {
  data[, setdiff(colnames(data), config$id_cols), drop = FALSE]
}

balance_training_set <- function(train_df, sampler, seed = 1L) {
  if (!is.data.frame(train_df)) stop("balance_training_set expects a data.frame.")
  if (!"num" %in% colnames(train_df)) stop("Training data must contain outcome column named 'num'.")
  
  balanced_df <- train_df
  balanced_df$num <- as.factor(as.character(balanced_df$num))
  if (sampler == "none") return(balanced_df)
  
  set.seed(seed)
  class_table <- table(balanced_df$num)
  if (length(class_table) != 2) {
    stop("Sampling requires exactly two classes. Found: ", paste(names(class_table), collapse = ", "))
  }
  
  split_by_class <- split(seq_len(nrow(balanced_df)), balanced_df$num)
  
  if (sampler == "under") {
    n_target <- min(as.integer(class_table))
    keep_idx <- unlist(lapply(split_by_class, function(idx) sample(idx, n_target, replace = FALSE)))
    return(balanced_df[keep_idx, , drop = FALSE])
  }
  
  if (sampler == "over") {
    n_target <- max(as.integer(class_table))
    keep_idx <- unlist(lapply(split_by_class, function(idx) {
      if (length(idx) == n_target) idx else sample(idx, n_target, replace = TRUE)
    }))
    return(balanced_df[keep_idx, , drop = FALSE])
  }
  
  if (sampler == "ROSE") {
    return(ROSE::ROSE(num ~ ., data = balanced_df, seed = seed)$data)
  }
  
  stop("Unknown sampler: ", sampler)
}

################################################################################
# 4. Metrics
################################################################################

calc_binary_metrics <- function(scores, actuals) {
  actuals <- as.numeric(as.character(actuals))
  scores  <- as.numeric(scores)
  
  ok <- complete.cases(scores, actuals)
  scores  <- scores[ok]
  actuals <- actuals[ok]
  
  na_row <- data.frame(auc = NA_real_, threshold = NA_real_, sensitivity = NA_real_,
                       specificity = NA_real_, accuracy = NA_real_,
                       balanced_accuracy = NA_real_, mcc = NA_real_, n_test = length(actuals))
  
  if (length(scores) == 0 || length(unique(actuals)) < 2 || length(unique(scores)) < 2) {
    return(na_row)
  }
  
  # Build the ROC object ONCE and reuse it for both AUC and the best threshold.
  # levels = c(0, 1) + direction = "<" pins the orientation so that a HIGHER
  # score always means "more likely class 1". This matches the prediction rule
  # below (predicted <- scores >= cutoff) and reproduces ROCR's AUC convention;
  # without it, pROC's default direction = "auto" can flip the AUC for
  # worse-than-random models and disagree with the threshold logic.
  modelroc <- roc(actuals, scores, levels = c(0, 1), direction = "<", quiet = TRUE)
  
  # AUC straight from the same pROC object (no ROCR needed).
  auc_value <- as.numeric(auc(modelroc))
  
  # Best threshold (Youden's J by default in pROC).
  cutoff <- as.numeric(coords(modelroc, "best", ret = "threshold", transpose = FALSE)$threshold)
  cutoff <- cutoff[is.finite(cutoff)]
  cutoff <- if (length(cutoff) == 0) stats::median(scores, na.rm = TRUE) else cutoff[1]
  
  predicted <- ifelse(scores >= cutoff, 1, 0)
  tp <- sum(actuals == 1 & predicted == 1); tn <- sum(actuals == 0 & predicted == 0)
  fp <- sum(actuals == 0 & predicted == 1); fn <- sum(actuals == 1 & predicted == 0)
  
  sensitivity <- ifelse((tp + fn) == 0, NA_real_, tp / (tp + fn))
  specificity <- ifelse((tn + fp) == 0, NA_real_, tn / (tn + fp))
  
  data.frame(
    auc = auc_value,
    threshold = cutoff,
    sensitivity = sensitivity,
    specificity = specificity,
    accuracy = mean(actuals == predicted),
    balanced_accuracy = mean(c(sensitivity, specificity), na.rm = TRUE),
    mcc = mcc(as.numeric(predicted), as.numeric(actuals)),
    n_test = length(actuals)
  )
}

make_prediction_frame <- function(test_data, scores, actuals, threshold,
                                  model_name, variant, train_file, config = CONFIG) {
  scores <- as.numeric(scores)
  threshold <- as.numeric(threshold)
  threshold <- threshold[is.finite(threshold)]
  threshold <- if (length(threshold) == 0) stats::median(scores, na.rm = TRUE) else threshold[1]
  
  data.frame(
    test_data[, config$id_cols, drop = FALSE],
    actual_endpoint = actuals,
    predicted_probability = scores,
    predicted_endpoint = ifelse(scores >= threshold, 1, 0),
    model = model_name,
    variant = variant,
    train_file = basename(train_file),
    check.names = FALSE
  )
}

################################################################################
# 5. Model scorers: each returns a probability vector for class "1" on test set
################################################################################

# Build a model-ready train/test pair once per (file, variant).
prepare_model_data <- function(train_data, test_data, config_sampler, config = CONFIG) {
  train_md <- remove_id_cols(train_data, config)
  test_md  <- remove_id_cols(test_data, config)
  
  colnames(train_md)[ncol(train_md)] <- "num"
  colnames(test_md)[ncol(test_md)]   <- "num"
  
  feats <- setdiff(colnames(train_md), "num")
  train_md[feats] <- lapply(train_md[feats], as.numeric)
  test_md[feats]  <- lapply(test_md[feats], as.numeric)
  
  train_md$num <- as.factor(train_md$num)
  train_bal <- balance_training_set(train_md, config_sampler$sampler, config_sampler$seed)
  train_bal$num <- as.factor(train_bal$num)
  
  list(
    train = train_bal,
    test = test_md,
    features = feats,
    test_actual = as.numeric(as.character(test_md$num))
  )
}

model_scorers <- list(
  nb = function(tr, te, feats, seed, config) {
    fit <- naiveBayes(x = tr[, feats, drop = FALSE], y = tr$num, laplace = 1)
    predict(fit, te[, feats, drop = FALSE], type = "raw")[, "1"]
  },
  
  svm = function(tr, te, feats, seed, config) {
    fit <- svm(x = tr[, feats, drop = FALSE], y = tr$num, scale = FALSE, probability = TRUE)
    p  <- predict(fit, te[, feats, drop = FALSE], probability = TRUE)
    pr <- attr(p, "probabilities")
    if ("1" %in% colnames(pr)) pr[, "1"] else as.numeric(p)
  },
  
  rf = function(tr, te, feats, seed, config) {
    set.seed(seed)
    fit <- randomForest(x = tr[, feats, drop = FALSE], y = tr$num)
    pr  <- predict(fit, te[, feats, drop = FALSE], type = "prob")
    if ("1" %in% colnames(pr)) pr[, "1"] else pr[, ncol(pr)]
  },
  
  nnet = function(tr, te, feats, seed, config) {
    set.seed(seed)
    fit <- nnet(x = as.matrix(tr[, feats, drop = FALSE]),
                y = as.numeric(as.character(tr$num)),
                size = config$nnet$size, MaxNWts = config$nnet$MaxNWts,
                maxit = config$nnet$maxit, linout = FALSE, trace = FALSE)
    as.numeric(predict(fit, as.matrix(te[, feats, drop = FALSE]), type = "raw"))
  },
  
  xgboost = function(tr, te, feats, seed, config) {
    set.seed(seed)
    train_y <- as.numeric(as.character(tr$num))
    test_y  <- as.numeric(as.character(te$num))
    if (!all(train_y %in% c(0, 1))) stop("XGBoost labels must be 0/1. Found: ",
                                         paste(sort(unique(train_y)), collapse = ", "))
    dtr <- xgb.DMatrix(as.matrix(tr[, feats, drop = FALSE]), label = train_y)
    dte <- xgb.DMatrix(as.matrix(te[, feats, drop = FALSE]), label = test_y)
    bst <- xgb.train(params = config$xgboost$params, data = dtr,
                     nrounds = config$xgboost$nrounds, verbose = 0)
    predict(bst, dte)
  }
)

################################################################################
# 6. Run one selected training file
################################################################################

run_one_selected_train_file <- function(train_file, test_file, config = CONFIG) {
  train_data_raw <- standardize_data(train_file, config)
  test_data_raw  <- standardize_data(test_file, config)
  
  aligned <- align_train_test(train_data_raw, test_data_raw, train_file, test_file, config)
  train_data <- aligned$train
  test_data  <- aligned$test
  
  output_dir <- file.path(dirname(train_file), config$prediction_output_dir)
  if (config$reset_prediction_output && dir.exists(output_dir)) unlink(output_dir, recursive = TRUE)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  all_metrics <- list()
  
  for (sc in config$sampler_configs) {
    variant_dir <- file.path(output_dir, sc$variant)
    dir.create(variant_dir, recursive = TRUE, showWarnings = FALSE)
    
    prep <- try_or_null(
      prepare_model_data(train_data, test_data, sc, config),
      paste0("[FAILED prep] ", train_file, " | ", sc$variant)
    )
    if (is.null(prep)) next
    
    for (model_name in RUN_MODELS) {
      scores <- try_or_null(
        model_scorers[[model_name]](prep$train, prep$test, prep$features, sc$seed, config),
        paste0("[FAILED ", model_name, "] ", train_file, " | ", sc$variant)
      )
      if (is.null(scores)) next
      
      metric <- cbind(
        data.frame(model = model_name, variant = sc$variant,
                   train_file = basename(train_file), test_file = basename(test_file),
                   n_features = length(aligned$features)),
        calc_binary_metrics(scores, prep$test_actual)
      )
      
      pred <- make_prediction_frame(test_data, scores, prep$test_actual,
                                    metric$threshold[1], model_name, sc$variant, train_file, config)
      
      write.csv(metric, file.path(variant_dir, paste0(Sys.Date(), "-", model_name, "_metrics.csv")), row.names = FALSE)
      write.csv(pred,   file.path(variant_dir, paste0(Sys.Date(), "-", model_name, "_predictions.csv")), row.names = FALSE)
      
      all_metrics[[paste(sc$variant, model_name, sep = "_")]] <- metric
    }
  }
  
  if (length(all_metrics) > 0) {
    metrics_all <- do.call(rbind, all_metrics)
    rownames(metrics_all) <- NULL
    write.csv(metrics_all, file.path(output_dir, paste0(Sys.Date(), "-metrics_all_models.csv")), row.names = FALSE)
  }
  
  invisible(TRUE)
}

################################################################################
# 7. Run one repeat folder
################################################################################

run_one_repeat_folder <- function(repeat_folder, config = CONFIG) {
  test_file <- find_one_file(repeat_folder, pattern = config$test_pattern,
                             recursive = FALSE, description = "testdata CSV")
  
  selected_train_files <- dir(repeat_folder, pattern = config$selected_train_pattern,
                              full.names = TRUE, recursive = TRUE)
  selected_train_files <- selected_train_files[grepl("selected_data_for_model", selected_train_files, fixed = TRUE)]
  selected_train_files <- selected_train_files[!grepl("model_prediction_results", selected_train_files, fixed = TRUE)]
  
  if (length(selected_train_files) == 0) {
    stop("No selected training CSV found under selected_data_for_model in folder: ", repeat_folder)
  }
  
  failed <- character(0)
  for (train_file in selected_train_files) {
    print(train_file)
    ok <- try_or_null(run_one_selected_train_file(train_file, test_file, config),
                      paste0("[FAILED train file] ", train_file))
    if (is.null(ok)) failed <- c(failed, train_file)
  }
  
  if (length(failed) > 0) message(length(failed), " selected training file(s) failed in ", repeat_folder)
  else message("Completed: ", repeat_folder)
  
  invisible(TRUE)
}

################################################################################
# 8. Run all repeat folders
################################################################################

failed_repeat_folders <- character(0)

for (repeat_folder in repeat_folders) {
  ok <- try_or_null(run_one_repeat_folder(repeat_folder, CONFIG),
                    paste0("[FAILED repeat folder] ", repeat_folder))
  if (is.null(ok)) failed_repeat_folders <- c(failed_repeat_folders, repeat_folder)
}

if (length(failed_repeat_folders) > 0) {
  message("\n", length(failed_repeat_folders), " repeat folder(s) failed:")
  message(paste(" -", failed_repeat_folders, collapse = "\n"))
} else {
  message("\nAll repeat folders completed.")
}

################################################################################
# 9. Aggregate every repeat's metrics into one top-level summary

################################################################################
#cribe the feature-selection setting.
extract_path_context <- function(metrics_file, root_dir = dir_path) {
  # Normalise both to forward slashes and drop any trailing slash on the root.
  mf   <- gsub("\\\\", "/", metrics_file)        # backslashes -> forward slashes
  root <- gsub("\\\\", "/", root_dir)
  root <- sub("/+$", "", root)                   # remove trailing slash(es)
  
  # Strip the root prefix to get a path relative to dir_path, using literal
  # (non-regex) string matching so special characters in the path are safe.
  if (startsWith(mf, root)) {
    rel <- substring(mf, nchar(root) + 1)        # drop the root prefix
  } else {
    rel <- mf                                    # unexpected layout; use as-is
  }
  
  # Split and drop empty pieces (handles any accidental double slashes).
  parts <- strsplit(rel, "/", fixed = TRUE)[[1]]
  parts <- parts[parts != ""]
  
  mpr_idx <- which(parts == "model_prediction_results")
  
  # Need at least: <repeat>/.../model_prediction_results/...
  if (length(mpr_idx) == 0 || mpr_idx[1] < 2) {
    return(list(repeat_folder = NA_character_, selection_dir = NA_character_))
  }
  mpr_idx <- mpr_idx[1]
  
  # First component = repeat folder; everything up to (but excluding)
  # model_prediction_results = the feature-selection sub-path.
  repeat_folder <- parts[1]
  selection_dir <- if (mpr_idx >= 3) {
    paste(parts[2:(mpr_idx - 1)], collapse = "/")
  } else {
    NA_character_
  }
  
  list(repeat_folder = repeat_folder, selection_dir = selection_dir)
}

# Recursively find every per-folder summary file produced in section 6.
metrics_files <- dir(
  dir_path,
  pattern = "metrics_all_models\\.csv$",
  full.names = TRUE,
  recursive = TRUE
)

if (length(metrics_files) == 0) {
  # Nothing to aggregate (e.g. every repeat failed). Warn but don't error out.
  message("\nNo per-folder metrics_all_models.csv found; skipping top-level summary.")
} else {
  # Read each file, prepend the context columns, and collect into a list.
  summary_rows <- list()
  
  for (f in metrics_files) {
    # Read one folder's summary; if a single file is unreadable (e.g. OneDrive
    # hiccup), skip it instead of aborting the whole aggregation.
    one <- try_or_null(
      read.csv(f, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE),
      paste0("[FAILED reading summary] ", f)
    )
    if (is.null(one) || nrow(one) == 0) next
    
    # Work out which repeat / selection this file belongs to.
    ctx <- extract_path_context(f, dir_path)
    
    # Prepend the two context columns so the combined table is self-contained:
    # you can tell every row's origin without looking at the folder path.
    one <- cbind(
      data.frame(
        repeat_folder = ctx$repeat_folder,
        selection_dir = ctx$selection_dir,
        stringsAsFactors = FALSE
      ),
      one
    )
    
    summary_rows[[f]] <- one
  }
  
  if (length(summary_rows) == 0) {
    message("\nAll per-folder summaries were empty or unreadable; no top-level summary written.")
  } else {
    # Stack all folders' rows into one data.frame. All per-folder files share the
    # same columns (they come from the same code), so rbind is safe here.
    combined <- do.call(rbind, summary_rows)
    rownames(combined) <- NULL
    
    # Write the single top-level summary next to the repeat folders.
    out_file <- file.path(dir_path, paste0(Sys.Date(), "-ALL_repeats_metrics_summary.csv"))
    write.csv(combined, out_file, row.names = FALSE)
    
    message("\nTop-level summary written: ", out_file,
            " (", nrow(combined), " rows from ", length(summary_rows), " folder(s)).")
  }
}
View(combined)
