rm(list = ls())

################################################################################
# 0. Configuration
################################################################################

CONFIG <- list(
  dir_path = "D:\\2025\\20250808_wnt\\2_ML\\",
  csv_pattern = "\\.csv$",

  fisher = list(
    output_dir = "fisher exact test/",
    stat_dir = "pvalue_features_stat/",
    selected_dir = "selected_data_for_model/",
    stat_cutoff = 0.05,
    thresholds = seq(0.01, 0.05, by = 0.01),
    subdir_prefix = "pvalue_"
  ),

  auc = list(
    output_dir = "auc_feature_selection/",
    stat_dir = "AUC_features_stat/",
    plot_dir = "plot_stat/",
    selected_dir = "selected_data_for_model/",
    stat_cutoff = 0.55,
    thresholds = seq(0.52, 0.60, by = 0.02),
    subdir_prefix = "pvalue_"
  ),

  xgboost = list(
    output_dir = "xg_featue_selection/",
    selected_dir = "selected_data_for_model/",
    top_n = seq(20, 100, by = 20),
    subdir_prefix = "xg_",
    params = list(
      objective = "binary:logistic",
      eval_metric = "error",
      max_depth = 3,
      eta = 0.01,
      gammma = 1,
      colsample_bytree = 0.5,
      min_child_weight = 1
    ),
    nrounds = 1000
  ),

  rf = list(
    output_dir = "RF_featue_selection/",
    selected_dir = "selected_data_for_model/",
    top_n = seq(20, 100, by = 20),
    subdir_prefix = "rf_"
  )
)

RUN_METHODS <- c("fisher", "auc", "xgboost", "rf")

################################################################################
# 1. Shared helpers retained once from all source scripts
################################################################################

safe_dir_create <- function(path) {
  if (!dir.exists(path)) {
    dir.create(path, recursive = TRUE, showWarnings = FALSE)
  }
  invisible(path)
}

safe_print_colnames <- function(data, from, to) {
  if (ncol(data) < from) {
    return(invisible(NULL))
  }
  idx <- from:min(to, ncol(data))
  print(colnames(data)[idx])
  invisible(NULL)
}

load_csv_files <- function(dir_path, csv_pattern = "\\.csv$", verbose = TRUE) {
  dir_path_name <- dir(dir_path, pattern = csv_pattern)
  if (length(dir_path_name) == 0) {
    stop("No CSV files found in dir_path: ", dir_path)
  }

  object_file_list <- list()

  for (i in seq_along(dir_path_name)) {
    dir_path_file <- paste0(dir_path, dir_path_name[i])
    object_file_list[[i]] <- read.csv(dir_path_file, header = TRUE, stringsAsFactors = FALSE)

    if (verbose) {
      print(dim(object_file_list[[i]]))
      safe_print_colnames(object_file_list[[i]], 1, 10)
      safe_print_colnames(object_file_list[[i]], 1020, 1027)
      if ("endpoint" %in% colnames(object_file_list[[i]])) {
        print(table(object_file_list[[i]]$endpoint))
      }
    }

    # Original scripts kept columns 1:2 as identifiers and converted all later
    # columns, including the endpoint/target column, to numeric.
    object_file_list[[i]][, -c(1:2)] <- sapply(object_file_list[[i]][, -c(1:2)], as.numeric)

    if (verbose) {
      print(colnames(object_file_list[[i]])[ncol(object_file_list[[i]])])
      print(dim(object_file_list[[i]]))
    }

    colnames(object_file_list[[i]])[ncol(object_file_list[[i]])] <- dir_path_name[i]
    names(object_file_list)[i] <- dir_path_name[i]
  }

  list(dir_path_name = dir_path_name, object_file_list = object_file_list)
}

subset_and_write_selected_data <- function(object_file_list,
                                           selected_features,
                                           output_path,
                                           target_name = NULL) {
  if (is.null(selected_features) || length(selected_features) == 0) {
    return(invisible(FALSE))
  }

  keep_cols <- c(
    colnames(object_file_list)[1:2],
    selected_features,
    colnames(object_file_list)[ncol(object_file_list)]
  )
  keep_cols <- keep_cols[keep_cols %in% colnames(object_file_list)]

  output_data <- object_file_list[, keep_cols, drop = FALSE]
  if (!is.null(target_name) && ncol(output_data) > 0) {
    colnames(output_data)[ncol(output_data)] <- target_name
  }

  write.csv(output_data, output_path, row.names = FALSE)
  invisible(TRUE)
}

################################################################################
# 2. Method 1: Fisher exact test feature selection
#    From fisher_exact_ttest_FS_1.R
################################################################################

run_fisher_fs <- function(config = CONFIG) {
  cat("\n========== Running Fisher exact test feature selection ==========" , "\n")
  loaded <- load_csv_files(config$dir_path, config$csv_pattern)
  dir_path_name <- loaded$dir_path_name
  object_file_list <- loaded$object_file_list

  output_file <- paste0(config$dir_path, config$fisher$output_dir)
  safe_dir_create(output_file)

  object_file_lisT_select <- list()

  for (i in seq_along(dir_path_name)) {
    a1 <- object_file_list[[i]]
    a <- a1[, c(grep("^V[[:digit:]]", colnames(a1)), ncol(a1)), drop = FALSE]

    b <- NULL
    for (mm in seq_len(ncol(a) - 1)) {
      print(mm)
      tox_taget <- a[which(a[, mm] == 1 & a[, ncol(a)] == 1), , drop = FALSE]
      tox_non_taget <- a[which(a[, mm] == 0 & a[, ncol(a)] == 1), , drop = FALSE]
      non_tox_target <- a[which(a[, mm] == 1 & a[, ncol(a)] == 0), , drop = FALSE]
      non_tox_non_taget <- a[which(a[, mm] == 0 & a[, ncol(a)] == 0), , drop = FALSE]

      tox_target <- matrix(
        c(nrow(tox_taget), nrow(non_tox_target), nrow(tox_non_taget), nrow(non_tox_non_taget)),
        nrow = 2,
        dimnames = list(tox = c("tox", "non_tox"), gene = c("tar", "nontar"))
      )
      tox_target_pvalue <- fisher.test(tox_target, alternative = "two.sided")$p.value
      b <- c(b, tox_target_pvalue)
    }

    e <- data.frame(b, colnames(a)[seq_len(ncol(a) - 1)])
    colnames(e) <- c("pvalue", colnames(a)[ncol(a)])
    f <- e[order(e$pvalue), ]
    object_file_lisT_select[[i]] <- f

    write.csv(f, paste0(output_file, Sys.Date(), "-", colnames(a)[ncol(a)], ".csv"), row.names = FALSE)
  }

  safe_dir_create(paste0(output_file, config$fisher$stat_dir))
  for (i in seq_along(dir_path_name)) {
    a_1 <- object_file_lisT_select[[i]][object_file_lisT_select[[i]]$pvalue <= config$fisher$stat_cutoff, ]
    out_file_pvalue <- paste0(
      output_file, config$fisher$stat_dir,
      Sys.Date(), "-", colnames(object_file_lisT_select[[i]])[2], "-", config$fisher$stat_cutoff, ".csv"
    )
    write.csv(a_1, out_file_pvalue, row.names = FALSE)
  }

  dir_path_target <- paste0(output_file, config$fisher$selected_dir)
  safe_dir_create(dir_path_target)
  for (threshold in config$fisher$thresholds) {
    safe_dir_create(paste0(dir_path_target, config$fisher$subdir_prefix, threshold))
  }

  for (j in seq_along(dir_path_name)) {
    for (threshold in config$fisher$thresholds) {
      category_assay <- as.character(
        object_file_lisT_select[[j]][object_file_lisT_select[[j]]$pvalue <= threshold, 2]
      )
      output_path <- paste0(
        dir_path_target, config$fisher$subdir_prefix, threshold, "/", Sys.Date(), "-",
        colnames(object_file_list[[j]])[ncol(object_file_list[[j]])], "-", threshold, ".csv"
      )
      subset_and_write_selected_data(object_file_list[[j]], category_assay, output_path)
    }
  }

  invisible(list(dir_path_name = dir_path_name, object_file_list = object_file_list, selected = object_file_lisT_select))
}

################################################################################
# 3. Method 2: ROC/AUC feature selection
#    From AUC_ROC_FS_2.R
################################################################################

getROC_AUC <- function(probs, true_Y) {
  probsSort <- sort(probs, decreasing = TRUE, index.return = TRUE)
  idx <- unlist(probsSort$ix)
  roc_y <- true_Y[idx]
  stack_x <- cumsum(roc_y == 1) / sum(roc_y == 1)
  stack_y <- cumsum(roc_y == 0) / sum(roc_y == 0)
  auc <- sum((stack_x[2:length(roc_y)] - stack_x[1:length(roc_y) - 1]) * stack_y[2:length(roc_y)])
  list(stack_x = stack_x, stack_y = stack_y, auc = auc)
}

run_auc_fs <- function(config = CONFIG) {
  cat("\n========== Running ROC/AUC feature selection ==========" , "\n")
  if (!requireNamespace("ROCR", quietly = TRUE)) {
    stop("Package 'ROCR' is required for AUC feature selection.")
  }

  loaded <- load_csv_files(config$dir_path, config$csv_pattern)
  dir_path_name <- loaded$dir_path_name
  object_file_list <- loaded$object_file_list

  output_file <- paste0(config$dir_path, config$auc$output_dir)
  safe_dir_create(output_file)

  object_file_lisT_select <- list()

  for (i in seq_along(dir_path_name)) {
    true_Y <- object_file_list[[i]][, ncol(object_file_list[[i]])]
    auc_all_1 <- auc_all_2 <- NULL

    for (j in 3:(ncol(object_file_list[[i]]) - 1)) {
      probs <- object_file_list[[i]][, j]

      aList <- getROC_AUC(probs, true_Y)
      auc_1 <- unlist(aList$auc)
      if (auc_1 < 0.5) auc_1 <- 1 - auc_1

      pred_auc <- ROCR::prediction(probs, true_Y)
      auc_2 <- ROCR::performance(pred_auc, "auc")@y.values[[1]]
      if (auc_2 < 0.5) auc_2 <- 1 - auc_2

      auc_all_1 <- c(auc_all_1, auc_1)
      auc_all_2 <- c(auc_all_2, auc_2)
    }

    all_auc_merge <- data.frame(
      auc_all_1,
      auc_all_2,
      colnames(object_file_list[[i]])[3:(ncol(object_file_list[[i]]) - 1)]
    )
    colnames(all_auc_merge) <- c("auc_without_proc", "auc_with_proc", names(object_file_list)[i])
    object_file_lisT_select[[i]] <- all_auc_merge[order(all_auc_merge$auc_with_proc), ]

    write.csv(
      object_file_lisT_select[[i]],
      paste0(output_file, Sys.Date(), "-", names(object_file_list)[i], "_AUC.csv"),
      row.names = TRUE
    )
  }

  safe_dir_create(paste0(output_file, config$auc$stat_dir))
  for (i in seq_along(dir_path_name)) {
    a_1 <- object_file_lisT_select[[i]][object_file_lisT_select[[i]]$auc_with_proc >= config$auc$stat_cutoff, ]
    out_file_pvalue <- paste0(
      output_file, config$auc$stat_dir,
      Sys.Date(), "-", colnames(object_file_lisT_select[[i]])[3], "-", config$auc$stat_cutoff, ".csv"
    )
    write.csv(a_1, out_file_pvalue, row.names = FALSE)
  }

  safe_dir_create(paste0(output_file, config$auc$plot_dir))

  dir_path_target <- paste0(output_file, config$auc$selected_dir)
  safe_dir_create(dir_path_target)
  for (threshold in config$auc$thresholds) {
    safe_dir_create(paste0(dir_path_target, config$auc$subdir_prefix, threshold))
  }

  for (j in seq_along(object_file_lisT_select)) {
    for (threshold in config$auc$thresholds) {
      category_assay <- as.character(
        object_file_lisT_select[[j]][object_file_lisT_select[[j]]$auc_with_proc >= threshold, 3]
      )
      output_path <- paste0(
        dir_path_target, config$auc$subdir_prefix, threshold, "/", Sys.Date(), "-",
        colnames(object_file_lisT_select[[j]])[3], "-", threshold, ".csv"
      )
      subset_and_write_selected_data(
        object_file_list[[j]],
        category_assay,
        output_path,
        target_name = colnames(object_file_lisT_select[[j]])[3]
      )
    }
  }

  invisible(list(dir_path_name = dir_path_name, object_file_list = object_file_list, selected = object_file_lisT_select))
}

################################################################################
# 4. Method 3: XGBoost feature importance selection
#    From XGBOOST_FS_3.R
################################################################################

run_xgboost_fs <- function(config = CONFIG) {
  cat("\n========== Running XGBoost feature importance selection ==========" , "\n")
  if (!requireNamespace("xgboost", quietly = TRUE)) {
    stop("Package 'xgboost' is required for XGBoost feature selection.")
  }

  loaded <- load_csv_files(config$dir_path, config$csv_pattern)
  dir_path_name <- loaded$dir_path_name
  object_file_list <- loaded$object_file_list

  output_root <- paste0(config$dir_path, config$xgboost$output_dir)
  safe_dir_create(output_root)
  object_file_lisT_select <- list()

  for (mm in seq_along(object_file_list)) {
    data <- object_file_list[[mm]]
    colnames(data) <- c(colnames(data[1:(ncol(data) - 1)]), "num")

    n_col_newdata <- ncol(data)
    train_label <- data[, n_col_newdata]
    train_data <- as.matrix(data[, -c(1, 2, n_col_newdata)])
    train_matrix <- xgboost::xgb.DMatrix(data = train_data, label = train_label)

    bst_model <- xgboost::xgb.train(
      params = config$xgboost$params,
      data = train_matrix,
      nrounds = config$xgboost$nrounds
    )

    select_features <- xgboost::xgb.importance(colnames(train_data), model = bst_model)
    f <- select_features[order(select_features$Gain, decreasing = TRUE), ]
    object_file_lisT_select[[mm]] <- f

    write.table(
      f,
      paste0(output_root, names(object_file_list)[[mm]]),
      row.names = TRUE,
      sep = "\t"
    )
  }

  dir_path_target <- paste0(output_root, config$xgboost$selected_dir)
  safe_dir_create(dir_path_target)
  for (top_n in config$xgboost$top_n) {
    safe_dir_create(paste0(dir_path_target, config$xgboost$subdir_prefix, top_n))
  }

  for (j in seq_along(dir_path_name)) {
    for (top_n in config$xgboost$top_n) {
      object_file_lisT_select[[j]] <- object_file_lisT_select[[j]][order(object_file_lisT_select[[j]]$Gain, decreasing = TRUE), ]
      category_assay <- object_file_lisT_select[[j]]$Feature[1:top_n]
      output_path <- paste0(
        dir_path_target, config$xgboost$subdir_prefix, top_n, "/", Sys.Date(), "-",
        colnames(object_file_list[[j]])[ncol(object_file_list[[j]])], "-", top_n, ".csv"
      )
      subset_and_write_selected_data(object_file_list[[j]], category_assay, output_path)
    }
  }

  invisible(list(dir_path_name = dir_path_name, object_file_list = object_file_list, selected = object_file_lisT_select))
}

################################################################################
# 5. Method 4: Random Forest feature importance selection
#    From RF_FS_4.R
################################################################################

run_rf_fs <- function(config = CONFIG) {
  cat("\n========== Running Random Forest feature importance selection ==========" , "\n")
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Package 'randomForest' is required for RF feature selection.")
  }

  loaded <- load_csv_files(config$dir_path, config$csv_pattern)
  dir_path_name <- loaded$dir_path_name
  object_file_list <- loaded$object_file_list

  output_root <- paste0(config$dir_path, config$rf$output_dir)
  safe_dir_create(output_root)
  object_file_lisT_select <- list()

  for (mm in seq_along(object_file_list)) {
    print(mm)
    data <- object_file_list[[mm]]
    colnames(data) <- c(colnames(data[1:(ncol(data) - 1)]), "num")
    data$num <- as.factor(data$num)

    fit_rf <- randomForest::randomForest(num ~ ., data = data[, -c(1:2)], importance = TRUE)
    select_features <- randomForest::importance(fit_rf)
    e <- data.frame(select_features)
    f <- e[order(e$MeanDecreaseGini, decreasing = TRUE), ]
    object_file_lisT_select[[mm]] <- f

    write.table(
      f,
      paste0(output_root, names(object_file_list)[[mm]]),
      row.names = TRUE,
      sep = "\t"
    )

    # Original RF script ended with varImpPlot(fit_rf). Uncomment to plot each model.
    # randomForest::varImpPlot(fit_rf)
  }

  dir_path_target <- paste0(output_root, config$rf$selected_dir)
  safe_dir_create(dir_path_target)
  for (top_n in config$rf$top_n) {
    safe_dir_create(paste0(dir_path_target, config$rf$subdir_prefix, top_n))
  }

  for (j in seq_along(dir_path_name)) {
    for (top_n in config$rf$top_n) {
      object_file_lisT_select[[j]] <- object_file_lisT_select[[j]][order(object_file_lisT_select[[j]]$MeanDecreaseGini, decreasing = TRUE), ]
      object_file_lisT_select[[j]]$Feature <- rownames(object_file_lisT_select[[j]])
      category_assay <- object_file_lisT_select[[j]]$Feature[1:top_n]
      output_path <- paste0(
        dir_path_target, config$rf$subdir_prefix, top_n, "/", Sys.Date(), "-",
        colnames(object_file_list[[j]])[ncol(object_file_list[[j]])], "-", top_n, ".csv"
      )
      subset_and_write_selected_data(object_file_list[[j]], category_assay, output_path)
    }
  }

  invisible(list(dir_path_name = dir_path_name, object_file_list = object_file_list, selected = object_file_lisT_select))
}

################################################################################
# 6. Run selected modules
################################################################################

run_all_feature_selection <- function(methods = RUN_METHODS, config = CONFIG) {
  results <- list()
  if ("fisher" %in% methods) results$fisher <- run_fisher_fs(config)
  if ("auc" %in% methods) results$auc <- run_auc_fs(config)
  if ("xgboost" %in% methods) results$xgboost <- run_xgboost_fs(config)
  if ("rf" %in% methods) results$rf <- run_rf_fs(config)
  invisible(results)
}

# Default behavior matches the four source scripts as a single sequential workflow.
# Edit RUN_METHODS above if you only want a subset, for example:
# RUN_METHODS <- c("fisher", "auc")
results <- run_all_feature_selection(RUN_METHODS, CONFIG)
