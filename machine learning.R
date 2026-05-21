# ============================================================

rm(list = ls())

# Original install notes preserved once:
# install.packages(c("e1071", "randomForest", "ROCR", "pROC"))
# install.packages("mltools")
# install.packages("xgboost")
# install.packages("DMwR")
# install.packages("DMwR_0.4.1.tar.gz", repos = NULL, type = "source")

############################################ library packages
library(e1071)          # naivebayes & SVM
library(randomForest)
library(nnet)
library(ROCR)
library(pROC)
library(mltools)
library(ROSE)
library(splitstackshape)
library(xgboost)
# library(Matrix)
# library(DMwR)

########################################## prepare data path
dir_path_ori <- "D:\\2025\\20250808_wnt\\2_ML\\"
fs_dir <- c("auc_feature_selection", "fisher exact test", "xg_featue_selection", "RF_featue_selection")

loop <- 20 # Repeat 20 times, preserved from all original scripts.

# ============================================================
# Variant-specific information that was different across files.
# ============================================================

classic_model_configs <- list(
  list(
    source_file = "model_70-30-test.R",
    variant = "test",
    sampler = "none",
    seed_multiplier = 2020L,
    output_suffix = "test"
  ),
  list(
    source_file = "model_70-30-test -down.R",
    variant = "down",
    sampler = "under",
    seed_multiplier = 2022L,
    output_suffix = "down"
  ),
  list(
    source_file = "model_70-30-test -rose.R",
    variant = "ROSE",
    sampler = "ROSE",
    seed_multiplier = 2021L,
    output_suffix = "ROSE"
  ),
  list(
    source_file = "model_70-30-test -up.R",
    variant = "up",
    sampler = "over",
    seed_multiplier = 2020L,
    output_suffix = "up"
  )
)

xgboost_configs <- list(
  list(
    source_file = "xgboost_4_loop_k-test.R",
    variant = "test",
    sampler = "none",
    seed_multiplier = 2020L,
    output_folder = "xgboost_test",
    output_prefix = "xg_test_"
  ),
  list(
    source_file = "xgboost_4_loop_k-test -down.R",
    variant = "down",
    sampler = "under",
    seed_multiplier = 2020L,
    output_folder = "xgboost_down",
    output_prefix = "xg_down_"
  ),
  list(
    source_file = "xgboost_4_loop_k-test -rose.R",
    variant = "rose",
    sampler = "ROSE",
    seed_multiplier = 2020L,
    output_folder = "xgboost_rose",
    output_prefix = "xg_rose_"
  ),
  list(
    source_file = "xgboost_4_loop_k-test -up.R",
    variant = "up",
    sampler = "over",
    seed_multiplier = 2020L,
    output_folder = "xgboost_up",
    output_prefix = "xg_up_"
  )
)

# ============================================================
# Common helper functions. These replace repeated blocks from the 8 source files.
# ============================================================

reset_output_dir <- function(target_dir) {
  unlink(target_dir, recursive = TRUE) # delete file and all involved
  dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)
}

load_csv_objects <- function(dir_path_target) {
  dir_path_name <- dir(dir_path_target, pattern = "*.csv") ## input data
  object_file_list <- list()
  for (aa in 1:length(dir_path_name)) {
    dir_path_file <- paste0(dir_path_target, "\\", dir_path_name[aa])
    object_file_list[[aa]] <- read.csv(dir_path_file, header = TRUE, stringsAsFactors = FALSE)
    names(object_file_list)[aa] <- dir_path_name[aa]
  }
  object_file_list
}

prepare_model_data <- function(data, make_num_factor = TRUE) {
  colnames(data) <- c(colnames(data[1:(ncol(data) - 1)]), "num")
  data[, -c(1:2)] <- sapply(data[, -c(1:2)], as.numeric) ################################################### ADD__1
  if (make_num_factor) {
    data$num <- as.factor(data$num)
  }
  data
}

split_train_test_by_mapping_id <- function(data) {
  # Original alternative preserved as comment:
  # data_test_list <- sample(unique(data$Mapping.ID), size = floor(length(unique(data$Mapping.ID)) * 0.3), replace = FALSE)
  data_test_list <- stratified(data, "num", .3)$Mapping.ID
  testdata <- data[data$Mapping.ID %in% data_test_list, ]
  traindata <- data[!data$Mapping.ID %in% data_test_list, ]
  testdata$Structure_SMILES_2D.QSAR <- testdata$Mapping.ID <- NULL
  traindata$Structure_SMILES_2D.QSAR <- traindata$Mapping.ID <- NULL
  testdata <- unique(testdata)
  traindata <- unique(traindata)
  list(traindata = traindata, testdata = testdata)
}

apply_sampling <- function(traindata, sampler) {
  if (sampler == "none") {
    return(traindata)
  }
  if (sampler == "under") {
    return(ovun.sample(num ~ ., data = traindata, method = "under", seed = 1)$data)
  }
  if (sampler == "over") {
    return(ovun.sample(num ~ ., data = traindata, method = "over")$data)
  }
  if (sampler == "ROSE") {
    return(ROSE(num ~ ., data = traindata, seed = 1)$data)
  }
  stop(paste("Unknown sampler:", sampler))
}

calc_binary_metrics <- function(scores, actuals) {
  pred_auc <- prediction(scores, actuals)
  auc_value <- performance(pred_auc, "auc")@y.values[[1]]

  modelroc <- pROC::roc(actuals, as.numeric(scores))
  cutoff <- pROC::coords(modelroc, "best", ret = "threshold")
  predict_results <- ifelse(scores > as.numeric(cutoff[1]), "1", "0")
  freq_default <- table(predict_results, actuals)

  Sensitivity <- freq_default[1, 1] / sum(freq_default[, 1])
  Specificity <- freq_default[2, 2] / sum(freq_default[, 2])
  accuracy_value <- mean(actuals == predict_results)
  balanced_accuracy_value <- (Sensitivity + Specificity) / 2
  mcc_value <- mcc(as.numeric(as.character(actuals)), as.numeric(as.character(predict_results)))

  if (auc_value < 0.5) {
    auc_value <- 1 - auc_value
    accuracy_value <- 1 - accuracy_value
    balanced_accuracy_value <- 1 - balanced_accuracy_value
    mcc_value <- abs(mcc_value)
  }

  c(
    auc = auc_value,
    accuracy = accuracy_value,
    Balanced_accuracy = balanced_accuracy_value,
    mcc_result = mcc_value
  )
}

summarize_loop_metrics <- function(metric_matrix) {
  # Keep the same output layout as the original scripts:
  # mean AUC, sd AUC, mean accuracy, sd accuracy, mean balanced accuracy, sd balanced accuracy, mean MCC, sd MCC.
  data.frame(
    mean(metric_matrix[, "auc"][metric_matrix[, "auc"] != 0]),
    sd(metric_matrix[, "auc"][metric_matrix[, "auc"] != 0]),
    mean(metric_matrix[, "accuracy"][metric_matrix[, "accuracy"] != 0]),
    sd(metric_matrix[, "accuracy"][metric_matrix[, "accuracy"] != 0]),
    mean(metric_matrix[, "Balanced_accuracy"][metric_matrix[, "Balanced_accuracy"] != 0]),
    sd(metric_matrix[, "Balanced_accuracy"][metric_matrix[, "Balanced_accuracy"] != 0]),
    mean(metric_matrix[, "mcc_result"][metric_matrix[, "mcc_result"] != 0]),
    sd(metric_matrix[, "mcc_result"][metric_matrix[, "mcc_result"] != 0])
  )
}

# ============================================================
# Common classic-model process: Naive Bayes, SVM, Random Forest, NNET.
# This replaces model_70-30-test*.R repeated bodies.
# ============================================================

run_classic_models <- function(data, config) {
  data <- prepare_model_data(data, make_num_factor = TRUE)
  n_col_newdata <- ncol(data) - 2

  nb_metrics <- matrix(0, nrow = loop, ncol = 4, dimnames = list(NULL, c("auc", "accuracy", "Balanced_accuracy", "mcc_result")))
  svm_metrics <- matrix(0, nrow = loop, ncol = 4, dimnames = list(NULL, c("auc", "accuracy", "Balanced_accuracy", "mcc_result")))
  rf_metrics <- matrix(0, nrow = loop, ncol = 4, dimnames = list(NULL, c("auc", "accuracy", "Balanced_accuracy", "mcc_result")))
  nnet_metrics <- matrix(0, nrow = loop, ncol = 4, dimnames = list(NULL, c("auc", "accuracy", "Balanced_accuracy", "mcc_result")))

  for (j in 1:loop) {
    tryCatch({
      set.seed(j * config$seed_multiplier)
      split_data <- split_train_test_by_mapping_id(data)
      traindata <- apply_sampling(split_data$traindata, config$sampler)
      testdata <- split_data$testdata

      ############################################# naivebayes_model
      fit_nb <- naiveBayes(num ~ ., data = traindata, laplace = 1)
      pre_auc_nb <- predict(fit_nb, testdata[, -n_col_newdata], type = "raw")
      nb_metrics[j, ] <- calc_binary_metrics(pre_auc_nb[, 2], testdata[, n_col_newdata])

      ############################################## svm
      fit_svm <- svm(num ~ ., data = traindata, scale = FALSE)
      pre_auc_svm <- predict(fit_svm, testdata[, -n_col_newdata], decision.values = TRUE)
      pre_auc_svm_d <- attr(pre_auc_svm, "decision.values")
      svm_metrics[j, ] <- calc_binary_metrics(pre_auc_svm_d, testdata[, n_col_newdata])

      ############################################## rf
      fit_rf <- randomForest(num ~ ., data = traindata)
      pre_auc_rf <- predict(fit_rf, testdata[, -n_col_newdata], type = "prob")
      rf_metrics[j, ] <- calc_binary_metrics(pre_auc_rf[, 2], testdata[, n_col_newdata])

      ############################################## nnet
      fit_nnet <- nnet(num ~ ., data = traindata, size = 10, MaxNWts = 100000)
      pre_auc_nnet <- predict(fit_nnet, testdata[, -n_col_newdata], type = "raw")
      nnet_metrics[j, ] <- calc_binary_metrics(pre_auc_nnet[, 1], testdata[, n_col_newdata])
    }, error = function(e) {
      cat("ERROR :", conditionMessage(e), "\n")
    })
  }

  list(
    nb = summarize_loop_metrics(nb_metrics),
    svm = summarize_loop_metrics(svm_metrics),
    rf = summarize_loop_metrics(rf_metrics),
    nnet = summarize_loop_metrics(nnet_metrics)
  )
}

run_classic_variant_for_target <- function(dir_path_target, config) {
  model_names <- c("nb", "svm", "rf", "nnet")
  for (model_name in model_names) {
    reset_output_dir(paste0(dir_path_target, "\\", model_name, "_", config$output_suffix))
  }

  object_file_list <- load_csv_objects(dir_path_target)
  names(object_file_list)
  colnames(object_file_list[[1]])

  for (mm in 1:length(object_file_list)) {
    data <- object_file_list[[mm]] # check point-1
    outputs <- run_classic_models(data, config)
    for (model_name in model_names) {
      folder_name <- paste0(model_name, "_", config$output_suffix)
      file_name_data <- paste0(dir_path_target, "\\", folder_name, "\\", folder_name, "_", names(object_file_list)[[mm]])
      write.csv(outputs[[model_name]], file_name_data)
    }
  }
}

# ============================================================
# Common XGBoost process. This replaces xgboost_4_loop_k-test*.R repeated bodies.
# ============================================================

run_xgboost_model <- function(data, config) {
  data <- prepare_model_data(data, make_num_factor = FALSE)
  n_col_newdata <- ncol(data) - 2

  xgboost_metrics <- matrix(0, nrow = loop, ncol = 4, dimnames = list(NULL, c("auc", "accuracy", "Balanced_accuracy", "mcc_result")))

  for (j in 1:loop) {
    tryCatch({
      set.seed(j * config$seed_multiplier)
      split_data <- split_train_test_by_mapping_id(data)
      traindata <- apply_sampling(split_data$traindata, config$sampler)
      testdata <- split_data$testdata

      train_label <- traindata[, n_col_newdata]
      train_matrix <- xgb.DMatrix(data = as.matrix(traindata[, -n_col_newdata]), label = train_label)
      val_label <- testdata[, n_col_newdata]
      val_matrix <- xgb.DMatrix(data = as.matrix(testdata[, -n_col_newdata]), label = val_label)

      xgb_params <- list(
        objective = "binary:logistic",
        eval_metric = "error",
        max_depth = 3,
        eta = 0.01,
        gammma = 1, # spelling preserved from original scripts
        colsample_bytree = 0.5,
        min_child_weight = 1
      )

      bst_model <- xgb.train(params = xgb_params, data = train_matrix, nrounds = 1000)
      pre_auc_xgboost <- predict(bst_model, newdata = val_matrix, type = "raw")
      xgboost_metrics[j, ] <- calc_binary_metrics(pre_auc_xgboost, testdata[, n_col_newdata])
    }, error = function(e) {
      cat("ERROR :", conditionMessage(e), "\n")
    })
  }

  summarize_loop_metrics(xgboost_metrics)
}

run_xgboost_variant_for_target <- function(dir_path_target, config) {
  reset_output_dir(paste0(dir_path_target, "\\", config$output_folder))

  object_file_list <- load_csv_objects(dir_path_target)
  for (mm in 1:length(object_file_list)) {
    data <- object_file_list[[mm]] # check point-1
    output_xgboost <- run_xgboost_model(data, config)
    file_name_data_xgboost <- paste0(dir_path_target, "\\", config$output_folder, "\\", config$output_prefix, names(object_file_list)[[mm]])
    write.csv(output_xgboost, file_name_data_xgboost)
  }
}

# ============================================================
# Main workflow.
# Set these flags to FALSE if you want to run only one model family.
# ============================================================

run_classic_model_family <- TRUE
run_xgboost_family <- TRUE

for (ii in 1:length(fs_dir)) {
  dir_path <- paste0(dir_path_ori, fs_dir[ii], "\\selected_data_for_model")
  dir_path_target <- list.dirs(dir_path, recursive = FALSE)
  dir_path_target

  for (nn in 1:length(dir_path_target)) {
    tryCatch({
      if (run_classic_model_family) {
        for (config in classic_model_configs) {
          run_classic_variant_for_target(dir_path_target[nn], config)
        }
      }

      if (run_xgboost_family) {
        for (config in xgboost_configs) {
          run_xgboost_variant_for_target(dir_path_target[nn], config)
        }
      }
    }, error = function(e) {
      cat("ERROR :", conditionMessage(e), "\n")
    })
  }
}

# feature importance notes preserved once:
# imp = xgb.importance(colnames(train_matrix), model = bst_model)
# xgb.plot.importance(imp)
# sink()
