library(e1071)
library(randomForest) 
library(nnet)
library(xgboost)
library(ROCR)
library(pROC)
library(mltools) 
library(DMwR)
fs_dir <- c("auc_fs","fisher exact test_fs","rf_fs","xgboost_fs")
for (ii in 1:length(fs_dir)) {
  dir_path <- paste0(dir_path_ori,fs_dir[ii])
  dir_path_target <- list.dirs(dir_path,recursive = F)
  dir_path_target
for (nn in 1:length(dir_path_target)) {
    dir_path_name <- dir(dir_path_target[nn],pattern = "*.csv") 
    object_file_list <- list()
    for (aa in 1:length(dir_path_name)) {
      dir_path_file <- paste0(dir_path_target[nn],dir_path_name[aa]) 
      object_file_list[[aa]] <- read.csv(dir_path_file,header = T,stringsAsFactors = F)
      names(object_file_list)[aa] <- dir_path_name[aa]
    }
    for (mm in 1:length(object_file_list)){
      data <- object_file_list[[mm]] 
      colnames(data) <- c(colnames(data[1:(ncol(data)-1)]),"num")
      data$num <- as.factor(data$num)
      n_col_newdata <- ncol(data)
      loop <- 20
      auc_nb <- Balanced_accuracy_nb <- mcc_result_nb  <- numeric(loop)
      auc_svm <- Balanced_accuracy_svm <-  mcc_result_svm <- numeric(loop)
      auc_rf <- Balanced_accuracy_rf <- mcc_result_rf <- numeric(loop)
      auc_nnet <- Balanced_accuracy_nnet <- mcc_result_nnet <- numeric(loop)
      auc_xgboost_mean <- Balanced_accuracy_xgboost_mean <- mcc_result_xgboost_mean <- numeric(loop)
      for (j in 1:loop) {
        set.seed(j*2020)
        data_random <- data[sample(nrow(data)),]
        testdata <- data[sample(nrow(data), size = floor(nrow(data)*0.3),replace = F), ]
        traindata <- data[-match(row.names(testdata),row.names(data)), ]
        traindata <- ovun.sample(num ~ ., data = traindata,  method = "over")$data
        ##############################################nb
        fit_nb <- naiveBayes(num ~ .,data = traindata,laplace = 1)
        pre_auc_nb <- predict(fit_nb, testdata[,-n_col_newdata], type = 'raw')
        pred_auc_nb <- prediction(pre_auc_nb[,2],testdata[,n_col_newdata])
        auc_nb[j] <- performance(pred_auc_nb,"auc")@y.values
        modelroc_nb <- roc(testdata[,n_col_newdata],as.numeric(pre_auc_nb[,2]))
        cutoff_nb <- coords(modelroc_nb, "best", ret = "threshold")
        predict.results <- ifelse(pre_auc_nb[,2] > cutoff_nb,"1","0")
        freq_default_nb <- table(predict.results, testdata[,n_col_newdata])
        Sensitivity_nb <- freq_default_nb[1,1]/sum(freq_default_nb[,1])
        Specificity_nb <- freq_default_nb[2,2]/sum(freq_default_nb[,2])
        accuracy_nb[j] <- mean(testdata[,n_col_newdata] == predict.results)
        Balanced_accuracy_nb[j] <- (Sensitivity_nb+Specificity_nb)/2
        preds_nb <- as.numeric(as.character(predict.results))  
        actuals_nb <- as.numeric(as.character(testdata$num))
        mcc_result_nb[j] <- mcc(actuals_nb,preds_nb)
        ##############################################svm
        fit_svm <- svm(num ~ .,data = traindata, scale = F)
        pre_auc_svm <- predict(fit_svm, testdata[,-n_col_newdata], decision.values = TRUE)
        pre_auc_svm_d <- attr(pre_auc_svm,"decision.values")
        pred_auc_svm <- prediction(pre_auc_svm_d,testdata[,n_col_newdata])
        auc_svm[j] <- performance(pred_auc_svm,"auc")@y.values
        modelroc_svm <- roc(testdata[,n_col_newdata],as.numeric(pre_auc_svm_d))
        modelroc_svm <- coords(modelroc_svm, "best", ret = "threshold")
        predict.results_svm <- ifelse(pre_auc_svm_d > modelroc_svm,"1","0")
        freq_default_svm <- table(predict.results_svm,testdata[,n_col_newdata])
        Sensitivity_svm <- freq_default_svm[1,1]/sum(freq_default_svm[,1])
        Specificity_svm <- freq_default_svm[2,2]/sum(freq_default_svm[,2])
        Balanced_accuracy_svm[j] <- (Sensitivity_svm+Specificity_svm)/2
        preds_svm <- as.numeric(as.character(predict.results_svm))  
        actuals_svm <- as.numeric(as.character(testdata$num))
        mcc_result_svm[j] <- mcc(actuals_svm,preds_svm)
        ##############################################rf
        fit_rf <- randomForest(num ~ .,data = traindata)  
        pre_auc_rf <- predict(fit_rf, testdata[,-n_col_newdata], type = "prob")
        pred_auc_rf <- prediction(pre_auc_rf[,2],testdata[,n_col_newdata])
        auc_rf[j] <- performance(pred_auc_rf,"auc")@y.values
        modelroc_rf <- roc(testdata[,n_col_newdata],as.numeric(pre_auc_rf[,2]))
        cutoff_rf<- coords(modelroc_rf, "best", ret = "threshold")
        predict.results_rf <- ifelse(pre_auc_rf[,2] > cutoff_rf,"1","0")
        freq_default_rf <- table(predict.results_rf,testdata[,n_col_newdata])
        Sensitivity_rf <- freq_default_rf[1,1]/sum(freq_default_rf[,1])
        Specificity_rf <- freq_default_rf[2,2]/sum(freq_default_rf[,2])
        Balanced_accuracy_rf[j] <- (Sensitivity_rf+Specificity_rf)/2
        preds_rf <- as.numeric(as.character(predict.results_rf))  
        actuals_rf <- as.numeric(as.character(testdata$num))
        mcc_result_rf[j] <- mcc(actuals_rf,preds_rf)
        ##############################################nnet
        fit_nnet <- nnet(num ~ .,data = traindata,size=10,MaxNWts = 100000)
        pre_auc_nnet <- predict(fit_nnet, testdata[,-n_col_newdata], type = 'raw')
        pred_auc_nnet <- prediction(pre_auc_nnet[,1],testdata[,n_col_newdata])
        auc_nnet[j] <- performance(pred_auc_nnet,"auc")@y.values
        modelroc_nnet <- roc(testdata[,n_col_newdata],as.numeric(pre_auc_nnet[,1]))
        cutoff_nnet <- coords(modelroc_nnet, "best", ret = "threshold")
        predict.results_nnet <- ifelse(pre_auc_nnet[,1] > cutoff_nnet,"1","0")
        freq_default_nnet <- table(predict.results_nnet,testdata[,n_col_newdata])
        Sensitivity_nnet <- freq_default_nnet[1,1]/sum(freq_default_nnet[,1])
        Specificity_nnet <- freq_default_nnet[2,2]/sum(freq_default_nnet[,2])
        Balanced_accuracy_nnet[j] <- (Sensitivity_nnet+Specificity_nnet)/2
        preds_nnet <- as.numeric(as.character(predict.results_nnet))  
        actuals_nnet <- as.numeric(as.character(testdata$num))
        mcc_result_nnet[j] <- mcc(actuals_nnet,preds_nnet)
        ##############################################XGBOOST
        train_matrix = xgb.DMatrix(data = as.matrix(traindata[,-n_col_newdata]), label = train_label)
        val_label = testdata[,n_col_newdata] 
        val_matrix = xgb.DMatrix(data = as.matrix(testdata[,-n_col_newdata]), label = val_label)
        xgb_params = list(objective = "binary:logistic",eval_metric = "error",max_depth = 3,
                          eta = 0.01,gammma = 1, colsample_bytree = 0.5, min_child_weight = 1)
        bst_model = xgb.train(params = xgb_params, data = train_matrix,nrounds = 1000)
        pre_auc_xgboost <- predict(bst_model, newdata = val_matrix, type = 'raw')
        pred_auc_xgboost <- prediction(pre_auc_xgboost,testdata[,n_col_newdata])
        auc_xgboost_mean[j] <- performance(pred_auc_xgboost,"auc")@y.values
        modelroc_xgboost <- roc(testdata[,n_col_newdata],as.numeric(pre_auc_xgboost))
        cutoff_xgboost <- coords(modelroc_xgboost, "best", ret = "threshold")
        predict.results <- ifelse(pre_auc_xgboost > cutoff_xgboost,"1","0")
        freq_default_xgboost <- table(predict.results, testdata[,n_col_newdata])
        Sensitivity_xgboost <- freq_default_xgboost[1,1]/sum(freq_default_xgboost[,1])
        Specificity_xgboost <- freq_default_xgboost[2,2]/sum(freq_default_xgboost[,2])
        Balanced_accuracy_xgboost_mean[j] <- (Sensitivity_xgboost+Specificity_xgboost)/2
        preds_xgboost <- as.numeric(as.character(predict.results))  
        actuals_xgboost <- as.numeric(as.character(testdata[,n_col_newdata]))
        mcc_result_xgboost_mean[j] <- mcc(actuals_xgboost,preds_xgboost)
      }
      #####svm nb
      output_nb <- data.frame(mean(auc_nb), sd(auc_nb), mean(Balanced_accuracy_nb), 
                              sd(Balanced_accuracy_nb),mean(mcc_result_nb),sd(mcc_result_nb)) 
      file_name_data_nb <- paste0(dir_path_target[nn], names(object_file_list)[[mm]])
      write.csv(output_nb,file_name_data_nb)
      #####svm
      output_svm <- data.frame(mean(auc_svm),sd(auc_svm),mean(Balanced_accuracy_svm),sd(Balanced_accuracy_svm),
                               mean(mcc_result_svm), sd(mcc_result_svm))
      file_name_data_svm <- paste0(dir_path_target[nn],names(object_file_list)[[mm]])
      write.csv(output_svm,file_name_data_svm)
      #####rf
      output_rf <- data.frame(mean(auc_rf),mean(Balanced_accuracy_rf),mean(mcc_result_rf),
                              sd(auc_rf),sd(Balanced_accuracy_rf),sd(mcc_result_rf))
      file_name_data_rf <- paste0(dir_path_target[nn],names(object_file_list)[[mm]])
      write.csv(output_rf,file_name_data_rf)
      #####nnet
      output_nnet <- data.frame(mean(auc_nnet), mean(Balanced_accuracy_nnet), mean(mcc_result_nnet),
                                sd(auc_nnet),sd(Balanced_accuracy_nnet), sd(mcc_result_nnet))
      file_name_data_nnet <- paste0(dir_path_target[nn],names(object_file_list)[[mm]])
      write.csv(output_nnet,file_name_data_nnet) 
      #####XGBOOST
      output_xgboost <- data.frame(mean(auc_xgboost_mean),mean(Balanced_accuracy_xgboost_mean),mean(mcc_result_xgboost_mean),
                                   sd(auc_xgboost_mean),sd(Balanced_accuracy_xgboost_mean),sd(mcc_result_xgboost_mean))
      file_name_data_xgboost <- paste0(dir_path_target[nn],names(object_file_list)[[mm]])
      write.csv(output_xgboost,file_name_data_xgboost)
  }
    }
     }
