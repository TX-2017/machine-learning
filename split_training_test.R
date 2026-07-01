rm(list = ls())
library(splitstackshape)
###############################input data 
dir_path <- "C:\\Users\\xut2\\OneDrive - National Institutes of Health\\Desktop\\2026\\2026_zoe\\1_raw_data_tox21_toxprint_ecfp4\\"
files_list <- dir(dir_path, pattern = "*.csv",full.names = T, recursive = F)
files_list
#####################################tox21
for (i in 1:length(files_list)) {
  i = 1
  file_split_folder <- sub("\\.csv$", "", files_list[i])
  dir.create(file_split_folder)
  repeat_folders <- paste0(file_split_folder, "\\repeat_", 1:20)
  lapply(repeat_folders, dir.create, recursive = TRUE, showWarnings = FALSE)
  df1 <- read.csv(files_list[i], header = T, stringsAsFactors = F)
  print(colnames(df1)[ncol(df1)]) #[1] "neurotoxicity"
  #str(df1)
  colnames(df1)[ncol(df1)] <- "endpoint"
  df1[,-1] <- sapply(df1[,-1], as.numeric)
  dim(df1) #[1] 1739 1026
  df1 <- unique(df1)
  ###########split 20 times
  test_id_list <- list()
  for (k in 1:length(repeat_folders)) {
    print(k)
    set.seed(k)
    test_id <- stratified(df1, "endpoint", .3)$Mapping.ID
    test_id_list[[k]] <- sort(unique(test_id))
    test_data <- df1[df1$Mapping.ID %in% test_id,]
    train_data <- df1[!df1$Mapping.ID %in% test_id,]
    #dim(test_data);dim(train_data) #[1] 3840   36  [1] 8992   36
    #unique(test_data$Mapping.ID) %in% unique(train_data$Mapping.ID)
    #table(test_data$endpoint); table(train_data$endpoint)
    print(table(test_data$endpoint))
    print(table(train_data$endpoint))
    print(length(intersect(unique(test_data$Mapping.ID), unique(train_data$Mapping.ID))))
    test_data <- unique(test_data)
    train_data <- unique(train_data)
    write.csv(test_data,  paste0(repeat_folders[k], "\\", Sys.Date(), "_repeat_", k, "-testdata.csv"), row.names = FALSE)
    write.csv(train_data,  paste0(repeat_folders[k], "\\", Sys.Date(), "_repeat_", k,"-traindata.csv"), row.names = FALSE)
  }
  print(length(unique(sapply(test_id_list, paste, collapse = "_"))))
}


