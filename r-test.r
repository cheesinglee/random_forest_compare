rm(list=ls())

matthews_phi <- function(matrix,i){
  tp = matrix[i,i]
  fp = colSums(matrix)[i] - tp
  fn = rowSums(matrix)[i] - tp
  tmp_matrix = matrix
  tmp_matrix[i,] = 0
  tmp_matrix[,i] = 0
  tn = sum(tmp_matrix)
  return((tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
}

compute_classification_metrics <- function(mat){
  accuracy = sum(diag(mat))/sum(mat)
  n_classes = nrow(mat)
  
  macro_precision = 0 
  macro_recall = 0  
  macro_phi = 0
  for (i in 1:n_classes){
    macro_recall = macro_recall + mat[i,i]/colSums(mat)[i]/n_classes
    macro_precision = macro_precision + mat[i,i]/rowSums(mat)[i]/n_classes
    macro_phi = macro_phi + matthews_phi(mat,i)/n_classes
  }
  macro_f1 = 2*macro_recall*macro_precision/(macro_recall+macro_precision)
  
  scores = c(accuracy,macro_precision,macro_recall,macro_f1,macro_phi)
  
  return(scores)
}

library(randomForest)
library(Metrics)
library(utils)
library(caret)

setwd('/home/cheesinglee/bigml/random_forest_compare/')
data_dir = 'csv-data/synth/classification/cat'
# data_dir = 'csv-data/test/'
nfolds = 10

listing = dir(data_dir,pattern='*.csv')
results_file = 'R_results_5.out'

CLASSIFICATION = TRUE

if (CLASSIFICATION == TRUE){
  write(c('Dataset','Rows','Fields','Train Time','Predict Time','Train Mem','Predict Mem','accuracy','precision','recall','f1','phi'),results_file,12,sep=',')
} else{
  write(c('Dataset','Rows','Fields','Train Time','Predict Time','Train Mem','Predict Mem','MAE','MSE','R-squared'),results_file,10,sep=',')
}

datasets = NULL
for (filename in listing){
  match_train = grepl(".*train.*.csv",filename)
  match_test = grepl(".*test.*.csv",filename)
  if (!match_train && !match_test)
    datasets = c(datasets,filename)
}
datasets = sort(datasets,decreasing=TRUE)
print('datasets: ')
print(datasets)

for (filename in datasets){
  train_files = NULL
  test_files = NULL
  basename = substr(filename,1,nchar(filename)-4)
  scores = NULL
  for ( i in 1:nfolds ){
    train_file = paste(basename,'_train_',i,'.csv',sep="")
    test_file = paste(basename,'_test_',i,'.csv',sep="")
    profile_file = paste('profile_',train_file,'.out',sep='')
    print(train_file)

    
    Rprof(filename=profile_file,memory.profiling=TRUE,interval=0.001)
    print('reading CSV files')
    train_data = read.csv(file.path(data_dir,train_file))
    test_data = read.csv(file.path(data_dir,test_file))
    x = train_data[,1:ncol(train_data)-1]
    y = train_data[,ncol(train_data)]
    xtest = test_data[,1:ncol(test_data)-1]
    ytest = test_data[,ncol(test_data)]
    
    print('training random forest')
    train_time =  unname(system.time(forest <- randomForest(x,y,ntree=10))[3])
    print('computing prediction')
    test_time = unname(system.time(ypredict <- predict(forest,xtest))[3])
    Rprof(NULL)
    profile_stats = summaryRprof(profile_file,memory='both')$by.total
    idx_train = match("\"randomForest\"",rownames(profile_stats))
    idx_test = match("\"predict\"",rownames(profile_stats))
    #train_time = profile_stats[idx_train,1]
    train_mem = profile_stats[idx_train,3]
    #test_time = profile_stats[idx_test,1]
    test_mem = profile_stats[idx_test,3]
    
    if (CLASSIFICATION){
      matrix = unname(confusionMatrix(ypredict,ytest)$table)
      fold_scores = compute_classification_metrics(matrix)
    } else {
      mean_squared_err = mse(ytest,ypredict)
      mean_abs_err = mae(ytest,ypredict)   
      n = length(ytest)
      residual_sum = mean_squared_err*n
      total_sum = sum((ytest - rep(mean(ytest),n))^2)
      r_squared = 1 - residual_sum/total_sum  
      fold_scores = c(mean_abs_err,mean_squared_err,r_squared)
    }
    print(c(train_time,test_time,train_mem,test_mem,fold_scores))
    scores = rbind(scores, c(train_time,test_time,train_mem,test_mem,
                             fold_scores))
  }
  mean_scores = colMeans(scores)
  print(mean_scores)
  write(c(basename,nrow(test_data)+nrow(train_data),ncol(test_data)-1,mean_scores),results_file,length(mean_scores)+3,append=TRUE,sep=',') 
}


# Rprof('rprofile.out',memory.profiling=TRUE)
# train = read.csv('synthdata_regression_100000_10_train_1.csv')
# test = read.csv('$synthdata_regression_100000_10_test_1.csv')
# # columns = colnames(train)
# # objective = columns[length(columns)]
# # formula_str = paste(objective,'.',sep='~')
# # f = as.formula(formula_str)
# # forest = randomForest(formula=f,data=train)
# x = train[,1:ncol(train)-1]
# y = train[,ncol(train)]
# xtest = test[,1:ncol(test)-1]
# ytest = test[,ncol(test)]
# forest = randomForest(x,y,ntree=10)
# ypredict = predict(forest,xtest)
# Rprof(NULL)
# profile_stats = summaryRprof('rprofile.out',memory='both')
