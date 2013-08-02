# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:56:36 2013

@author: cheesinglee
"""

from numpy import *
import logging
import cProfile
import re
import os

from memory_profiler import memory_usage

class ForestTester:
    n_trees = 10     
    sample_rate = 1
    randomize = False
    bootstrap = True
    seed = 1234567890
    logger = None
    profiler = None
    regression = False
    baseline_mem = 0
     
    def __init__(self,n_trees,sample_rate,randomize,bootstrap,seed,regression=False):
        self.n_trees = n_trees
        self.sample_rate = sample_rate
        self.randomize = randomize
        self.bootstrap = bootstrap
        self.seed = seed
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
#        file_handler = logging.FileHandler(self.__class__.__name__+'.log')
#        self.logger.addHandler(file_handler)
        self.profiler = cProfile.Profile()
        self.regression = regression
        
    def set_encodings(self,filename):
        pass
        
    def compute_accuracy(self,matrix,class_idx=None):
        """
        compute accuracy score from confusion matrix
        """
        return trace(matrix)/sum(matrix)
        
    def compute_precision(self,matrix,class_idx=None):
        """
        compute precision score
        
        if optional kwarg class_idx is given, compute the precision for a 
        single class, otherwise compute macro average
        """
        
        n_classes = size(matrix,0)
        if class_idx:
            assert (class_idx < n_classes)  and (class_idx >= 0)
            return matrix[class_idx][class_idx]/sum(matrix,0)[class_idx]
        else:
            macro_precision = 0
            for i in range(n_classes):
                macro_precision += matrix[i][i]/sum(matrix,0)[i]
            return macro_precision/n_classes
            
    def compute_recall(self,matrix,class_idx=None):
        """
        compute recall score
        
        class_idx(int): class for which score is calculated. Defaults value is
                        None, which specifies computation of macro average
        """
        
        n_classes = size(matrix,0)
        if class_idx:
            assert (class_idx < n_classes) and (class_idx >= 0)
            return matrix[class_idx][class_idx]/sum(matrix,1)[class_idx]
        else:
            macro_recall = 0
            for i in range(n_classes):
                macro_recall += matrix[i][i]/sum(matrix,1)[i]
            return macro_recall/n_classes
            
    def compute_f1(self,matrix,class_idx=None):
        """
        compute F1 score
        """
        
        precision = self.compute_precision(matrix,class_idx)
        recall = self.compute_recall(matrix,class_idx)
        
        return 2*precision*recall/(precision+recall)
        
    def compute_phi(self,matrix,class_idx=None):
        """
        compute phi coefficient
        """
        
        def matthews_phi(matrix,i):
            tp = matrix[i][i]
            fp = sum(matrix,0)[i] - tp
            fn = sum(matrix,1)[i] - tp
            tmp_matrix = matrix.copy()
            tmp_matrix[i,:] = 0
            tmp_matrix[:,i] = 0
            tn = sum(tmp_matrix)
            return (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            
        n_classes = size(matrix,0)
        if class_idx:
            return matthews_phi(matrix,class_idx)
        else:
            macro_phi = 0
            for i in range(n_classes):
                macro_phi += matthews_phi(matrix,i)
            return macro_phi/n_classes
            
    def compute_mse(self,predictions,true_vals):
        diff = predictions - true_vals
        return mean(diff*diff)
        
    def compute_mae(self,predictions,true_vals):
        diff = predictions - true_vals
        return mean(abs(diff))
        
    def compute_r_squared(self,predictions,true_vals):
        sample_mean = mean(true_vals)
        ss_tot = self.compute_mse(true_vals,sample_mean)
        ss_res = self.compute_mse(predictions,true_vals)
        return 1 - ss_res/ss_tot
            
    def compute_classification_scores(self,matrix,class_idx=None):
        scores = zeros(5)
        scores[0] = self.compute_accuracy(matrix,class_idx)
        scores[1] = self.compute_recall(matrix,class_idx)
        scores[2] = self.compute_precision(matrix,class_idx)
        scores[3] = self.compute_f1(matrix,class_idx)
        scores[4] = self.compute_phi(matrix,class_idx)
        return scores
        
    def compute_regression_scores(self,predictions,true_vals):
        mae = self.compute_mae(predictions,true_vals)
        mse = self.compute_mse(predictions,true_vals)
        r_squared = self.compute_r_squared(predictions,true_vals)
        return array([mae,mse,r_squared])
                
    def train_and_test(self,train_file,test_file):
        """
        method to be overwritten by subclasses. 
        """
        print 'training file = ',train_file
        print 'test file = ',test_file
        return eye(3)
        
    def get_results(self):
        return (self.mean_scores,self.std_scores,self.all_scores)
        
    def cross_validate(self, validation_sets,numeric=False):
        if self.regression:
            results_dim = 7
        else:
            results_dim = 9
        all_scores = zeros((0,results_dim))
        self.baseline_mem = memory_usage(os.getpid())[0]
        for train_file,test_file in validation_sets:
            try:
                result,train_time,predict_time,train_mem,predict_mem = self.train_and_test(train_file,test_file,numeric)
            except OSError:
                all_scores = array(['FAIL']*results_dim)
                mean_scores = all_scores
                std_scores = all_scores
                self.logger.error('Failed')
                return (all_scores,mean_scores,std_scores)
            else:
                self.logger.debug('\n'+str(result))
                self.logger.debug(str([train_mem,predict_mem]))
                self.logger.debug(str([train_time,predict_time]))
                if self.regression:
                    fold_scores = self.compute_regression_scores(result[0],result[1])
                else:
                    fold_scores = self.compute_classification_scores(result.astype(float))
                times = array([train_time,predict_time])
                mems = array([train_mem,predict_mem])
                fold_scores = hstack((fold_scores,times,mems))
                print(fold_scores)
                all_scores = vstack((all_scores,fold_scores))
            
        mean_scores = mean(all_scores,0)
        std_scores = std(all_scores,0)
        self.mean_scores = mean_scores
        self.std_scores = std_scores
        self.all_scores = all_scores
        return (mean_scores,std_scores,all_scores)
            
         
