# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:06:26 2013

@author: cheesinglee
"""

import cProfile
import pstats
import subprocess
import re
import csv
from memory_profiler import memory_usage,profile

from PyWiseRF import WiseRF
from sklearn.metrics import confusion_matrix
from numpy import ceil,array

from sklearn_tester import SklearnTester

def mem_usage(pid):
    proc_file = '/proc/{}/statm'.format(pid)
#    print proc_file
    result = ''
    with open(proc_file) as fid:
        result = fid.next()
#        print result
        
    tokens = result.split(' ')
    return float(tokens[0])

class WiseRFTester(SklearnTester):
    
    # ENTER THE PATH TO YOUR WISERF BINARY HERE 
    wiserf = '/path/to/wiserf'

    def train_and_test(self,train_file,test_file,numeric=False):
        if numeric:
            # find out how many columns there are
            pattern = r'.*_(\d+)_(\d+)_test_\d+.csv'
            result = re.match(pattern,test_file)
            class_column = int(result.group(2))+1
            print 'class column: ',class_column
    #        with open(test_file) as fid:
    #            line = fid.readline()
    #            
    #        tokens = line.split(',')
    #        class_column = len(tokens)
                
            model_file = 'tmp.model'
            predictions_file = 'predictions.txt'
            n_jobs = int(ceil(1.25*self.n_cores))
            
            if self.regression:
                command = 'learn-regressor'
            else:
                command = 'learn-classifier'
                
            call = [self.wiserf,command,'--num-trees',str(self.n_trees),
                    '--in-file',train_file,'--model-file',model_file,
                    '--nthreads',str(n_jobs),'--class-column',str(class_column)] 
            
            pr = cProfile.Profile()
            print 'Training on: ',train_file
            pr.enable()
            p = subprocess.Popen(call)
    #        peak_train_mem = mem_usage(p.pid)
            peak_train_mem = memory_usage(p.pid,timeout=0.1)[0]
            while p.poll() == None:
    #            tmp = mem_usage(p.pid)
                tmp = memory_usage(p.pid,timeout=0.1)[0]
                peak_train_mem = max([peak_train_mem,tmp])
            
            pr.disable()
            p = pstats.Stats(pr)
            self.logger.debug('Training time: %f s',p.total_tt)
            self.logger.debug('Training memory: %f',peak_train_mem)
            train_time = p.total_tt
            
            if self.regression:
                command = 'test-regressor'
            else:
                command = 'test-classifier'
                
            call = [self.wiserf,command,
                    '--in-file',test_file,'--model-file',model_file,
                    '--nthreads',str(n_jobs),'--class-column',str(class_column),
                    '--predictions-file',predictions_file]
    
            print 'Testing on: ',test_file             
            pr.clear()
            pr.enable()              
    #        Y_predict = forest.predict(X_test)
            p = subprocess.Popen(call)
            peak_predict_mem = memory_usage(p.pid,timeout=0.1)[0]
            while p.poll() == None:
                tmp = memory_usage(p.pid,timeout=0.1)[0]
                peak_predict_mem = max([peak_predict_mem,tmp])            
            pr.disable()
            p = pstats.Stats(pr)
            self.logger.debug('Prediction time: %f s',p.total_tt)
            self.logger.debug('Prediction memory: %f',peak_predict_mem)
            predict_time = p.total_tt
    
            # read predictions and true values        
            with open(predictions_file) as fid:
                predictions = array(fid.readline().strip().split(' '),dtype=float)[1:]
            with open(test_file) as fid:
                reader = csv.reader(fid)
                reader.next()
                true_values = [line.pop() for line in reader]
                true_values = array(true_values,dtype=float)
                
            if self.regression:
                results = (predictions,true_values)
            else:
                results = confusion_matrix(true_values,predictions)        
        else:
            if self.regression:
                method = 'regression'
            else:
                method = 'classification'
                
            pr = cProfile.Profile()
            pr.enable()
            X_train,Y_train = self.load_csv(train_file)
            X_test,Y_test = self.load_csv(test_file)
            pr.disable()
            p = pstats.Stats(pr)
            self.logger.debug('Loading time: %f s',p.total_tt)
            
            pr.clear()
            pr.enable()
            forest = WiseRF(n_estimators=self.n_trees,
                            method = method,
                            n_jobs=int(ceil(1.25*self.n_cores)),
                            random_state=self.seed)
            pr.disable()
            p = pstats.Stats(pr)
            self.logger.debug('Forest constructor time: %f s',p.total_tt)
    
    
            pr.clear()
            pr.enable()                        
            usage = memory_usage((WiseRF.fit,(forest,X_train,Y_train),{}),max_usage=True)
            pr.disable()
            p = pstats.Stats(pr)
            peak_train_mem = usage - self.baseline_mem
            train_time = p.total_tt
            self.logger.debug('Training time: %f s',p.total_tt)
            self.logger.debug('Training memory: %f',peak_train_mem)
            
            print 'Testing on: ',test_file             
            pr.clear()
            pr.enable()              
            (usage,Y_predict) = memory_usage((WiseRF.predict,(forest,X_test),{}),max_usage=True,retval=True)
            pr.disable()
            p = pstats.Stats(pr)
            peak_predict_mem = usage - self.baseline_mem
            self.logger.debug('Prediction time: %f s',p.total_tt)
            self.logger.debug('Prediction memory: %f',peak_predict_mem)
            predict_time = p.total_tt
            
            if self.regression:
                results = (Y_test,Y_predict)
            else:
                results = confusion_matrix(Y_test,Y_predict)   
        
        return results,train_time,predict_time,peak_train_mem,peak_predict_mem


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    tester = WiseRFTester(10,1,True,True,0,regression=True)
    dataset = '/home/cheesinglee/bigml/random_forest_compare/synthdata_regression_1000_10'
#    tester.set_encodings('csv-data/synth/synthdata_'+dataset+'.csv')
    matrix = tester.train_and_test(dataset+'_train_1.csv',
                          dataset+'_test_1.csv',
                          numeric=True)
    print matrix                          
