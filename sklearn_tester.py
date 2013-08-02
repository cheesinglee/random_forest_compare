# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:50:09 2013

@author: cheesinglee
"""

import cProfile
import pstats
import csv
import os
import gc
import sys
import time

from multiprocessing import cpu_count
from memory_profiler import memory_usage

from numpy import array,loadtxt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from forest_tester import ForestTester

class SmartDictReader(csv.DictReader):
    """
    DictReader with typechecking for numeric values
    """
    
    def next(self):
        if self.line_num == 0:
            # Used only for its side effect.
            self.fieldnames
        row = self.reader.next()
        self.line_num = self.reader.line_num

        # unlike the basic reader, we prefer not to return blanks,
        # because we will typically wind up with a dict full of None
        # values
        while row == []:
            row = self.reader.next()
            
        typechecked_row = []
        for item in row:
            try:
                typechecked_row.append(float(item))
            except ValueError:
                typechecked_row.append(item)
                
        d = dict(zip(self.fieldnames, typechecked_row))
        lf = len(self.fieldnames)
        lr = len(row)
        if lf < lr:
            d[self.restkey] = row[lf:]
        elif lf > lr:
            for key in self.fieldnames[lr:]:
                d[key] = self.restval
        return d

class SklearnTester(ForestTester):
    data_vectorizer = None
    label_encoder = None
    n_cores = 1
    
    def __init__(self,*args,**kwargs):
        ForestTester.__init__(self,*args,**kwargs)
        self.n_cores = cpu_count()
    
    def set_encodings(self,filename):
        """
        Fit the feature and class label encoders from a csv file
        """
        with open(filename) as fid:
            reader = SmartDictReader(fid)
            data = [row for row in reader]
            label_key = reader.fieldnames[-1]
            
        labels = [row.pop(label_key) for row in data]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        
        self.data_vectorizer = DictVectorizer()
        self.data_vectorizer.fit(data)
    
    def load_csv(self,filename):
        with open(filename) as fid:
            reader = SmartDictReader(fid)
            data = [row for row in reader]
            label_key = reader.fieldnames[-1]
        
        labels = [row.pop(label_key) for row in data]
        self.data_vectorizer = DictVectorizer()
        self.data_vectorizer.fit(data)
        X = self.data_vectorizer.transform(data).toarray()            
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)
        Y = self.label_encoder.transform(labels)
        
        return (X,Y)
        
    def load_csv_numeric(self,filename):
        with open(filename) as fid:
#            reader = csv.reader(fid)
#            # skip the header line
#            reader.next()
#            for row in reader:
#                l = row.pop()
#                data.append(map(float,row))
#                labels.append(int(l))
            all_data = loadtxt(fid,delimiter=',',skiprows=1)
                
        data = all_data[:,0:-1]
        labels = all_data[:,-1]
        return (data,labels)
        
    def train_and_test(self,train_file,test_file, numeric=False):
#        baseline_mem = memory_usage(os.getpid())[0]
        if numeric:
            X_train,Y_train = self.load_csv_numeric(train_file)
#            X_test,Y_test = self.load_csv_numeric(test_file)
        else:
            X_train,Y_train = self.load_csv(train_file)
#            X_test,Y_test = self.load_csv(test_file)
        
        if self.regression:
            ForestObject = RandomForestRegressor
        else:
            ForestObject = RandomForestClassifier
            
        forest = ForestObject(n_estimators=self.n_trees,
                          bootstrap=self.bootstrap,
                          random_state=self.seed,
                          n_jobs=1)

                                        
        pr = cProfile.Profile()
        pr.enable()
#        baseline_mem2 = sum([sys.getsizeof(o) for o in gc.get_objects()])
#        gc.disable()
        try:        
            usage = memory_usage((ForestObject.fit,(forest,X_train,Y_train),{}),
                                 max_usage=True,interval=0.01)      
        except OSError:
            self.logger.error('Memory error while training')
            return -1
#        usage2 = sum([sys.getsizeof(o) for o in gc.get_objects()])
        pr.disable()
        train_mem = usage  - self.baseline_mem
#        train_mem2 = usage2 - baseline_mem2
#        gc.enable()
        p = pstats.Stats(pr)
        train_time = p.total_tt
        
        del X_train,Y_train
#        time.sleep(3)
        pr.clear()
        pr.enable()
        
#        baseline_mem = memory_usage(os.getpid())[0]
        
        if numeric:
#            X_train,Y_train = self.load_csv_numeric(train_file)
            X_test,Y_test = self.load_csv_numeric(test_file)
        else:
#            X_train,Y_train = self.load_csv(train_file)
            X_test,Y_test = self.load_csv(test_file)
        usage,Y_predict = memory_usage((ForestObject.predict,(forest,X_test),{}),
                                       max_usage=True,retval=True,interval=0.01)
        pr.disable()
        predict_mem = usage - self.baseline_mem
        p = pstats.Stats(pr)
        predict_time = p.total_tt
        
        if self.regression:
            result = (Y_predict,Y_test)
        else:
            result = confusion_matrix(Y_test,Y_predict)
        
        del forest, X_test, Y_test
        gc.collect()
        return result,train_time,predict_time,train_mem,predict_mem
        
if __name__ == '__main__':
    from os import listdir
    from fnmatch import fnmatch
    from pprint import pprint
    import logging
    import time

    logging.basicConfig(level=logging.DEBUG)
#    validation_sets = []
#    template = 'csv-data/synth/numeric_class/synthdata_100000_10_{}_{}.csv'    
#    for i in range(1,2):
#        train = template.format('train',i)
#        test = template.format('test',i)
#        validation_sets.append((train,test))
#        
#    print 'cross validating with:'
#    pprint(validation_sets)
    
    tester = SklearnTester(10,1,True,True,1234567890,regression=True)
#    tester.set_encodings('csv-data/classification/adult.csv')
#    [mean_scores,std_scores,all_scores] = tester.cross_validate(validation_sets,numeric=True,regression=True)
#    print mean_scores
#    print std_scores
#    print all_scores
    
    dataset = '/home/cheesinglee/bigml/random_forest_compare/synthdata_regression_1000_10'
#    tester.set_encodings('csv-data/synth/synthdata_'+dataset+'.csv')
    for i in range(1):
        matrix = tester.train_and_test(dataset+'_train_1.csv',
                          dataset+'_test_1.csv',
                          numeric=True)
#        time.sleep(5)
        print matrix  
