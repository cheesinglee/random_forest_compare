# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:00:27 2013

@author: cheesinglee
"""

from os.path import basename, join
import os
import logging
import csv

from cProfile import Profile
from pstats import Stats
from memory_profiler import memory_usage

from forest_tester import ForestTester
from bigml.api import BigML,check_resource
from bigml.ensemble import Ensemble
from bigml.fields import Fields
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

def make_confusion_matrix(true_labels,predict_labels):
    all_labels = true_labels + predict_labels
    encoder = LabelEncoder()
    encoder.fit(all_labels)
    
    true_labels_enc = encoder.transform(true_labels)
    predict_labels_enc = encoder.transform(predict_labels)
    
    return confusion_matrix(true_labels_enc,predict_labels_enc)

class BigMLTester(ForestTester):
    api = None
    authenticated = False
    source_res = None
    ensemble_res = None
    logger = None
    train_time = -1
    predict_time = -1
    results = None
    test_data = None

    def __init__(self,*args,**kwargs):  
        print args
        print kwargs
        bigml_user = kwargs.get('bigml_user',None)
        bigml_key = kwargs.get('bigml_key',None)
        ForestTester.__init__(self,*args,**kwargs)
        self.authenticate(bigml_user,bigml_key)
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler('BigMLTester.log'))
        self.logger.setLevel(logging.DEBUG)
    
    def authenticate(self,bigml_user,bigml_key):
        """
        initialize the BigML API, do a short test to check authentication
        """
        
        self.api = BigML(username=bigml_user,api_key=bigml_key)
        
        result = self.api.list_sources()
        if result['code'] == 200:
            self.authenticated = True
        else:
            self.authenticated = False
    
    def upload_source(self,filename):
        """
        Upload a sourcefile to BigML. Return resource value.
        """
        assert self.authenticated, 'Not authenticated!'
        
        # check if source file has already been uploaded  
        query_string = 'name={}'.format(filename)
        matching_sources = self.api.list_sources(query_string)['objects']  
        if len(matching_sources) > 0:
            source = matching_sources[0]
            self.logger.info('{0} is already present in BigML'.format(basename(filename)))
        else:
            self.logger.info('uploading source to BigML...')
            source = self.api.create_source(filename,{'name':filename})
            # enter polling loop until source becomes ready
            check_resource(source['resource'],self.api.get_source)  
        
        return source['resource']
        
    def make_dataset(self,source_res):
        """
        Create a BigML dataset from the given source resource. Returns dataset
        resource value.
        """
        assert self.authenticated, 'Not authenticated!'
        
        # check if dataset has already been created
        query_string = 'source={}'.format(source_res)
        matching_datasets = self.api.list_datasets(query_string)['objects']
        if len(matching_datasets) > 0:
            dataset = matching_datasets[0]
            self.logger.info('A dataset already exits for this source')
        else:
            filename = self.api.get_source(source_res)['object']['file_name']
            datasetname = "{0}'s dataset".format(filename)
            dataset = self.api.create_dataset(source_res,{'name':datasetname})
            # enter polling loop until dataset becomes ready
            check_resource(dataset['resource'],self.api.get_dataset)        
                  
        return dataset['resource']
        
    def train_ensemble(self,train_data):
        assert self.authenticated, 'Not authenticated!'
               
        ensemble_args = {'number_of_models':self.n_trees,
                     'sample_rate':self.sample_rate,
                     'randomize':self.randomize,
                     'replacement':self.bootstrap,
                     'tlp':5}
        ensemble = self.api.create_ensemble(train_data,ensemble_args)
        self.ensemble_res = ensemble['resource']

        # enter polling loop until ensemble becomes ready
        ensemble = check_resource(self.ensemble_res,self.api.get_ensemble)
            
        self.logger.info('Ensemble is ready')
        self.train_time = ensemble['object']['status']['elapsed']/1000
        
                
    def test_ensemble(self,test_file):
        assert self.authenticated, 'Not authenticated!'
        
        # download a local copy of the ensemble
        self.logger.info('Creating local ensemble')
        local_ensemble = Ensemble(self.ensemble_res,api=self.api)
        
        # make the Fields object
        source = self.api.get_source(self.source_res)
        fields = Fields(source['object']['fields'])
        
        self.logger.info('Reading test data and generating predictions')
        true_labels = []
        predict_labels = []
        pr = Profile()
        pr.enable()
        with open(test_file) as fid:
            test_reader = csv.reader(fid)
            # skip the header line
            test_reader.next()
            for row in test_reader:
                row_list = [val for val in row]
                true_labels.append(row_list.pop())
                instance = fields.pair(row_list)
                predict_labels.append(local_ensemble.predict(instance,
                                                         by_name=False,
                                                         method=1))

        pr.disable()
        ps = Stats(pr)
        self.predict_time = ps.total_tt
#        eval_args = {'combiner':1}
#        evaluation = self.api.create_evaluation(self.ensemble_res,test_data,eval_args)
#        check_resource(evaluation['resource'],self.api.get_evaluation)   
#        evaluation = self.api.get_evaluation(evaluation['resource'])
#        matrix = evaluation['object']['result']['model']['confusion_matrix']
#        self.predict_time = evaluation['object']['status']['elapsed']/1000
        if self.regression:
            self.results = (predict_labels,true_labels)
        else:
            self.results = make_confusion_matrix(true_labels,predict_labels)
        
    def unpack_results(self,results):
        """
        unpack relevant metrics from results dictionary, depending on whether
        evaluation task was classification or regression
        """
        
        is_classification = results.has_key('average_phi')
        is_regression = results.has_key('r_squared')
        
        if is_classification:
            return [results['accuracy'],
                    results['average_recall'],
                    results['average_precision'],
                    results['average_f_measure'],
                    results['average_phi']]
        elif is_regression:
            return [results['mean_absolute_error'],
                    results['mean_squared_error'],
                    results['r_squared']]
    
    def train_and_test(self,train_file,test_file,numeric):
        self.logger.info('Uploading %s',train_file)
        self.source_res = self.upload_source(train_file)
        self.logger.debug('Training source id is: %s',self.source_res)
        
#        self.logger.info('Uploading %s',test_file)
#        test_source = self.upload_source(test_file)
#        self.logger.debug('Test source id is: %s',test_source)
        
        self.logger.info('Creating training dataset')
        train_data = self.make_dataset(self.source_res)
        self.logger.debug('Training dataset id is: %s',train_data)
        
        self.logger.info('Training ensemble...')
#        baseline_mem = memory_usage(os.getpid())[0]
        usage = memory_usage((BigMLTester.train_ensemble,(self,train_data),{}),interval=2,max_usage=True)
        peak_train_mem = usage - self.baseline_mem
        self.logger.debug('Ensemble id is: %s',self.ensemble_res)
        
        self.logger.info('Testing ensemble...')
#        baseline_mem = memory_usage(os.getpid())[0]
        usage = memory_usage((BigMLTester.test_ensemble,(self,test_file),{}),interval=2,max_usage=True)
        peak_predict_mem = usage - self.baseline_mem
        return self.results,self.train_time,self.predict_time,peak_train_mem,peak_predict_mem
        
if __name__ == '__main__':
    from os import listdir
    from fnmatch import fnmatch
    
    logging.basicConfig(level=logging.INFO)
    
    datadir = 'csv-data/synth/categorical_class'
    pattern = 'synthdata_20000_1000_train_1.csv'
    train_files = [join(datadir,f) for f in listdir(datadir) if fnmatch(f,pattern)]
    pattern = 'synthdata_20000_1000_test_1.csv'
    test_files = [join(datadir,f) for f in listdir(datadir) if fnmatch(f,pattern)]
    
    train_files.sort()
    test_files.sort()
    validation_sets = zip(train_files,test_files)
    tester = BigMLTester(10,1,True,True,1)
    mean_scores,std_scores,all_scores = tester.cross_validate(validation_sets)
    print mean_scores
    print std_scores
        
    
