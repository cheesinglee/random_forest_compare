# -*- coding: utf-8 -*-
from os import listdir, stat
from os.path import join,basename,split,splitext
from fnmatch import fnmatch
#from csv import DictReader, DictWriter
import csv
import re
import logging

from memory_profiler import memory_usage

from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from bigml.api import BigML

from forest_tester import ForestTester
from bigml_tester import BigMLTester
from sklearn_tester import SklearnTester
from wiserf_tester import WiseRFTester

DEFAULT_DATA_DIR='csv-data/test'

# random forest parameters
NUMBER_OF_TREES = 10
SAMPLE_RATE = 1.0
RANDOMIZE = True
BOOTSTRAP = True
SEED = 1234567890

NFOLDS=10

# BigML api object
api = None
    

def generate_cross_validation(filename,n_folds):
    """
    use sklearn kfolds class to create kfold cross validation data sources
    """

    # read the sourcefile
    with open(filename) as source_file:
#        reader = DictReader(source_file)
        reader = csv.reader(source_file)        
        fieldnames = reader.next()
        data = [row for row in reader]
#        fieldnames = reader.fieldnames
    
    # extract target labels and transform to sklearn format
#    label_key = fieldnames[-1]
    labels = [row[-1] for row in data]
    lenc = LabelEncoder()
    Y = lenc.fit_transform(labels)
        
    # create iterable to generate folds
    kfolds = StratifiedKFold(Y,n_folds)

    # iterate over folds and write CSV files    
    n = 1
    (head,tail) = split(filename)
    template = join(head,splitext(tail)[0])+'_{t}_{num}.csv'
    fold_filenames = []
    for idx_train,idx_test in kfolds:
        data_train = [data[i] for i in idx_train]
        data_test = [data[i] for i in idx_test]
        
        filename_train = template.format(t='train',num=str(n))
        filename_test = template.format(t='test',num=str(n))
        fold_filenames.append((filename_train,filename_test))
        n += 1
        
        with open(filename_train,'w') as train_file:
#            writer = DictWriter(train_file,fieldnames)
            writer = csv.writer(train_file)
            writer.writerow(fieldnames)
#            writer.writeheader()
            writer.writerows(data_train)
            
        with open(filename_test,'w') as test_file:
#            writer = DictWriter(test_file,fieldnames)
#            writer.writeheader()
            writer = csv.writer(test_file)
            writer.writerow(fieldnames)
            writer.writerows(data_test)    
        
    return fold_filenames   
    

if __name__ == "__main__":
    import sys
    from datetime import datetime
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 1:
        print 'Usage: forest_compare.py [method] [data_dir] [regression?]'
        sys.exit(0)
    
    if len(sys.argv) > 2:
        data_dir = sys.argv[2]
    else:
        data_dir = DEFAULT_DATA_DIR
    print 'Data directory is ',data_dir  
        
    if len(sys.argv) > 3:
        regression = (sys.argv[3].lower() == 'regression')
    else:
        regression = False

    if len(sys.argv) > 3:
        numeric = (sys.argv[4].lower() == 'numeric')
    else:
        numeric = False

    print 'Regression: ',regression      
    
    arg = sys.argv[1]
    print 'Got argument: ',arg
    if arg.lower() == 'wiserf':
        Tester = WiseRFTester
#        rf_tester = WiseRFTester(NUMBER_OF_TREES,SAMPLE_RATE,RANDOMIZE,BOOTSTRAP,SEED,regression=regression)
    elif arg.lower() == 'sklearn':
        Tester = SklearnTester
#        rf_tester = SklearnTester(NUMBER_OF_TREES,SAMPLE_RATE,RANDOMIZE,BOOTSTRAP,SEED,regression=regression)
    elif arg.lower() == 'bigml':
        Tester = BigMLTester
#        rf_tester = BigMLTester(NUMBER_OF_TREES,SAMPLE_RATE,RANDOMIZE,BOOTSTRAP,SEED,regression=regression)
    

    data = []
    
    print 'Generating cross-validation datasets...'
    for datafile in listdir(data_dir):     
        if fnmatch(datafile,'*.csv') and \
        not fnmatch(datafile,'*test*.csv') and \
        not fnmatch(datafile,'*train*.csv'):
            print datafile
            pattern = '{}_train_*.csv'.format(splitext(datafile)[0])
            training_files = [join(data_dir,filename) for filename in listdir(data_dir) if fnmatch(filename,pattern)]   
            training_files.sort()
            
            pattern = '{}_test_*.csv'.format(splitext(datafile)[0])
            test_files = [join(data_dir,filename) for filename in listdir(data_dir) if fnmatch(filename,pattern)]   
            test_files.sort()
            
            filename = join(data_dir,datafile)
            if len(training_files) == NFOLDS and len(test_files)==NFOLDS:
                print 'cross-validation data already exists for ',datafile
                validation_sets = zip(training_files,test_files)
            else:
        #        source = api.create_source(filename,{'name':datafile})
                validation_sets = generate_cross_validation(filename,NFOLDS)
            data.append({'filename':filename,'folds':validation_sets})
    print 'Finished generating cross-validation data'
    
    results = []
    if regression:
        fieldnames = [
        'name',
        'size',
        'mean_mae',
        'mean_mse',
        'mean_r_squared',
        'mean_train_time',
        'mean_predict_time',
        'mean_train_mem',
        'mean_predict_mem',
        'std_mae',
        'std_mse',
        'std_r_squared',
        'std_train_time',
        'std_predict_time',
        'std_train_mem',
        'std_predict_mem'
        ]
    else:
        fieldnames = [
            'name',
            'size',
            'mean_acc',
            'mean_recall',
            'mean_precision',
            'mean_f1',
            'mean_phi',
            'mean_train_time',
            'mean_predict_time',
            'mean_train_mem',
            'mean_predict_mem',
            'std_acc',
            'std_recall',
            'std_precision',
            'std_f1',
            'std_phi',
            'std_train_time',
            'std_predict_time',
            'std_train_mem',
            'std_predict_mem'
        ]

    filename = Tester.__name__+'_results_'+datetime.now().isoformat()+'.txt'
    with open(filename,'w') as fid:
        writer = csv.DictWriter(fid,fieldnames)
        writer.writeheader()    
    
    for test in data:
        print test['filename']
        validation_sets = test['folds']
        
        # get the name of the dataset
        pattern = r'(.*)_(test|train)_[\d]+.csv'
        result = re.match(pattern,validation_sets[0][0])
        if result is not None:
            dataset_name = result.group(1)
        else:
            dataset_name = validation_sets[0][0]
        dataset_name = basename(dataset_name)
        # get dataset size
        train_size = stat(validation_sets[0][0])[6]
        test_size = stat(validation_sets[0][1])[6]
        dataset_size = int(train_size) + int(test_size)
            
#        rf_tester.set_encodings(test['filename'])
#        baseline_mem = memory_usage()[0]
#        mem_usage = memory_usage((ForestTester.cross_validate,(rf_tester,validation_sets),{'numeric':True}),interval=5)
#        mem_usage = max(mem_usage) - baseline_mem
        rf_tester = Tester(NUMBER_OF_TREES,SAMPLE_RATE,RANDOMIZE,BOOTSTRAP,SEED,regression=regression)
        rf_tester.cross_validate(validation_sets,numeric=numeric)            
        means,std_devs,_ = rf_tester.get_results()
        del rf_tester
        output_list = [dataset_name,dataset_size] + list(means) + list(std_devs)
        results.append(dict(zip(fieldnames,output_list)))
            
        with open(filename,'a') as fid:
            writer = csv.DictWriter(fid,fieldnames)
            writer.writerow(dict(zip(fieldnames,output_list)))

        
        
