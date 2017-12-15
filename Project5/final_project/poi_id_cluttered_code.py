#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import sys
import pickle
sys.path.append("../tools/") 

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

from sklearn.preprocessing import Imputer

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = pd.concat([df.drop('email_address', axis=1).apply(pd.to_numeric, errors='coerce'), \
                df['email_address']], axis=1) 
                
drop_columns = ["other", "email_address"]
df = df.drop(drop_columns, axis=1)

df = df.drop(['TOTAL','THE TRAVEL AGENCY IN THE PARK'])    

# the persons of interest - 18
df.poi[df.poi == True]
#BELDEN TIMOTHY N        True
#BOWEN JR RAYMOND M      True
#CALGER CHRISTOPHER F    True
#CAUSEY RICHARD A        True
#COLWELL WESLEY          True
#DELAINEY DAVID W        True
#FASTOW ANDREW S         True
#GLISAN JR BEN F         True
#HANNON KEVIN P          True
#HIRKO JOSEPH            True
#KOENIG MARK E           True
#KOPPER MICHAEL J        True
#LAY KENNETH L           True
#RICE KENNETH D          True
#RIEKER PAULA H          True
#SHELBY REX              True
#SKILLING JEFFREY K      True
#YEAGER F SCOTT          True          
df.poi.sum()/float(len(df))
# proportion of poi's is low: 12.3% 
(df['poi'] == True).sum()
#18 POIs

# naive model for accuracy just assigns false to everyone
# accuracy would be 128/144 or 88.88% on this model
# precision would be 0
# recall would be 0

#salary                       0.264976
#to_messages                  0.058954
#deferral_payments           -0.098428
#total_payments               0.230102
#exercised_stock_options      0.503551
#bonus                        0.302384
#restricted_stock             0.224814
#shared_receipt_with_poi      0.228313
#restricted_stock_deferred         NaN
#total_stock_value            0.366462
#expenses                     0.060292
#loan_advances                0.999851
#from_messages               -0.074308
#other                        0.120270
#from_this_person_to_poi      0.112940
#poi                          1.000000
#director_fees                     NaN
#deferred_income             -0.265698
#long_term_incentive          0.254723
#from_poi_to_this_person      0.167722
#Name: poi, dtype: float64
                


# who is missing a lot of data
df.isnull().sum(axis=1).sort_values(inplace=False).tail(10)
# 'LOCKHART EUGENE E' nothing except POI false
# 'WODRASKA JOHN' nothing except POI false, total_payments
# df.loc[['SCRIMSHAW MATTHEW','WHALEY DAVID A','GRAMM WENDY L','WROBEL BRUCE']]
df.isnull().sum(axis=1).sort_values(inplace=False).quantile(.95)
# 95% have only 15 nulls, these have 16
df = df.drop(['LOCKHART EUGENE E','WODRASKA JOHN','SCRIMSHAW MATTHEW','WHALEY DAVID A','GRAMM WENDY L','WROBEL BRUCE'])

# which columns are missing a lot of values?
df.isnull().sum()
df = df.drop(['director_fees', 'loan_advances'], axis=1)
# loan advances is only on 3 people
#df = df.drop('loan_advances', axis=1)
#director_fees and restricted_stock_deferred have few values and only when POI is false
df = df.drop(['restricted_stock_deferred'], axis=1)
#too few compared to others
df = df.drop(['deferral_payments', 'deferred_income'], axis=1)
# correlations with POI?
df.corr()['poi'].sort_values()

#Some exploration
#df['people'] = df.index
df.describe()

sns.boxplot('poi', 'salary',
           data=df[df['salary'] < df['salary'].quantile(.95)])
           
sns.boxplot('poi', 'total_stock_value',
           data=df[df['total_stock_value'] < df['total_stock_value'].quantile(.95)])
# from_this_person_to_poi / from_messages
# from_poi_to_this_person / to_messages
df['sent_to_poi_pct'] = df['from_this_person_to_poi'] / df['from_messages']
df['received_from_poi_pct'] = df['from_poi_to_this_person'] / df['to_messages']
           
# strong corr: shared_receipt_with_poi, loan_advances, from_this_person_to_poi
# from_poi_to_this_person
features_list = [u'poi', u'salary', u'to_messages', u'total_payments',
       u'exercised_stock_options', u'bonus', u'restricted_stock',
       u'shared_receipt_with_poi', u'total_stock_value', u'expenses',
       u'from_messages', u'from_this_person_to_poi',
       u'long_term_incentive', u'from_poi_to_this_person']
       #'sent_to_poi_pct', 'received_from_poi_pct']

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')

### Use imputer to fill in missing values.

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)
    
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#acc = accuracy_score(labels_test, pred)
#prec = precision_score(labels_test, pred)
#reca = recall_score(labels_test, pred)
#matrix = confusion_matrix(labels_test, pred)
#tn, fp, fn, tp = matrix.ravel()

imp = Imputer(strategy='median')
imputed_features = imp.fit_transform(features)
i_features_train, i_features_test, i_labels_train, i_labels_test = \
    train_test_split(imputed_features, labels, test_size=0.25, random_state=42)
    
selector = SelectKBest(k='all')
i_features_train = selector.fit(i_features_train, i_labels_train)
i_features_list = selector.transform(features_list[1:]).tolist()[0]
i_features_test = selector.transform(imputed_features)

my_dataset = df.to_dict(orient='index')
features_list = [u'poi', u'salary', u'to_messages', u'total_payments',
       u'exercised_stock_options', u'bonus', u'restricted_stock',
       u'shared_receipt_with_poi', u'total_stock_value', u'expenses',
       u'from_messages', u'from_this_person_to_poi',
       u'long_term_incentive', u'from_poi_to_this_person']
scores_dict = dict.fromkeys(features_list[1:], np.array([]))       
def mean_selector_score(dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    imp = Imputer(strategy='median')
    features = imp.fit_transform(features)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    selector = SelectKBest(k='all')
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        pipe.fit(features_train, labels_train)
        for score, feature in zip(pipe.named_steps['selector'].scores_, features_list[1:]):
            scores_dict[feature] = np.append(scores_dict[feature], score)
mean_selector_score(my_dataset, features_list)

scores_df = pd.DataFrame.from_dict(scores_dict)

features_list_score_order = [u'exercised_stock_options', u'total_stock_value', u'bonus', u'salary',
       u'long_term_incentive', u'restricted_stock', u'total_payments',
       u'shared_receipt_with_poi', u'from_poi_to_this_person',
       u'from_this_person_to_poi', u'expenses', u'to_messages',
       u'from_messages']

#acc = []
#prec = []
#reca = []
#def try_all_k_best(max=13):
#    data = featureFormat(my_dataset, features_list, sort_keys = True)
#    labels, features = targetFeatureSplit(data)
#    features_train, features_test, labels_train, labels_test = \
#        train_test_split(features, labels, test_size=0.25, random_state=42)
#
#    for k in range(1,max+1):
#        pipe = Pipeline([('impute', Imputer(strategy='median')), 
#                         ('select', SelectKBest(k=k)),
#                         ('classify', LogisticRegressionCV())])
#        pipe.fit(features_train, labels_train)
#        total_predictions, accuracy, precision, recall, f1, f2 = \
#          test_classifier(pipe, my_dataset, features_list, folds=1000)
#        acc.append(accuracy)
#        prec.append(precision)
#        reca.append(recall)     
#
#try_all_k_best()
#test_df = pd.DataFrame({"prec": prec, "reca": reca})
#test_df['total'] = test_df['prec'] + test_df['reca']
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.25, random_state=42)

pipe = Pipeline([('impute', Imputer(strategy='median')), 
                    ('select', SelectKBest(k=3)),
                    ('classify', LogisticRegressionCV())])
pipe.fit(features_train, labels_train)
        
pred = pipe.predict(features_test)
acc = accuracy_score(labels_test, pred)
prec = precision_score(labels_test, pred)
reca = recall_score(labels_test, pred)  

dump_classifier_and_data(pipe, my_dataset, features_list)


features_list_score_order = [u'exercised_stock_options', u'total_stock_value', u'bonus', u'salary',
       u'long_term_incentive', u'restricted_stock', u'total_payments',
       u'shared_receipt_with_poi', u'from_poi_to_this_person',
       u'from_this_person_to_poi', u'expenses', u'to_messages',
       u'from_messages']
testing_features_list = [u'poi']
acc = []
prec = []
reca = []
acc_all = []
prec_all = []
reca_all = []
results_dict = {}
def tuneNB():
    for i in range(1, 20):
        acc = []
        prec = []
        reca = []
        testing_features_list = [u'poi']
        for feature in features_list_score_order:
            testing_features_list.append(feature)
            pipe = Pipeline([('impute', Imputer(strategy='median')), 
                    ('classify', GaussianNB(priors=[(i/2.)*.1, (1 - (i/2.)*.1)]))])
            total_predictions, accuracy, precision, recall, f1, f2 = \
                test_classifier(pipe, my_dataset, testing_features_list, folds=200)
            acc.append(accuracy)
            prec.append(precision)
            reca.append(recall)
        acc_all.append(acc)
        prec_all.append(prec)
        reca_all.append(reca)
        results_dict['prec' + str(i)] = prec
        results_dict['reca' + str(i)] = reca
        results_dict['acc' + str(i)] = acc
tuneNB()
test_df = pd.DataFrame(results_dict)

# drop an i if no prec over .3 or reca over .3 or acc over .7
drop_me = []
for i in range(1,10):
    if (test_df['prec' + str(i)] > .3).sum() == 0 or \
       (test_df['reca' + str(i)] > .3).sum() == 0 or \
       (test_df['acc' + str(i)] > .7).sum() == 0:
        drop_me.extend(['prec' + str(i), 'reca' + str(i), 'acc' + str(i)])
test_df = test_df.drop(drop_me, axis=1)

# no single row with prec over .3 and reca over .3 and acc over .75
drop_me = []
for i in range(1,20):
    if (pd.concat([(test_df['prec' + str(i)] > .4),
       (test_df['reca' + str(i)] > .4),
       (test_df['acc' + str(i)] > .825)], axis=1).sum(axis=1) == 3).sum() == 0:
        drop_me.extend(['prec' + str(i), 'reca' + str(i), 'acc' + str(i)])
len(drop_me)
test_df = test_df.drop(drop_me, axis=1)

test_df = test_df.drop([7,8,9,10,11,12])
test_df = test_df.drop([0,1,6])


features_list = [u'poi', u'salary', u'total_payments',
       u'exercised_stock_options', u'bonus', u'total_stock_value']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

pipe = Pipeline([('impute', Imputer(strategy='median')), 
        ('select', SelectKBest(k='all')),
        ('classify', GaussianNB(priors=[7*.1, (1 - 7*.1)]))])
pipe.fit(features_train, labels_train)

total_predictions, accuracy, precision, recall, f1, f2 = \
    test_classifier(pipe, my_dataset, features_list, folds=1000)

dump_classifier_and_data(pipe, my_dataset, features_list)


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)

pipe = Pipeline([('impute', Imputer(strategy='median')), 
        ('select', SelectKBest(k=6)),
        ('classify', GaussianNB(priors=[7*.1, (1 - 7*.1)]))])
pipe.fit(features_train, labels_train)

total_predictions, accuracy, precision, recall, f1, f2 = \
    test_classifier(pipe, my_dataset, features_list, folds=50)

dump_classifier_and_data(pipe, my_dataset, features_list)


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)
    
pipe = Pipeline([('impute', Imputer(strategy='median')), 
        ('select', SelectKBest(k=5)),
        ('classify', GaussianNB())])
pipe.fit(features_train, labels_train)

total_predictions, accuracy, precision, recall, f1, f2 = \
    test_classifier(pipe, my_dataset, features_list, folds=50)

dump_classifier_and_data(pipe, my_dataset, features_list)


features_list_score_order = [u'poi', u'exercised_stock_options', u'total_stock_value', u'bonus']
pipe = Pipeline([('impute', Imputer(strategy='median')), 
        ('classify', GaussianNB(priors=[.15, .85]))])
total_predictions, accuracy, precision, recall, f1, f2 = \
    test_classifier(pipe, my_dataset, features_list_score_order, folds=1000)
    
features_list_score_order = [u'poi', u'exercised_stock_options', u'total_stock_value', u'bonus', u'salary',
       u'long_term_incentive']
pipe = Pipeline([('impute', Imputer(strategy='median')), 
        ('classify', GaussianNB(priors=[.15, .85]))])
total_predictions, accuracy, precision, recall, f1, f2 = \
    test_classifier(pipe, my_dataset, features_list_score_order, folds=1000)
    
features_list_score_order = [u'poi', u'exercised_stock_options', u'total_stock_value', u'bonus', u'salary',
       u'long_term_incentive']
pipe = Pipeline([('impute', Imputer(strategy='median')), 
        ('classify', GaussianNB(priors=[.1, .9]))])
total_predictions, accuracy, precision, recall, f1, f2 = \
    test_classifier(pipe, my_dataset, features_list_score_order, folds=1000)
    

features_list_score_order = [u'poi', u'exercised_stock_options', u'total_stock_value', u'bonus']
pipe = GaussianNB(priors=[.15,.85])
total_predictions, accuracy, precision, recall, f1, f2 = \
    test_classifier(pipe, my_dataset, features_list_score_order, folds=1000)
#GaussianNB(priors=[0.15, 0.85])
#        Accuracy: 0.82443       Precision: 0.39685      Recall: 0.44050 F1: 0.41754
#F2: 0.43102
#        Total predictions: 14000        True positives:  881    False positives: 1339
#        False negatives: 1119   True negatives: 10661
#        
        
dump_classifier_and_data(pipe, my_dataset, features_list_score_order)

#features_list = [u'poi', u'exercised_stock_options', u'total_stock_value', u'bonus', u'salary',
#       u'long_term_incentive']
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)
#imp = Imputer(strategy='median')
#imputed_features = imp.fit_transform(features)
#new_data = np.concatenate((np.array([labels]).T, imputed_features), axis=1)
#my_new_data_list = []
#new_data_dict = {}
#for values in new_data:
#    new_data_dict = {}
#    for feature_idx, value in enumerate(values):
#        
#my_new_dataset = dict(enumerate(new_data))
#    
#clf = GaussianNB()
#total_predictions, accuracy, precision, recall, f1, f2 = \
#    test_classifier(clf, my_new_dataset, features_list, folds=1000)

#Pipeline(steps=[('impute', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('classify', GaussianNB(priors=[0.15, 0.85]))])
#        Accuracy: 0.82393       Precision: 0.39503      Recall: 0.43750 F1: 0.41518     F2: 0.42829
#        Total predictions: 14000        True positives:  875    False positives: 1340   False negatives: 1125   True negatives: 10660
#
#
#Pipeline(steps=[('impute', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('classify', GaussianNB(priors=[0.15, 0.85]))])
#        Accuracy: 0.82086       Precision: 0.38966      Recall: 0.44850 F1: 0.41702     F2: 0.43535
#        Total predictions: 14000        True positives:  897    False positives: 1405   False negatives: 1103   True negatives: 10595
#
#Pipeline(steps=[('impute', Imputer(axis=0, copy=True, missing_values='NaN', strategy='median', verbose=0)), ('classify', GaussianNB(priors=[0.1, 0.9]))])
#        Accuracy: 0.81679       Precision: 0.38408      Recall: 0.46800 F1: 0.42191     F2: 0.44840
#        Total predictions: 14000        True positives:  936    False positives: 1501   False negatives: 1064   True negatives: 10499