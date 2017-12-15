#!/usr/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import sys
import pickle
sys.path.append("../../tools/") 

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

# test_classifier from tester.py but added return values for use.
def my_test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
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
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        return total_predictions, accuracy, precision, recall, f1, f2
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."
        
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = pd.concat([df.drop('email_address', axis=1).apply(pd.to_numeric, errors='coerce'), \
                df['email_address']], axis=1) 
                
drop_columns = ["other", "email_address"]
df = df.drop(drop_columns, axis=1)


df = df.drop(['TOTAL','THE TRAVEL AGENCY IN THE PARK'])    
                
df = df.drop(['LOCKHART EUGENE E','WODRASKA JOHN','SCRIMSHAW MATTHEW','WHALEY DAVID A','GRAMM WENDY L','WROBEL BRUCE'])

df = df.drop(['director_fees', 'loan_advances'], axis=1)
df = df.drop(['restricted_stock_deferred'], axis=1)
df = df.drop(['deferral_payments', 'deferred_income'], axis=1)

# from_this_person_to_poi / from_messages
# from_poi_to_this_person / to_messages
df['sent_to_poi_pct'] = df['from_this_person_to_poi'] / df['from_messages']
df['received_from_poi_pct'] = df['from_poi_to_this_person'] / df['to_messages']
           
features_list = [u'poi', u'salary', u'to_messages', u'total_payments',
       u'exercised_stock_options', u'bonus', u'restricted_stock',
       u'shared_receipt_with_poi', u'total_stock_value', u'expenses',
       u'from_messages', u'from_this_person_to_poi',
       u'long_term_incentive', u'from_poi_to_this_person']

### Store to my_dataset for easy export below.
my_dataset = df.to_dict(orient='index')

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
acc = []
prec = []
reca = []
def try_all_k_best(max=13):
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.25, random_state=42)

    for k in range(1,max+1):
        pipe = Pipeline([('impute', Imputer(strategy='median')), 
                         ('select', SelectKBest(k=k)),
                         ('classify', LogisticRegressionCV())])
        pipe.fit(features_train, labels_train)
        total_predictions, accuracy, precision, recall, f1, f2 = \
          my_test_classifier(pipe, my_dataset, features_list, folds=1000)
        acc.append(accuracy)
        prec.append(precision)
        reca.append(recall)     

#try_all_k_best()
test_df = pd.DataFrame({"prec": prec, "reca": reca})
test_df['total'] = test_df['prec'] + test_df['reca']

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.25, random_state=42)

pipe = Pipeline([('impute', Imputer(strategy='median')), 
                    ('select', SelectKBest(k=3)),
                    ('classify', LogisticRegressionCV())])
pipe.fit(features_train, labels_train)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
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
#mean_selector_score(my_dataset, features_list)

scores_df = pd.DataFrame.from_dict(scores_dict)

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
                my_test_classifier(pipe, my_dataset, testing_features_list, folds=200)
            acc.append(accuracy)
            prec.append(precision)
            reca.append(recall)
        acc_all.append(acc)
        prec_all.append(prec)
        reca_all.append(reca)
        results_dict['prec' + str(i)] = prec
        results_dict['reca' + str(i)] = reca
        results_dict['acc' + str(i)] = acc
#tuneNB()
test_df = pd.DataFrame(results_dict)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results

features_list_score_order = [u'poi', u'exercised_stock_options', u'total_stock_value', u'bonus']
pipe = Pipeline([('impute', Imputer(strategy='median')), 
        ('classify', GaussianNB(priors=[.15, .85]))])
total_predictions, accuracy, precision, recall, f1, f2 = \
    my_test_classifier(pipe, my_dataset, features_list_score_order, folds=1000)   
        
dump_classifier_and_data(pipe, my_dataset, features_list_score_order)
