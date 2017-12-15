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

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
my_dataset = data_dict

features_list = [u'poi', u'salary', u'to_messages', u'total_payments',
       u'exercised_stock_options', u'bonus', u'restricted_stock',
       u'shared_receipt_with_poi', u'total_stock_value', u'expenses',
       u'loan_advances', u'from_messages', u'from_this_person_to_poi',
       u'from_poi_to_this_person'] 
       
pred = []
acc = []
prec = []
reca = []
choices = []
def try_all_k_best(max=13):
    for k in range(1,max+1):
        data = featureFormat(my_dataset, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=42)
        selector = SelectKBest(k=k)
        features_train = selector.fit_transform(features_train, labels_train)
        features_test = selector.transform(features_test)
        choices.append(selector.transform(np.array(features_list[1:]).reshape(1, -1)))
        lr_cv = LogisticRegressionCV()
        lr_cv.fit(features_train, labels_train)
        pred.append(lr_cv.predict(features_test))
        acc.append(accuracy_score(labels_test, pred[k-1]))
        prec.append(precision_score(labels_test, pred[k-1]))
        reca.append(recall_score(labels_test, pred[k-1]))     

try_all_k_best()
test_df = pd.DataFrame({"prec": prec, "reca": reca})
test_df['total'] = test_df['prec'] + test_df['reca']

from sklearn.linear_model import LogisticRegressionCV

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    
selector = SelectKBest(k=8)
features_train = selector.fit_transform(features_train, labels_train)
features_list = selector.transform(features_list[1:]).tolist()[0]
features_list.insert(0, 'poi')
features_test = selector.transform(features_test)

clf = LogisticRegressionCV()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
prec = precision_score(labels_test, pred)
reca = recall_score(labels_test, pred)

dump_classifier_and_data(clf, my_dataset, features_list)