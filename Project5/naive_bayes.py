from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_train)

### now print the accuracy
### accuracy = no. of points classified correctly / all points (in test set)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
return accuracy

