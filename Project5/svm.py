from sklearn import svm
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)  

clf.predict([[2., 2.]])

from sklearn.svm import SVC
clf = SVC(kernel="linear")

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)