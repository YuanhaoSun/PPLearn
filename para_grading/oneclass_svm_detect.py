"""
======================================================
One-class detector for share statement
======================================================
Using one class SVM.
"""

print __doc__

from time import time
import numpy as np
import csv

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.svm import OneClassSVM

###############################################################################
# Preprocessing

# Load categories
train_category = ['ShareState']
test_category  = ['Unlabeled']

# Load data
print "Loading training and test dataset:"
data_train = load_files('./Train', categories = train_category,
                        shuffle = True, random_state = 42)

data_test = load_files('./Test', categories = test_category,
                        shuffle = True, random_state = 42)
print 'data loaded'
print len(data_train.data)
print len(data_test.data)
print

# Extract features
print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()

vectorizer = Vectorizer(max_features=10000)
X_test = vectorizer.fit_transform(data_test.data)
X_test = Normalizer(norm="l2", copy=False).transform(X_test)

X = vectorizer.transform(data_train.data)
X = Normalizer(norm="l2", copy=False).transform(X)


# feature selection
# ch2 = SelectKBest(chi2, k = 16000)
# X = ch2.fit_transform(X, y)

X = X.toarray()
X_test = X_test.toarray()

n_samples, n_features = X.shape
test_samples, test_features = X_test.shape
print "done in %fs" % (time() - t0)
print "Train set - n_samples: %d, n_features: %d" % (n_samples, n_features)
print "Test set  - n_samples: %d, n_features: %d" % (test_samples, test_features)
print


# fit the model
# when nu=0.01, gamma=0.0034607 is the smallest to generate >0 result
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.005) 
clf.fit(X)
# predit on X_test
y_pred = clf.predict(X_test)

# Count number of selected items given different gamma and nu
# This change is interesting
# Could further study systematically usign grid search
# 
count = 0
for i, pred in enumerate(y_pred):
    if pred != -1:
        count += 1
print count


csvWriter = csv.writer(open("detected.csv","wb"))
for i, pred in enumerate(y_pred):
    if pred != -1:
        # print X_test.data[i]
        csvWriter.writerow([data_test.data[i]])