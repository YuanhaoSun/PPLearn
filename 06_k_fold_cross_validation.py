"""
======================================================
Classificatio with CrossValidation
======================================================
Using CrossValidation to get the fair accuracy of algorithms.
"""

print __doc__

from time import time
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics

import treelearn 

from sklearn.cross_validation import KFold



###############################################################################
# Preprocessing

# Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention']

# Load data
print "Loading privacy policy dataset for categories:"
print categories if categories else "all"
data_set = load_files('Privacypolicy_balance/raw', categories = categories,
                        shuffle = True, random_state = 42)
print 'data loaded'
print

# Extract features
print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_features=10000)
X = vectorizer.fit_transform(data_set.data)
X = Normalizer(norm="l2", copy=False).transform(X)

y = data_set.target

# feature selection
ch2 = SelectKBest(chi2, k = 1800)
X = ch2.fit_transform(X, y)

X = X.toarray()

n_samples, n_features = X.shape
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print


###############################################################################
# Test a classifier using K-fold Cross Validation

# Setup 10 fold cross validation
num_fold = 10
kf = KFold(n_samples, k=num_fold, indices=True)

# Note: NBs are not working
# clf = DecisionTreeClassifier(max_depth=12, min_split=3)
# clf = BernoulliNB(alpha=.1)
# clf = MultinomialNB(alpha=.01)
# clf = OneVsRestClassifier(LogisticRegression(penalty='l2'))
# clf = KNeighborsClassifier(n_neighbors=13)
# clf = RidgeClassifier(tol=1e-1)
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
# clf = LinearSVC(loss='l2', penalty='l1', C=1000, dual=False, tol=1e-3)
# Add Random Forest from treelearn library
clf = treelearn.ClassifierEnsemble(bagging_percent=0.8, base_model = treelearn.RandomizedTree(),
            stacking_model=SVC(C=1024, kernel='rbf', degree=3, gamma=0.001, probability=True))

# Initialize variables for couting the average
f1_all = 0.0
acc_all = 0.0
pre_all = 0.0
rec_all = 0.0

# Test for 10 rounds using the results from 10 fold cross validations
for train_index, test_index in kf:

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    train_time = time() - t0

    pred = clf.predict(X_test)
    test_time = time() - t0

    # metrics
    f1_score = metrics.f1_score(y_test, pred)
    acc_score = metrics.zero_one_score(y_test, pred)
    pre_score = metrics.precision_score(y_test, pred)
    rec_score = metrics.recall_score(y_test, pred)
    f1_all += f1_score
    acc_all += acc_score
    pre_all += pre_score
    rec_all += rec_score

f1_all = f1_all/num_fold
acc_all = acc_all/num_fold
pre_all = pre_all/num_fold
rec_all = rec_all/num_fold

print
print clf
print "average f1-score:   %0.5f" % f1_all
print "average accuracy:   %0.5f" % acc_all
print "average precision:  %0.5f" % pre_all
print "averege recall:     %0.5f" % rec_all