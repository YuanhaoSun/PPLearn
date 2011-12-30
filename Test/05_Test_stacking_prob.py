"""
======================================================
Final test
======================================================
"""

print __doc__

from time import time
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer, normalize, scale
from sklearn.feature_selection import SelectKBest, chi2

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

# Regression
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, LogisticRegression, ElasticNet, Lars
from sklearn.neighbors import KNeighborsRegressor

from sklearn import metrics
from sklearn.cross_validation import KFold, ShuffleSplit
from sklearn.utils import shuffle

###############################################################################
# Preprocessing

# Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention', 'Linkout', 'California']

# Load data
data_train = load_files('../Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
data_test = load_files('../Privacypolicy/test_04', categories = categories,
                        shuffle = True, random_state = 42)                        

y_train, y_test = data_train.target, data_test.target

# Extract features
vectorizer = Vectorizer(max_features=10000)
vectorizer.analyzer.stop_words = set([])

X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)
# X = Normalizer(norm="l2", copy=False).transform(X)

# # Feature selection
# ch2 = SelectKBest(chi2, k = 1150)
# X = ch2.fit_transform(X, y)
X_train = X_train.toarray()
X_test = X_test.toarray()


# X = X.toarray()
# X_den = X.toarray()
n_samples, n_features = X_train.shape
n_samples_test, n_features_test = X_test.shape

t0 = time()

###############################################################################
# Setup part
# 
# Notation:
# N: number for training examples; K: number of models in level 0
# X: feature matrix; y: result array; z_k: prediction result array for k's model
# 

# Setup 10 fold cross validation
fold_num = 10
kf = KFold(n_samples, k=fold_num, indices=True)

# set number of neighbors for kNN
n_neighb = 19

# Brute-force implementation
clf_mNB = MultinomialNB(alpha=.01)
# clf_kNN = KNeighborsClassifier(n_neighbors=n_neighb)
clf_ridge = RidgeClassifier(tol=1e-1)
clf_lSVC = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
clf_SVC = SVC(C=32, gamma=0.0625, probability=True)
# clf_SGD = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")

n_categories = len(set(y_train))
z = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)

###############################################################################
# Stacking
# 
# initialize empty y and z

# Test for 10 rounds using the results from 10 fold cross validations
for i, (train_index, test_index) in enumerate(kf):

    print "run %d" % (i+1)

    X_train_train, X_train_test = X_train[train_index], X_train[test_index]
    y_train_train, y_train_test = y_train[train_index], y_train[test_index]

    # X_den_train, X_den_test = X_den[train_index], X_den[test_index]

    # feed models
    clf_mNB.fit(X_train_train, y_train_train)
    # clf_kNN.fit(X_train_train, y_train_train)
    clf_ridge.fit(X_train_train, y_train_train)
    clf_lSVC.fit(X_train_train, y_train_train)
    clf_SVC.fit(X_train_train, y_train_train)

    # get prediction for this fold run
    prob_mNB    = clf_mNB.predict_proba(X_train_test)
    # prob_kNN    = clf_kNN.predict_proba(X_train_test)
    prob_ridge  = clf_ridge.decision_function(X_train_test)
    prob_lSVC   = clf_lSVC.decision_function(X_train_test)
    prob_SVC    = clf_SVC.predict_proba(X_train_test)

    # update z array for each model
    # z_temp = prob_lSVC
    # z_temp = (prob_ridge + prob_lSVC)
    z_temp = (prob_mNB + prob_ridge + prob_lSVC + prob_SVC)
    z = np.append(z, z_temp, axis=0)


# remove the first sub-1d-array of z, due to the creation with 0s
z = np.delete(z, 0, 0)
# the result of z is a 2d array with shape of (n_samples, n_categories)
# the elements are the sum of probabilities of classifiers on each (sample,category) pair
# Possible preprocessing on z
# z = normalize(z, norm="l2")
# z = scale(z)



###############################################################################
# Test part of stacking

# Brute-force implementation
clf_mNB = MultinomialNB(alpha=.01)
# clf_kNN = KNeighborsClassifier(n_neighbors=n_neighb)
clf_ridge = RidgeClassifier(tol=1e-1)
clf_lSVC = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
clf_SVC = SVC(C=32, gamma=0.0625, probability=True)
# clf_SGD = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")

###############################################################################
# Stacking
# 

# feed models
clf_mNB.fit(X_train, y_train)
# clf_kNN.fit(X_train, y_train)
clf_ridge.fit(X_train, y_train)
clf_lSVC.fit(X_train, y_train)
clf_SVC.fit(X_train, y_train)

# get prediction for this fold run
test_prob_mNB    = clf_mNB.predict_proba(X_test)
# prob_kNN    = clf_kNN.predict_proba(X_test)
test_prob_ridge  = clf_ridge.decision_function(X_test)
test_prob_lSVC   = clf_lSVC.decision_function(X_test)
test_prob_SVC    = clf_SVC.predict_proba(X_test)

test_z = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)



# update z array for each model
test_z_temp = (test_prob_mNB + test_prob_ridge + test_prob_lSVC + test_prob_SVC)
# test_z_temp = ( prob_ridge + prob_lSVC )
# test_z_temp = prob_lSVC
test_z = np.append(test_z, test_z_temp, axis=0)

# remove the first sub-1d-array of z, due to the creation with 0s
test_z = np.delete(test_z, 0, 0)
# the result of z is a 2d array with shape of (n_samples, n_categories)
# the elements are the sum of probabilities of classifiers on each (sample,category) pair
# Possible preprocessing on z
# test_z = normalize(test_z, norm="l2")
# z = scale(z)



###############################################################################
# Test classifier on test dataset

# clf = DecisionTreeClassifier(max_depth=14, min_split=5)
# clf = MultinomialNB(alpha=.01)
# clf = KNeighborsClassifier(n_neighbors=19)
clf = RidgeClassifier(tol=1e-1)
# clf = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
# clf = SVC(C=32, gamma=0.0625)
# print clf

clf.fit(z, y_train)
pred = clf.predict(test_z)

pre_score = metrics.precision_score(y_test, pred)
rec_score = metrics.recall_score(y_test, pred)

# print "average f1-score:   %0.5f" % ((2*pre_score*rec_score)/(pre_score+rec_score))
# print "average f5-score:   %0.5f" % ((1.25*pre_score*rec_score)/(0.25*pre_score+rec_score))
print "average f1-score:   %0.2f" % (100*((2*pre_score*rec_score)/(pre_score+rec_score)))
print "average f5-score:   %0.2f" % (100*((1.25*pre_score*rec_score)/(0.25*pre_score+rec_score)))
print "average precision:  %0.5f" % pre_score
print "averege recall:     %0.5f" % rec_score