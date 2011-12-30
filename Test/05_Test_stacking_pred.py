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
clf_kNN = KNeighborsClassifier(n_neighbors=n_neighb)
clf_ridge = RidgeClassifier(tol=1e-1)
clf_lSVC = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
clf_SVC = SVC(C=32, gamma=0.0625)
# clf_SGD = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")

# empty ndarrays for predication results z_kn
z_mNB = np.array([], dtype=np.int32)
z_kNN = np.array([], dtype=np.int32)
z_ridge = np.array([], dtype=np.int32)
z_lSVC = np.array([], dtype=np.int32)
z_SVC = np.array([], dtype=np.int32)


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
    clf_kNN.fit(X_train_train, y_train_train)
    clf_ridge.fit(X_train_train, y_train_train)
    clf_lSVC.fit(X_train_train, y_train_train)
    clf_SVC.fit(X_train_train, y_train_train)

    # get prediction for this fold run
    pred_mNB    = clf_mNB.predict(X_train_test)
    pred_kNN    = clf_kNN.predict(X_train_test)
    pred_ridge  = clf_ridge.predict(X_train_test)
    pred_lSVC   = clf_lSVC.predict(X_train_test)
    pred_SVC    = clf_SVC.predict(X_train_test)

    # update z array for each model
    z_mNB   = np.append(z_mNB    , pred_mNB  , axis=None)
    z_kNN   = np.append(z_kNN    , pred_kNN  , axis=None)
    z_ridge = np.append(z_ridge  , pred_ridge, axis=None)
    z_lSVC  = np.append(z_lSVC   , pred_lSVC , axis=None)
    z_SVC   = np.append(z_SVC    , pred_SVC  , axis=None)


# putting z's from each model into one 2d matrix
# this is the (feature) input, similar as X, for level 1
# In level 1, y is still y.
# z = np.array([z_bNB, z_mNB, z_kNN, z_ridge, z_SGD, z_lSVC, z_SVC, z_tree, z_logis], dtype=np.int32)
z = np.array([z_mNB, z_kNN, z_ridge, z_lSVC, z_SVC], dtype=np.int32)
z = z.transpose()
# np.savetxt('z.txt', z, fmt='%d')

n_samples = z.shape[0]
n_features = z.shape[1]
n_categories = len(set(y_train))


# Level 1 input representation I
# transfer separate nominal into category count representation
# Example: [1 1 1 1 1 2] (len = #classifiers) 
#            --> [0 5 1 0 0 0 0 0 0 ... 0] (len = #categories)
z_temp = np.zeros((n_samples, n_categories))
for i, row in enumerate(z):
    for j, item in enumerate(row):
        z_temp[i,int(item)] += 1
z = z_temp.copy()



###############################################################################
# Test part of stacking

fold_num = 10
kf = KFold(n_samples_test, k=fold_num, indices=True)

# Brute-force implementation
clf_mNB = MultinomialNB(alpha=.01)
clf_kNN = KNeighborsClassifier(n_neighbors=n_neighb)
clf_ridge = RidgeClassifier(tol=1e-1)
clf_lSVC = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
clf_SVC = SVC(C=32, gamma=0.0625)
# clf_SGD = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")

# empty ndarrays for predication results z_kn
test_z_mNB = np.array([], dtype=np.int32)
test_z_kNN = np.array([], dtype=np.int32)
test_z_ridge = np.array([], dtype=np.int32)
test_z_lSVC = np.array([], dtype=np.int32)
test_z_SVC = np.array([], dtype=np.int32)

###############################################################################
# Stacking
# 
# initialize empty y and z

# Test for 10 rounds using the results from 10 fold cross validations
# for i, (train_index, test_index) in enumerate(kf):

#     print "run %d" % (i+1)

#     X_train_train, X_train_test = X_test[train_index], X_test[test_index]
#     y_train_train, y_train_test = y_test[train_index], y_test[test_index]

#     # X_den_train, X_den_test = X_den[train_index], X_den[test_index]

# feed models
clf_mNB.fit(X_train, y_train)
clf_kNN.fit(X_train, y_train)
clf_ridge.fit(X_train, y_train)
clf_lSVC.fit(X_train, y_train)
clf_SVC.fit(X_train, y_train)

# get prediction for this fold run
pred_mNB    = clf_mNB.predict(X_test)
pred_kNN    = clf_kNN.predict(X_test)
pred_ridge  = clf_ridge.predict(X_test)
pred_lSVC   = clf_lSVC.predict(X_test)
pred_SVC    = clf_SVC.predict(X_test)

# update z array for each model
test_z_mNB   = np.append(test_z_mNB    , pred_mNB  , axis=None)
test_z_kNN   = np.append(test_z_kNN    , pred_kNN  , axis=None)
test_z_ridge = np.append(test_z_ridge  , pred_ridge, axis=None)
test_z_lSVC  = np.append(test_z_lSVC   , pred_lSVC , axis=None)
test_z_SVC   = np.append(test_z_SVC    , pred_SVC  , axis=None)


# putting z's from each model into one 2d matrix
# this is the (feature) input, similar as X, for level 1
# In level 1, y is still y.
# z = np.array([z_bNB, z_mNB, z_kNN, z_ridge, z_SGD, z_lSVC, z_SVC, z_tree, z_logis], dtype=np.int32)
test_z = np.array([test_z_mNB, test_z_kNN, test_z_ridge, test_z_lSVC, test_z_SVC], dtype=np.int32)
test_z = test_z.transpose()
# np.savetxt('z.txt', z, fmt='%d')

n_samples = test_z.shape[0]
n_features = test_z.shape[1]
n_categories = len(set(y_train))


# Level 1 input representation I
# transfer separate nominal into category count representation
# Example: [1 1 1 1 1 2] (len = #classifiers) 
#            --> [0 5 1 0 0 0 0 0 0 ... 0] (len = #categories)
test_z_temp = np.zeros((n_samples, n_categories))
for i, row in enumerate(test_z):
    for j, item in enumerate(row):
        test_z_temp[i,int(item)] += 1
test_z = test_z_temp.copy()



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
print (time()-t0)
t1 = time()
pred = clf.predict(test_z)
print (time()-t1)

pre_score = metrics.precision_score(y_test, pred)
rec_score = metrics.recall_score(y_test, pred)

# print "average f1-score:   %0.5f" % ((2*pre_score*rec_score)/(pre_score+rec_score))
# print "average f5-score:   %0.5f" % ((1.25*pre_score*rec_score)/(0.25*pre_score+rec_score))
print "average f1-score:   %0.2f" % (100*((2*pre_score*rec_score)/(pre_score+rec_score)))
print "average f5-score:   %0.2f" % (100*((1.25*pre_score*rec_score)/(0.25*pre_score+rec_score)))
print "average precision:  %0.5f" % pre_score
print "averege recall:     %0.5f" % rec_score