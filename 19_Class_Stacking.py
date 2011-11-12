"""
======================================================
Test classifiers using 10-fold Cross Validation
======================================================
Test classifiers using 10-fold Cross Validation
"""

print __doc__

from time import time
import numpy as np

from itertools import izip

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
from sklearn.utils import check_arrays

from sklearn.cross_validation import KFold


# pairwise implementation
# for appending a list using one element with its previous element
def pairwise(iterable):
    "s -> (s0,s1), (s2,s3), (s4, s5), ..."
    a = iter(iterable)
    return izip(a, a)


###############################################################################
# Preprocessing

# Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention']

# Load data
print "Loading privacy policy dataset for categories:"
print categories if categories else "all"
data_set = load_files('Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
print 'data loaded'
# print "%d documents" % len(data_set.data)
# print "%d categories" % len(data_set.target_names)
print

# Extract features
print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_features=10000)
X = vectorizer.fit_transform(data_set.data)
X = Normalizer(norm="l2", copy=False).transform(X)

y = data_set.target

# # Feature selection
# select_chi2 = 1900
# print ("Extracting %d best features by a chi-squared test" % select_chi2)
# t0 = time()
# ch2 = SelectKBest(chi2, k = select_chi2)
# X = ch2.fit_transform(X, y)
# print "Done in %fs" % (time() - t0)
# print "L1:      n_samples: %d, n_features: %d" % X.shape
# print

X_den = X.toarray()

n_samples, n_features = X.shape
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print


###############################################################################
# setup part
# 
# Notation:
# N: number for training examples; K: number of models in level 0
# X: feature matrix; y: result array; z_k: prediction result array for k's model
# 

# Setup 10 fold cross validation
fold_num = 10
kf = KFold(n_samples, k=fold_num, indices=True)

# set number of neighbors for kNN
n_neighb = 13

# Best implementation... Too confusion for now...
# # Make a classifier list
# clfs = []
# clfs.append(BernoulliNB(alpha=.01))
# clfs.append(MultinomialNB(alpha=.01))
# clfs.append(KNeighborsClassifier(n_neighbors=n_neighb))
# clfs.append(RidgeClassifier(tol=1e-1))
# # clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l1"))
# clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"))
# # clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"))
# # clfs.append(LinearSVC(loss='l2', penalty='l1', C=1000, dual=False, tol=1e-3))
# clfs.append(LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3))
# clfs.append(SVC(C=1000))

# Brute-force implementation
clf_bNB = BernoulliNB(alpha=.01)
clf_mNB = MultinomialNB(alpha=.01)
clf_kNN = KNeighborsClassifier(n_neighbors=n_neighb)
clf_ridge = RidgeClassifier(tol=1e-1)
# clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l1"))
clf_SGD = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")
# clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"))
# clfs.append(LinearSVC(loss='l2', penalty='l1', C=1000, dual=False, tol=1e-3))
clf_lSVC = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
clf_SVC = SVC(C=1000)

# empty ndarrays for predication results z_kn
z_bNB = np.array([], dtype=np.int32)
z_mNB = np.array([], dtype=np.int32)
z_kNN = np.array([], dtype=np.int32)
z_ridge = np.array([], dtype=np.int32)
z_SGD = np.array([], dtype=np.int32)
z_lSVC = np.array([], dtype=np.int32)
z_SVC = np.array([], dtype=np.int32)
# Best implementation... Too confusion for now...
# z_m = []
# z_m.append(z_bNB)
# z_m.append(z_mNB)
# z_m.append(z_kNN)
# z_m.append(z_ridge)
# z_m.append(z_SGD)
# z_m.append(z_lSVC)
# z_m.append(z_SVC)

# Best implementation... Too confusion for now...
# clfs_den = []
# clfs_den.append(DecisionTreeClassifier(min_split=5))
# # clfs_den.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l1')))
# clfs_den.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l2')))

clf_tree = DecisionTreeClassifier(min_split=5)
clf_logis = OneVsRestClassifier(LogisticRegression(C=1000,penalty='l2'))

# empty ndarrays for predication results z_kn
z_tree = np.array([], dtype=np.int32)
z_logis = np.array([], dtype=np.int32)
# z_m_den = []
# z_m_den.append(z_tree)
# z_m_den.append(z_logis)



###############################################################################
# stacking

# initialize empty y and z

print X_den.shape
print y.shape

# Test for 10 rounds using the results from 10 fold cross validations
for i, (train_index, test_index) in enumerate(kf):

    print "run %d" % (i+1)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_den_train, X_den_test = X_den[train_index], X_den[test_index]

    # feed models
    clf_bNB.fit(X_train, y_train)
    clf_mNB.fit(X_train, y_train)
    clf_kNN.fit(X_train, y_train)
    clf_ridge.fit(X_train, y_train)
    clf_SGD.fit(X_train, y_train)
    clf_lSVC.fit(X_train, y_train)
    clf_SVC.fit(X_train, y_train)
    clf_tree.fit(X_den_train, y_train)
    clf_logis.fit(X_den_train, y_train)

    # get prediction for this fold run
    pred_bNB    = clf_bNB.predict(X_test)
    pred_mNB    = clf_mNB.predict(X_test)
    pred_kNN    = clf_kNN.predict(X_test)
    pred_ridge  = clf_ridge.predict(X_test)
    pred_SGD    = clf_SGD.predict(X_test)
    pred_lSVC   = clf_lSVC.predict(X_test)
    pred_SVC    = clf_SVC.predict(X_test)
    pred_tree   = clf_tree.predict(X_den_test)
    pred_logis  = clf_logis.predict(X_den_test)

    # update z array for each model
    z_bNB   = np.append(z_bNB    , pred_bNB  , axis=None)
    z_mNB   = np.append(z_mNB    , pred_mNB  , axis=None)
    z_kNN   = np.append(z_kNN    , pred_kNN  , axis=None)
    z_ridge = np.append(z_ridge  , pred_ridge, axis=None)
    z_SGD   = np.append(z_SGD    , pred_SGD  , axis=None)
    z_lSVC  = np.append(z_lSVC   , pred_lSVC , axis=None)
    z_SVC   = np.append(z_SVC    , pred_SVC  , axis=None)
    z_tree  = np.append(z_tree   , pred_tree , axis=None)
    z_logis = np.append(z_logis  , pred_logis, axis=None)

    # Best implementation... Too confusion for now...
    # # update z array for each model during k-fold CV
    # # finally, z (array of predication) of  each model
    # # will be full size for all training examples
    # for clf, (z_temp_head, z_temp_tail) in zip(clfs, pairwise(z_m)):
    #     clf.fit(X_train, y_train)
    #     pred = clf.predict(X_test)
    #     z_temp = np.append(z_temp, pred, axis=None)


    # Best implementation... Too confusion for now...
    # for clf, z_temp_den in zip(clfs_den, z_m_den):
    #     clf.fit(X_den_train, y_train)
    #     pred = clf.predict(X_den_test)
    #     z_temp_den = np.append(z_temp_den, pred, axis=None)
    #     print z_temp_den.shape




# putting z's from each model into one 2d matrix
# this is the (feature) input, similar as X, for level 1
# In level 1, y is still y.
z = np.array([z_bNB, z_mNB, z_kNN, z_ridge, z_SGD, z_lSVC, z_SVC, z_tree, z_logis], dtype=np.int32)
# z = np.array([z_bNB, z_mNB], dtype=np.int32)
z = z.transpose()
print z.shape
print z


###############################################################################
# level 1 traing

# clf = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
clf = BernoulliNB(alpha=.01)

# Initialize variables for couting the average
f1_all = 0.0
acc_all = 0.0
pre_all = 0.0
rec_all = 0.0

# level 1 evaluation
for train_index, test_index in kf:

    z_train, z_test = z[train_index], z[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(z_train, y_train)
    pred = clf.predict(z_test)

    #Scores
    f1_score = metrics.f1_score(y_test, pred)
    f1_all += f1_score
    
    # change the calculateion of accuracy
    # acc_score = class_metrics(y_test, pred, acc=1)
    # acc_score  = np.mean(pred.ravel() == y_test.ravel()) * 100
    acc_score =  float( np.sum(pred == y_test) )  / float(len(y_test))
    acc_all += acc_score

    pre_score = metrics.precision_score(y_test, pred)
    rec_score = metrics.recall_score(y_test, pred)
    pre_all += pre_score
    rec_all += rec_score

f1_all = f1_all/fold_num
acc_all = acc_all/fold_num
pre_all = pre_all/fold_num
rec_all = rec_all/fold_num

# average the metrics from 10 fold
print clf
print "average f1-score:   %0.5f" % f1_all
print "average precision:  %0.5f" % pre_all
print "averege recall:     %0.5f" % rec_all
# changed accuracy calculation
# NOT APPLICABLE any more: accuracy is not a good metrics
#                   because the sparse nature of matrix
print "average accuracy:   %0.5f" % acc_all
print









# for clf in clfs:

#     # Initialize variables for couting the average
#     f1_all = 0.0
#     acc_all = 0.0
#     pre_all = 0.0
#     rec_all = 0.0

#     # Test for 10 rounds using the results from 10 fold cross validations
#     for train_index, test_index in kf:

#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         clf.fit(X_train, y_train)
#         pred = clf.predict(X_test)

#         #Scores
#         f1_score = metrics.f1_score(y_test, pred)
#         f1_all += f1_score
        
#         # change the calculateion of accuracy
#         # acc_score = class_metrics(y_test, pred, acc=1)
#         # acc_score  = np.mean(pred.ravel() == y_test.ravel()) * 100
#         acc_score =  float( np.sum(pred == y_test) )  / float(len(y_test))
#         acc_all += acc_score

#         pre_score = metrics.precision_score(y_test, pred)
#         rec_score = metrics.recall_score(y_test, pred)
#         pre_all += pre_score
#         rec_all += rec_score

#     f1_all = f1_all/fold_num
#     acc_all = acc_all/fold_num
#     pre_all = pre_all/fold_num
#     rec_all = rec_all/fold_num

#     # average the metrics from 10 fold
#     print clf
#     print "average f1-score:   %0.5f" % f1_all
#     print "average precision:  %0.5f" % pre_all
#     print "averege recall:     %0.5f" % rec_all
#     # changed accuracy calculation
#     # NOT APPLICABLE any more: accuracy is not a good metrics
#     #                   because the sparse nature of matrix
#     print "average accuracy:   %0.5f" % acc_all
#     print


# for clf in clfs_den:

#     # Initialize variables for couting the average
#     f1_all = 0.0
#     acc_all = 0.0
#     pre_all = 0.0
#     rec_all = 0.0

#     # Test for 10 rounds using the results from 10 fold cross validations
#     for train_index, test_index in kf:

#         X_train, X_test = X_den[train_index], X_den[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#         clf.fit(X_train, y_train)
#         pred = clf.predict(X_test)

#         #Scores
#         f1_score = metrics.f1_score(y_test, pred)
#         f1_all += f1_score

#         # change the calculateion of accuracy
#         # acc_score = class_metrics(y_test, pred, acc=1)
#         acc_score  = np.mean(pred.ravel() == y_test.ravel()) * 100
#         acc_all += acc_score
        
#         pre_score = metrics.precision_score(y_test, pred)
#         rec_score = metrics.recall_score(y_test, pred)
#         pre_all += pre_score
#         rec_all += rec_score

#     # average the metrics from 10 fold
#     f1_all = f1_all/fold_num
#     acc_all = acc_all/fold_num
#     pre_all = pre_all/fold_num
#     rec_all = rec_all/fold_num

#     print clf
#     print "average f1-score:   %0.5f" % f1_all
#     print "average precision:  %0.5f" % pre_all
#     print "averege recall:     %0.5f" % rec_all
#     # changed accuracy calculation
#     # NOT APPLICABLE any more: accuracy is not a good metrics
#     #                   because the sparse nature of matrix
#     # print "average accuracy:   %0.5f" % acc_all
#     print