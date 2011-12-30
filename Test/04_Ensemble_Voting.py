"""
======================================================
Ensemble: Voting
======================================================
Ensemble learning
"""

print __doc__

from time import time
import numpy as np
# from itertools import izip

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
from sklearn.cross_validation import KFold

# return mode from a list
def most_common(lst):
    return max(set(lst), key=lst.count)


###############################################################################
# Preprocessing

# Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention', 'Linkout', 'California']

# Load data
data_set = load_files('../Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)

# Extract features
vectorizer = Vectorizer(max_features=10000)
X = vectorizer.fit_transform(data_set.data)
# X = Normalizer(norm="l2", copy=False).transform(X)

y = data_set.target

# Feature selection
select_chi2 = 1200
print ("Extracting %d best features by a chi-squared test" % select_chi2)
t0 = time()
ch2 = SelectKBest(chi2, k = select_chi2)
X = ch2.fit_transform(X, y)
print "Done in %fs" % (time() - t0)
print "L1:      n_samples: %d, n_features: %d" % X.shape
print

X_den = X.toarray()

n_samples, n_features = X.shape


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

# initialize empty y and z

print 'X_den shape: ', X_den.shape
print 'y shape:     ', y.shape

# Test for 10 rounds using the results from 10 fold cross validations
for i, (train_index, test_index) in enumerate(kf):

    print "run %d" % (i+1)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    X_den_train, X_den_test = X_den[train_index], X_den[test_index]

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
    # z_bNB   = np.append(z_bNB    , pred_bNB  , axis=None)
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
print 'z shape:     ', z.shape

n_samples = z.shape[0]
n_features = z.shape[1]
print 'n_samples', n_samples
print 'n_features', n_features


###############################################################################
# Voting

# array to store the final voting results
v = np.array([], dtype=np.int32)

for row in z:
    row_list = row.tolist()
    mode = most_common(row_list)
    v = np.append(v, mode, axis=None)

# make sure the shape of v is same as y
if (v.shape == y.shape):
    
    # scores
    # v here is same as pred in other situations
    f1_score = metrics.f1_score(y, v)
    acc_score =  float( np.sum(v == y) )  / float(len(y))
    pre_score = metrics.precision_score(y, v)
    rec_score = metrics.recall_score(y, v)

    # metrics of voting
    print "average f1-score:   %0.5f" % f1_score
    print "average precision:  %0.5f" % pre_score
    print "averege recall:     %0.5f" % rec_score
    print "average f1    :     %0.5f" % (2*pre_score*rec_score/(pre_score+rec_score))
    # print metrics.classification_report(y, v) 
    # print metrics.confusion_matrix(y, v)

else:
    print 'format error of v: not same as shape of y!'
