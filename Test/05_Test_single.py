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

t0 = time()

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

###############################################################################
# Test classifier on test dataset

# clf = DecisionTreeClassifier(max_depth=14, min_split=5)
# clf = MultinomialNB(alpha=.01)
# clf = KNeighborsClassifier(n_neighbors=19)
# clf = RidgeClassifier(tol=1e-1)
clf = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
# clf = SVC(C=32, gamma=0.0625)
print clf

t0 = time()
clf.fit(X_train, y_train)
print (time()-t0)
t1 = time()
pred = clf.predict(X_test)
print (time()-t1)

pre_score = metrics.precision_score(y_test, pred)
rec_score = metrics.recall_score(y_test, pred)

print "average f1-score:   %0.2f" % (100*((2*pre_score*rec_score)/(pre_score+rec_score)))
print "average f5-score:   %0.2f" % (100*((1.25*pre_score*rec_score)/(0.25*pre_score+rec_score)))
print "average precision:  %0.5f" % pre_score
print "averege recall:     %0.5f" % rec_score