"""
======================================================
Classification - Save model
======================================================
Save models
"""

import numpy as np
import sys
from time import time

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm.sparse import LinearSVC, SVC

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.externals import joblib

###############################################################################
# Preprocessing

# Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor', 'Truste', 'Change', 'Location', 'Children', 'Contact', 
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

# X = X.todense()
X_den = X.toarray()

y = data_set.target
n_samples, n_features = X.shape
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print

# Save models
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print 'vectorizer saved'
print

clf = MultinomialNB(alpha=.01)
clf.fit(X, y)
joblib.dump(clf, 'models/classifier_MultinomialNB.pkl')
print 'NB saved'
print

clf = KNeighborsClassifier(n_neighbors=27)
clf.fit(X, y)
joblib.dump(clf, 'models/classifier_kNN.pkl')
print 'kNN saved'
print

clf = RidgeClassifier(alpha=2)
clf.fit(X, y)
joblib.dump(clf, 'models/classifier_Ridge.pkl')
print 'Ridge saved'
print

clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")
clf.fit(X, y)
joblib.dump(clf, 'models/classifier_SGD.pkl')
print 'SGD saved'
print

clf = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False)
clf.fit(X, y)
joblib.dump(clf, 'models/classifier_LinearSVC.pkl')
print 'LinearSVC saved'
print

clf = SVC(C=32, gamma=0.03125, kernel='rbf')
clf.fit(X, y)
joblib.dump(clf, 'models/classifier_SVC.pkl')
print 'SVC saved'
print

clf = OneVsRestClassifier(LogisticRegression(C=1000, penalty='l2'))
clf.fit(X_den, y)
joblib.dump(clf, 'models/classifier_Logistic.pkl')
print 'Logistics saved'
print