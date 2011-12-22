"""
======================================================
Classification - PyLearner
======================================================
Testing various classifiers available from Scikits-Learn
"""

import logging
import numpy as np
from operator import itemgetter
from optparse import OptionParser
import sys
from time import time

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import metrics
from sklearn.externals import joblib

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier



###############################################################################
# Load some categories from the training set
###############################################################################

# # Load categories
# categories = ['nolimitshare','notsell', 'notsellnotshare', 'notsharemarketing', 'sellshare', 
#             'shareforexception', 'shareforexceptionandconsent','shareonlyconsent']
# categories3 = ['good','neutral', 'bad']
# # Load data
# print "Loading privacy policy dataset for categories:"
# data_train = load_files('../Dataset/ShareStatement/raw', categories = categories,
#                         shuffle = True, random_state = 42)
# data_test = load_files('../Dataset/ShareStatement/test', categories = categories, 
#                         shuffle = True, random_state = 42)
# # data_train = load_files('../Dataset/ShareStatement3/raw', categories = categories3,
# #                         shuffle = True, random_state = 42)
# # data_test = load_files('../Dataset/ShareStatement3/test', categories = categories3,
# #                         shuffle = True, random_state = 42)
# print 'data loaded'



# # load from pickle
# # load data and initialize classification variables
# data_train = joblib.load('../Dataset/train_datasets/data_set_origin.pkl')
# data_train = joblib.load('../Dataset/train_datasets/data_set_negation_bigram.pkl')
# data_train = joblib.load('../Dataset/train_datasets/data_set_pos_negation_bigram.pkl')
# data_train = joblib.load('../Dataset/train_datasets/data_set_pos_tagged.pkl')

# data_test = joblib.load('../Dataset/test_datasets/data_set_origin.pkl')
# data_test = joblib.load('../Dataset/test_datasets/data_set_negation_bigram.pkl')
# data_test = joblib.load('../Dataset/test_datasets/data_set_pos_negation_bigram.pkl')
# data_test = joblib.load('../Dataset/test_datasets/data_set_pos_tagged.pkl')


# data_train = joblib.load('../Dataset/train_datasets_3/data_set_origin.pkl')
# data_train = joblib.load('../Dataset/train_datasets_3/data_set_negation_bigram.pkl')
data_train = joblib.load('../Dataset/train_datasets_3/data_set_pos_negation_bigram.pkl')
# data_train = joblib.load('../Dataset/train_datasets_3/data_set_pos_tagged.pkl')

# data_test = joblib.load('../Dataset/test_datasets_3/data_set_origin.pkl')
# data_test = joblib.load('../Dataset/test_datasets_3/data_set_negation_bigram.pkl')
data_test = joblib.load('../Dataset/test_datasets_3/data_set_pos_negation_bigram.pkl')
# data_test = joblib.load('../Dataset/test_datasets_3/data_set_pos_tagged.pkl')
categories = data_train.target_names


# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print "Extracting features from the training dataset using a sparse vectorizer"
vectorizer = Vectorizer(max_features=5000)
vectorizer.analyzer.stop_words = set([]) # stopword redefine

X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

# # Build dictionary after vectorizer is fit
# # print vectorizer.vocabulary
vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary.iteritems(), key=itemgetter(1))])



# Iteration for feature selection
X_train_origin = X_train
X_test_origin = X_test
num_total_features = X_train_origin.shape[1]

for i in range(50):

    ch2 = SelectKBest(chi2, k = num_total_features*(i+1)*0.02)
    X_train = ch2.fit_transform(X_train_origin, y_train)
    X_test = ch2.transform(X_test_origin)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    # clf = BernoulliNB(alpha=.1)
    # clf = MultinomialNB(alpha=.01)
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = RidgeClassifier(tol=1e-1)
    # clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_split=2, random_state=42)
    # clf = SGDClassifier(alpha=.01, n_iter=50, penalty="l2")
    clf = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
    # clf = SVC(C=1000, kernel='rbf', gamma=1e-3)

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # # print out top words for each category
    # for i, category in enumerate(categories):
    #             top = np.argsort(clf.coef_[i, :])[-20:]
    #             print "%s: %s" % (category, " ".join(vocabulary[top]))
    #             print
    # print
    # print

    pre_score = metrics.precision_score(y_test, pred)
    rec_score = metrics.recall_score(y_test, pred)

    # print clf
    # print ((2*pre_score*rec_score)/(pre_score+rec_score))
    print ((1.25*pre_score*rec_score)/(0.25*pre_score+rec_score))
    # print "average f1-score:   %0.5f" % ((2*pre_score*rec_score)/(pre_score+rec_score))
    # print "average f5-score:   %0.5f" % ((1.25*pre_score*rec_score)/(0.25*pre_score+rec_score))
    # print "average precision:  %0.5f" % pre_score
    # print "averege recall:     %0.5f" % rec_score