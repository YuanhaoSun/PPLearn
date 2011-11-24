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
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm.sparse import LinearSVC
from sklearn.utils.extmath import density
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier



###############################################################################
# Load some categories from the training set
###############################################################################

# # Load categories
# categories = ['nolimitshare','notsell', 'notsellnotshare', 'sellshare', 'shareforexception', 
#             'shareforexceptionandconsent','shareonlyconsent', 'notsharemarketing']
# # Load data
# print "Loading privacy policy dataset for categories:"
# data_train = load_files('../Dataset/ShareStatement/raw', categories = categories,
#                         shuffle = True, random_state = 42)
# data_test = load_files('../Dataset/ShareStatement/test', categories = categories, 
#                         shuffle = True, random_state = 42)
# print 'data loaded'




# # load from pickle
# # load data and initialize classification variables
# data_set = joblib.load('../Dataset/train_datasets/data_set_origin.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_stemmed.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized_pos.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_pos_selected.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_pos_tagged.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_pos_bagged.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_sem_firstsense.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_sem_internal_sentence_wsd.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_sem_corpus_sentence_wsd.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_sem_corpus_word_wsd.pkl')
# # data_set = joblib.load('../Dataset/train_datasets/data_set_sem_internal_word_wsd.pkl')
# data_set = joblib.load('../Dataset/test_datasets/data_set_origin.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_stemmed.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_lemmatized.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_lemmatized_pos.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_pos_selected.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_pos_tagged.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_pos_bagged.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_sem_firstsense.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_sem_internal_sentence_wsd.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_sem_corpus_sentence_wsd.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_sem_corpus_word_wsd.pkl')
# # data_set = joblib.load('../Dataset/test_datasets/data_set_sem_internal_word_wsd.pkl')
# categories = data_set.target_names


print "%d documents (training set)" % len(data_train.data)
print "%d documents (testing set)" % len(data_test.data)
print "%d categories" % len(categories)
print

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print "Extracting features from the training dataset using a sparse vectorizer"
vectorizer = Vectorizer(max_features=5000)
vectorizer.analyzer.stop_words = set([])
# vectorizer.analyzer.stop_words = set(["amazon", "com", "inc", "emcnn", "alexann", "realnetworks", "googlenn", "linkedinnn",
#                                     "fox", "zyngann", "eann", "yahoorb", "travelzoo", "kalturann", "2cocd", "ign", "blizzardnn",
#                                     "jobstreetcom", "surveymonkeynn", "microsoftnn", "wral","spe", "t", "mobile", "opendns",
#                                     "bentley", "allvoices", "watson", "dyn", "ae", "dow", "jones", "webm", "toysrus", "bonnier",
#                                     "skype", "wnd", "landrover", "icue", "sei", "entersect", "padeals", "acs", "e",
#                                     "getty", "images", "winamp", "lionsgate", ])


X_train = vectorizer.fit_transform(data_train.data)
X_test = vectorizer.transform(data_test.data)

print "X_train: n_samples: %d, n_features: %d" % X_train.shape
print "X_test : n_samples: %d, n_features: %d" % X_test.shape
print

# Try todense
# X_train = X_train.todense()
# X_test = X_test.todense()

# ch2 = SelectKBest(chi2, k=200)
# X_train = ch2.fit_transform(X_train, y_train)
# X_test = ch2.transform(X_test)

print "X_train: n_samples: %d, n_features: %d" % X_train.shape
print "X_test : n_samples: %d, n_features: %d" % X_test.shape
print




clf = BernoulliNB(alpha=.1)
# clf = MultinomialNB(alpha=.01)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf = RidgeClassifier(tol=1e-1)
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
# clf = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)



clf.fit(X_train, y_train)
pred = clf.predict(X_test)

f1_score  = metrics.f1_score(y_test, pred)
acc_score = metrics.zero_one_score(y_test, pred)
pre_score = metrics.precision_score(y_test, pred)
rec_score = metrics.recall_score(y_test, pred)

print data_test.target_names
# print metrics.classification_report(y_test, pred)
print metrics.confusion_matrix(y_test, pred)

print clf
print "average f1-score:   %0.5f" % f1_score
print "average accuracy:   %0.5f" % acc_score
print "average precision:  %0.5f" % pre_score
print "averege recall:     %0.5f" % rec_score