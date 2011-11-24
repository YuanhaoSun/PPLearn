from time import time
import numpy as np
from operator import itemgetter

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.cross_validation import KFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.lda import LDA
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

import treelearn 





###############################################################################
# Preprocessing


# # Load from raw data
# # Load categories
# categories = ['nolimitshare','notsell', 'notsellnotshare', 'sellshare', 'shareforexception', 
#             'shareforexceptionandconsent','shareonlyconsent', 'notsharemarketing']
# # Load data
# print "Loading privacy policy dataset for categories:"
# print categories if categories else "all"
# data_set = load_files('../Dataset/ShareStatement/raw', categories = categories,
#                         shuffle = True, random_state = 42)
# print 'data loaded'
# print


# load from pickle
# load data and initialize classification variables
data_set = joblib.load('../Dataset/train_datasets/data_set_origin.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_stemmed.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized_pos.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_pos_selected.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_pos_tagged.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_pos_bagged.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_firstsense.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_internal_sentence_wsd.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_corpus_sentence_wsd.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_corpus_word_wsd.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_internal_word_wsd.pkl')
categories = data_set.target_names


y = data_set.target


# Extract features
vectorizer = Vectorizer(max_features=10000)

# Engineering nGram
# vectorizer.analyzer.max_n = 2

# Engineering stopword
# vectorizer.analyzer.stop_words = set([])
vectorizer.analyzer.stop_words = set(["amazon", "com", "inc", "emcnn", "alexann", "realnetworks", "googlenn", "linkedinnn",
                                    "fox", "zyngann", "eann", "yahoorb", "travelzoo", "kalturann", "2cocd", "ign", "blizzardnn",
                                    "jobstreetcom", "surveymonkeynn", "microsoftnn", "wral", "spe", "t", "mobile", "opendns",
                                    "bentley", "allvoices", "watson", "dyn", "ae", "dow", "jones", "webm", "toysrus", "bonnier",
                                    "skype", "wnd", "landrover", "icue", "sei", "entersect", "padeals", "acs", "e",
                                    "getty", "images", "winamp", "lionsgate", ])
# vectorizer.analyzer.stop_words = set(["amazon", "com", "inc", "emc", "alexa", "realnetworks", "google", "linkedin",
#                                     "fox", "zynga", "ea", "yahoo", "travelzoo", "kaltura", "2co", "ign", "blizzard",
#                                     "jobstreetcom", "surveymonkey", "microsoft", "wral", "spe", "t", "mobile", "opendns",
#                                     "bentley", "allvoices", "watson", "dyn", "ae", "dow", "jones", "webm", "toysrus", "bonnier",
#                                     "skype", "wnd", "landrover", "icue", "sei", "entersect", "padeals", "acs", "e",
#                                     "getty", "images", "winamp", "lionsgate", ])
# vectorizer.analyzer.stop_words = set(["we", "do", "you", "your", "the", "that", "this", 
#                                     "is", "was", "are", "were", "being", "be", "been",
#                                     "for", "of", "as", "in",  "to", "at", "by",
#                                     # "or", "and",
#                                     "ve",
#                                     "amazon", "com", "inc", "emc", "alexa", "realnetworks", "google", "linkedin",
#                                     "fox", "zynga", "ea", "yahoo", "travelzoo", "kaltura", "2co", "ign", "blizzard",
#                                     "jobstreetcom", "surveymonkey", "microsoft", "wral", "spe", "t", "mobile", "opendns",
#                                     "bentley", "allvoices", "watson", "dyn", "ae", "dow", "jones", "webm", "toysrus", "bonnier",
#                                     "skype", "wnd", "landrover", "icue", "sei", "entersect", "padeals", "acs", "e",
#                                     "getty", "images", "winamp", "lionsgate", ])

X = vectorizer.fit_transform(data_set.data)
# X = Normalizer(norm="l2", copy=False).transform(X)

# get back the terms of all training samples from Vectorizor
# terms = vectorizer.inverse_transform(X)
# print terms[0]

# # Build dictionary after vectorizer is fit
# # print vectorizer.vocabulary
vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary.iteritems(), key=itemgetter(1))])

# # Engineering feature selection
ch2 = SelectKBest(chi2, k = 85)
X = ch2.fit_transform(X, y)

# X = X.toarray()
# X = X.todense()

n_samples, n_features = X.shape
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print











###############################################################################
# Test classifier using K-fold Cross Validation

# Setup 10 fold cross validation
num_fold = n_samples # leave-one-out
# num_fold = 5
kf = KFold(n_samples, k=num_fold, indices=True)

# Note: NBs are not working
# clf = DecisionTreeClassifier(max_depth=10, min_split=2)
# clf = LDA() # not working with >2D
clf = BernoulliNB(alpha=.1)
# clf = MultinomialNB(alpha=.01)
# clf = OneVsRestClassifier(LogisticRegression(penalty='l2'))
# clf = KNeighborsClassifier(n_neighbors=3)
# clf = RidgeClassifier(tol=1e-1)
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
# clf = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
# Add Random Forest from treelearn library
# clf = treelearn.ClassifierEnsemble(bagging_percent=0.5, base_model = treelearn.RandomizedTree(), num_models=200,
#             stacking_model=SVC(C=1024, kernel='rbf', degree=3, gamma=0.001, probability=True))
# clf = treelearn.ClassifierEnsemble(bagging_percent=0.5, base_model = treelearn.ObliqueTree(max_depth=5, num_features_per_node=10), 
#             num_models=20)


# Initialize variables for couting the average
f1_all = 0.0
acc_all = 0.0
pre_all = 0.0
rec_all = 0.0

# Test for 10 rounds using the results from 10 fold cross validations
for train_index, test_index in kf:

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # fit and predict
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    # metrics
    f1_score  = metrics.f1_score(y_test, pred)
    acc_score = metrics.zero_one_score(y_test, pred)
    pre_score = metrics.precision_score(y_test, pred)
    rec_score = metrics.recall_score(y_test, pred)
    f1_all  += f1_score
    acc_all += acc_score
    pre_all += pre_score
    rec_all += rec_score

    # print data_set.target_names
    # print metrics.classification_report(y_test, pred)
    # print metrics.confusion_matrix(y_test, pred)

    # # print out top words for each category
    # for i, category in enumerate(categories):
    #             top15 = np.argsort(clf.coef_[i, :])[-5:]
    #             print "%s: %s" % (category, " ".join(vocabulary[top15]))
    #             print
    # print
    # print

f1_all  = f1_all/num_fold
acc_all = acc_all/num_fold
pre_all = pre_all/num_fold
rec_all = rec_all/num_fold

print clf
print "average f1-score:   %0.5f" % f1_all
# print "average accuracy:   %0.5f" % acc_all
print "average precision:  %0.5f" % pre_all
print "averege recall:     %0.5f" % rec_all