from time import time
import numpy as np
from operator import itemgetter
from StringIO import StringIO 

from sklearn.datasets import load_files
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import metrics
import ml_metrics
from sklearn.externals import joblib
from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.lda import LDA
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

import treelearn 



###############################################################################
# Preprocessing


# # Load from raw data
# # Load categories
categories = ['nolimitshare','notsell', 'notsellnotshare', 'notsharemarketing', 'sellshare', 
            'shareforexception', 'shareforexceptionandconsent','shareonlyconsent']
categories3 = ['good','neutral', 'bad']
# # Load data
# print "Loading privacy policy dataset for categories:"
# print categories if categories else "all"
data_set = load_files('../Dataset/ShareStatement/raw', categories = categories,
                        shuffle = True, random_state = 42)
# data_set = load_files('../Dataset/ShareStatement3/raw', categories = categories3,
                        # shuffle = True, random_state = 42)
# print 'data loaded'
# print

# data_set = joblib.load('../Dataset/test_datasets/data_set_pos_tagged.pkl')

# load from pickle
# load data and initialize classification variables
# data_set = joblib.load('../Dataset/train_datasets/data_set_origin.pkl')
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
vectorizer.analyzer.stop_words = set([])
# vectorizer.analyzer.stop_words = set(["amazonnn", "comnn", "incnn", "emcnn", "alexann", "realnetworks", "googlenn", "googlevbp", "linkedinnn",
#                                     "foxnn", "zyngann", "eann", "yahoorb", "travelzoo", "kalturann", "2cocd", "ign", "blizzardnn",
#                                     "jobstreetcom", "surveymonkeynn", "microsoftnn", "wraljj", "spenn", "tnn", "mobile", "opendnsnns",
#                                     "bentleynn", "allvoicesnns", "watsonnn", "dynnn", "aenn", "downn", "jonesnns", "webmnn", "toysrus", "bonnierjjr",
#                                     "skypenn", "wndnn", "landrovernn", "icuenn", "seinn", "entersectnn", "padealsnns", "acsnns", "enn",
#                                     "gettynn", "imagesnns", "winampvbp", "lionsgatenn", "opendnnn", "allvoicenn", "padealnn", "imagenn",
#                                     "jonenn", "acnn", ])
# vectorizer.analyzer.stop_words = set(["amazon", "com", "inc", "emc", "alexa", "realnetworks", "google", "linkedin",
#                                     "fox", "zynga", "ea", "yahoo", "travelzoo", "kaltura", "2co", "ign", "blizzard",
#                                     "jobstreetcom", "surveymonkey", "microsoft", "wral", "spe", "t", "mobile", "opendns",
#                                     "bentley", "allvoices", "watson", "dyn", "ae", "dow", "jones", "webm", "toysrus", "bonnier",
#                                     "skype", "wnd", "landrover", "icue", "sei", "entersect", "padeals", "acs", "e",
#                                     "getty", "images", "winamp", "lionsgate", "opendn", "allvoice", "padeal", "image",
#                                     "getti", "gett", "jone", "ac"])
# vectorizer.analyzer.stop_words = set(['as', 'of', 'in', 'you', 'rent', 'we', 'the', 'sell', 'parties', 'we', 'with', 'not', 'personal',
#                                     'third', 'to', 'share', 'your', 'information', 'or', ]) #threshold 20 on training set
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
# ch2 = SelectKBest(chi2, k = 125)
# X = ch2.fit_transform(X, y)

X = X.toarray()
# X = X.todense()

n_samples, n_features = X.shape
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print





###############################################################################
# Test classifier using n run of K-fold Cross Validation


X_orig = X
y_orig = y

# Note: NBs are not working
# clf = DecisionTreeClassifier(max_depth=10, min_split=2)
# clf = LDA() # not working with >2D
clf = BernoulliNB(alpha=.1) # used for grading classification
# clf = MultinomialNB(alpha=.01)
# clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_split=1, random_state=42)
# clf = OneVsRestClassifier(LogisticRegression(penalty='l1'))
# clf = KNeighborsClassifier(n_neighbors=2)
# clf = RidgeClassifier(tol=1e-1)
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="l1")
# clf = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
# Add Random Forest from treelearn library
# clf = treelearn.ClassifierEnsemble(bagging_percent=0.5, base_model = treelearn.RandomizedTree(), num_models=200,
#             stacking_model=SVC(C=1024, kernel='rbf', degree=3, gamma=0.001, probability=True))
# clf = treelearn.ClassifierEnsemble(bagging_percent=0.5, base_model = treelearn.ObliqueTree(max_depth=5, num_features_per_node=10), 
#             num_models=20)
print clf

num_run = 10

# lists to hold all n*k data
f1_total = []
f5_total = []
acc_total = []
pre_total = []
rec_total = []

# 50 run of Kfold
for i in range(num_run):

    X, y = shuffle(X_orig, y_orig, random_state=(i+50))
    # Setup 10 fold cross validation
    # num_fold = n_samples # leave-one-out
    num_fold = 10
    # num_fold = 5
    kf = KFold(n_samples, k=num_fold, indices=True)
    # Stratified is not valid in our case here, due to limited number of training sample in the smaller sample
    # kf = StratifiedKFold(y, k=num_fold, indices=True)
    # 10* ShuffleSplit validation - this is another way to do the n*k
    # For CV on raw (8 categories) -- 6, 7, 8, 11, 12, 13, 25, 26, 33, 34, 35
    # kf = ShuffleSplit(n_samples, n_iterations=10, test_fraction=0.1, indices=True, random_state=26)

    # Initialize variables for couting the average
    f1_all = []
    f5_all = []
    acc_all = []
    pre_all = []
    rec_all = []

    # Test for 10 rounds using the results from 10 fold cross validations
    for train_index, test_index in kf:

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit and predict
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        # print y_test
        # print pred
        # print type(pred)

        # output tree into graph
        # out = StringIO()
        # out = export_graphviz(clf, out_file=out)

        # metrics
        # # Mic
        # f1_score = ml_metrics.f1_score(y_test, pred)
        # f5_score = ml_metrics.fbeta_score(y_test, pred, beta=0.5)
        # acc_score = ml_metrics.precision_score(y_test, pred)
        # pre_score = ml_metrics.precision_score(y_test, pred)
        # rec_score = ml_metrics.recall_score(y_test, pred)
        # # Original
        f1_score = metrics.f1_score(y_test, pred)
        f5_score = metrics.fbeta_score(y_test, pred, beta=0.5)
        acc_score = metrics.zero_one_score(y_test, pred)
        pre_score = metrics.precision_score(y_test, pred)
        rec_score = metrics.recall_score(y_test, pred)
        # # Wei
        # f1_score = ml_metrics.f1_weighted_score(y_test, pred)
        # f5_score = ml_metrics.fbeta_weighted_score(y_test, pred, beta=0.5)
        # acc_score = ml_metrics.precision_weighted_score(y_test, pred)
        # pre_score = ml_metrics.precision_weighted_score(y_test, pred)
        # rec_score = ml_metrics.recall_weighted_score(y_test, pred)
        # Mac
        # f1_score = ml_metrics.f1_macro_score(y_test, pred)
        # f5_score = ml_metrics.fbeta_macro_score(y_test, pred, beta=0.5)
        # acc_score = ml_metrics.precision_macro_score(y_test, pred)
        # pre_score = ml_metrics.precision_macro_score(y_test, pred)
        # rec_score = ml_metrics.recall_macro_score(y_test, pred)

        f1_all.append(f1_score)
        f5_all.append(f5_score)
        acc_all.append(acc_score)
        pre_all.append(pre_score)
        rec_all.append(rec_score)

        # print "f1",f1_score
        # print "f5",f5_score
        # print "ac",acc_score
        # print "pr",pre_score
        # print "re",rec_score
        # print
        # # This can get the category measures.
        # pre,rec,f1,support = metrics.precision_recall_fscore_support(y_test, pred)
        # print "f1",f1
        # print "pr",pre
        # print "re",rec
        # print


        # print data_set.target_names
        # print "F1:       ", f1_score
        # print "Precision:", pre_score
        # print "Recall:   ", rec_score
        # print metrics.classification_report(y_test, pred)
        # print metrics.confusion_matrix(y_test, pred)


        # # print out top words for each category
        # for i, category in enumerate(categories):
        #             top = np.argsort(clf.coef_[i, :])[-50:]
        #             print "%s: %s" % (category, " ".join(vocabulary[top]))
        #             print
        # print
        # print

    # put the lists into numpy array for calculating the results
    f1_all_array  = np.asarray(f1_all)
    f5_all_array  = np.asarray(f5_all)
    acc_all_array  = np.asarray(acc_all)
    pre_all_array  = np.asarray(pre_all)
    rec_all_array  = np.asarray(rec_all)

    # add to the total k*n data set
    f1_total += f1_all
    f5_total += f5_all
    acc_total += acc_all
    pre_total += pre_all
    rec_total += rec_all

    # Result for each run
    # print "average f1-score:   %0.5f (+/- %0.2f)" % ( f1_all_array.mean(), f1_all_array.std() / 2 )
    # print "average f0.5-score: %0.5f (+/- %0.2f)" % ( f5_all_array.mean(), f1_all_array.std() / 2 )
    # print "average accuracy:   %0.5f (+/- %0.2f)" % ( acc_all_array.mean(), acc_all_array.std() / 2 )
    # print "average precision:  %0.5f (+/- %0.2f)" % ( pre_all_array.mean(), pre_all_array.std() / 2 )
    # print "averege recall:     %0.5f (+/- %0.2f)" % ( rec_all_array.mean(), rec_all_array.std() / 2 )


# put the total k*n lists into numpy array for calculating the overall results
f1_total_array  = np.asarray(f1_total)
f5_total_array  = np.asarray(f5_total)
acc_total_array  = np.asarray(acc_total)
pre_total_array  = np.asarray(pre_total)
rec_total_array  = np.asarray(rec_total)

# Report f1 and f0.5 using the final precision and recall for consistancy
# print "Overall f1-score:   %0.5f (+/- %0.2f)" % ( f1_total_array.mean(), f1_total_array.std() / 2 )
# print "Overall f0.5-score: %0.5f (+/- %0.2f)" % ( f5_total_array.mean(), f5_total_array.std() / 2 )
# print "Overall accuracy:   %0.5f (+/- %0.2f)" % ( acc_total_array.mean(), acc_total_array.std() / 2 )
print "Overall precision:  %0.5f (+/- %0.2f)" % ( pre_total_array.mean(), pre_total_array.std() / 2 )
print "Overall recall:     %0.5f (+/- %0.2f)" % ( rec_total_array.mean(), rec_total_array.std() / 2 )
# print (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean())
print "Overall f1-score:   %0.5f (+/- %0.2f)" % ( (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean()), f1_total_array.std() / 2 )
print "Overall f0.5-score: %0.5f (+/- %0.2f)" % ( (1.25*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+0.25*pre_total_array.mean()), f1_total_array.std() / 2 )
