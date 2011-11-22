from time import time
import numpy as np

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


###############################################################################
# Preprocessing


# Load categories
categories = ['nolimitshare','notsell', 'notsellnotshare', 'sellshare', 'shareforexception', 
            'shareforexceptionandconsent','shareonlyconsent',]


# Load data
print "Loading privacy policy dataset for categories:"
print categories if categories else "all"
data_set = load_files('ShareStatement/raw', categories = categories,
                        shuffle = True, random_state = 42)
print 'data loaded'
print

# Extract features
print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_features=10000)
# # Remove Using Stopword
vectorizer.analyzer.stop_words = set([])
# vectorizer.analyzer.stop_words = set(["we", "do", "you", "your", "the", "that", "this", 
#                                     "is", "was", "are", "were", "being", "be", "been",
#                                     "for", "of", "as", "in",  "to", "at", "by",
#                                     # "or", "and",
#                                     "ve",
#                                     "amazon", "com", "inc", "emc", "alexa", "realnetworks", "google", "linkedin",
#                                     "fox", "zynga", "ea", "yahoo", "travelzoo", "kaltura", "2co", ])
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
ch2 = SelectKBest(chi2, k = 95)
X = ch2.fit_transform(X, y)


X_den = X.toarray()

n_samples, n_features = X.shape
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print


###############################################################################
# Test a classifier using K-fold Cross Validation

# Setup 10 fold cross validation
fold_num = 3
kf = KFold(n_samples, k=fold_num, indices=True)

#set number of neighbors for kNN
n_neighb = 3


# Make a classifier list
clfs = []
# Note: NBs are not working
clfs.append(MultinomialNB(alpha=.01))
clfs.append(BernoulliNB(alpha=.01))
clfs.append(KNeighborsClassifier(n_neighbors=13))
clfs.append(RidgeClassifier(tol=1e-1))
clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l1"))
clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"))
clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"))
clfs.append(LinearSVC(loss='l2', penalty='l1', C=1000, dual=False, tol=1e-3))
clfs.append(LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3))
clfs.append(SVC(C=1000))
clfs.append(SVC(C=1000,kernel="linear"))


# Treat logistic regression specially due to requirement for array input
clfs_reg = []
clfs_reg.append(DecisionTreeClassifier(min_split=6))
clfs_reg.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l1')))
clfs_reg.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l2')))


for clf in clfs:

    # Initialize variables for couting the average
    f1_all = 0.0
    acc_all = 0.0
    pre_all = 0.0
    rec_all = 0.0

    # Test for 10 rounds using the results from 10 fold cross validations
    for train_index, test_index in kf:

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

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


for clf in clfs_reg:

    # Initialize variables for couting the average
    f1_all = 0.0
    acc_all = 0.0
    pre_all = 0.0
    rec_all = 0.0

    # Test for 10 rounds using the results from 10 fold cross validations
    for train_index, test_index in kf:

        X_train, X_test = X_den[train_index], X_den[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        #Scores
        f1_score = metrics.f1_score(y_test, pred)
        f1_all += f1_score

        # change the calculateion of accuracy
        # acc_score = class_metrics(y_test, pred, acc=1)
        acc_score  = np.mean(pred.ravel() == y_test.ravel()) * 100
        acc_all += acc_score
        
        pre_score = metrics.precision_score(y_test, pred)
        rec_score = metrics.recall_score(y_test, pred)
        pre_all += pre_score
        rec_all += rec_score

    # average the metrics from 10 fold
    f1_all = f1_all/fold_num
    acc_all = acc_all/fold_num
    pre_all = pre_all/fold_num
    rec_all = rec_all/fold_num

    print clf
    print "average f1-score:   %0.5f" % f1_all
    print "average precision:  %0.5f" % pre_all
    print "averege recall:     %0.5f" % rec_all
    # changed accuracy calculation
    # NOT APPLICABLE any more: accuracy is not a good metrics
    #                   because the sparse nature of matrix
    # print "average accuracy:   %0.5f" % acc_all
    print