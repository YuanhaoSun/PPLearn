"""
======================================================
Test classifiers using feature selection
======================================================
Test classifiers using feature selection with
cross validation
"""

print __doc__

import csv

from time import time
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics
from sklearn.utils import check_arrays

from sklearn.cross_validation import KFold


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

X_orgin = X

y = data_set.target


# Make a classifier list
clfs = []
# Note: NBs are not working
clfs.append(BernoulliNB(alpha=.01))
clfs.append(MultinomialNB(alpha=.01))
clfs.append(KNeighborsClassifier(n_neighbors=13))
clfs.append(RidgeClassifier(tol=1e-1))
clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l1"))
clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"))
clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"))
clfs.append(LinearSVC(loss='l2', penalty='l1', C=1000, dual=False, tol=1e-3))
clfs.append(LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3))
clfs.append(SVC(C=1000))

# Treat logistic regression specially due to requirement for array input
clfs_reg = []
clfs_reg.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l1')))
clfs_reg.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l2')))

# avoid repeat in file names
i = 0

for clf in clfs:

    # define classifier
    clf_name = str(clf)[0:10]

    # avoid repeat in file names
    i += 1
    # write results to csv file
    csvWriter = csv.writer(open('feature_tests/' + str(i) + clf_name + ".csv", 'wb'))
    # head raw
    csvWriter.writerow(['Feature#', 'F1', 'Precision', 'Recall'])

    # Feature selection tests
    for select_chi2 in range(100, 2600, 100):

        # print ("Extracting %d best features by a chi-squared test" % select_chi2)
        # t0 = time()
        X = X_orgin
        ch2 = SelectKBest(chi2, k = select_chi2)
        X = ch2.fit_transform(X, y)
        # print "Done in %fs" % (time() - t0)
        # print "L1:      n_samples: %d, n_features: %d" % X.shape
        # print

        n_samples, n_features = X.shape
        # print "done in %fs" % (time() - t0)
        # print "n_samples: %d, n_features: %d" % (n_samples, n_features)
        # print


        ###############################################################################
        # Test a classifier using K-fold Cross Validation

        # k fold's k
        num_fold = 10
        
        # Setup 10 fold cross validation
        kf = KFold(n_samples, k=num_fold, indices=True)

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

            # Could also use the scikit-learn API for precision and recall
            pre_score = metrics.precision_score(y_test, pred)
            rec_score = metrics.recall_score(y_test, pred)
            # pre_score = class_metrics(y_test, pred, pre=1)
            pre_all += pre_score
            # rec_score = class_metrics(y_test, pred, rec=1)
            rec_all += rec_score

        f1_all = f1_all/num_fold
        acc_all = acc_all/num_fold
        pre_all = pre_all/num_fold
        rec_all = rec_all/num_fold

        # write raw of metrics details to csv file
        csvWriter.writerow([select_chi2, f1_all, pre_all, rec_all])

        # average the metrics from 10 fold
        print select_chi2
        print "average f1-score:   %0.5f" % f1_all
        print "average precision:  %0.5f" % pre_all
        print "averege recall:     %0.5f" % rec_all
        print