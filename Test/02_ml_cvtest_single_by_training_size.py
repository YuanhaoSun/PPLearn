from time import time
import numpy as np
from operator import itemgetter
from StringIO import StringIO
import math

from sklearn.datasets import load_files
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

import treelearn 



###############################################################################
# Preprocessing


# # Load from raw data
# # Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention', 'Linkout', 'California']


t0 = time()

for i in range(10, 105, 5):
    # print i

    # # Load data
    data_set = load_files('./Privacypolicy/raw'+str(i), categories = categories,
                            shuffle = True, random_state = 42)
    y = data_set.target

    # Extract features
    vectorizer = Vectorizer(max_features=10000)
    # Engineering stopword
    vectorizer.analyzer.stop_words = set([])

    X = vectorizer.fit_transform(data_set.data)
    X = Normalizer(norm="l2", copy=False).transform(X)

    # # Engineering feature selection
    # ch2 = SelectKBest(chi2, k = 125)
    # X = ch2.fit_transform(X, y)

    X = X.toarray()
    n_samples, n_features = X.shape
    # print "n_samples: %d, n_features: %d" % (n_samples, n_features)
    # print


    ###############################################################################
    # Test classifier using n run of K-fold Cross Validation

    X_orig = X
    y_orig = y

    # Note: NBs are not working
    # clf = DecisionTreeClassifier(max_depth=16, min_split=5)
    # clf = BernoulliNB(alpha=.1) # used for grading classification
    # clf = MultinomialNB(alpha=.01)
    # clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_split=1, random_state=42)
    # clf = OneVsRestClassifier(LogisticRegression(penalty='l1'))
    # nn_num = math.ceil(n_samples/30)
    # clf = KNeighborsClassifier(n_neighbors=nn_num)
    # clf = RidgeClassifier(tol=1e-1)
    # clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="l1")
    # clf = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
    clf = SVC(C=32, gamma=0.0625)
    # print clf

    num_run = 10

    # lists to hold all n*k data
    f1_total = []
    f5_total = []
    acc_total = []
    pre_total = []
    rec_total = []

    # 10 run of Kfold
    for i in range(num_run):

        X, y = shuffle(X_orig, y_orig, random_state=(i+60)) 
        # Setup 10 fold cross validation
        # num_fold = n_samples # leave-one-out
        # 0 90030, 10 89958, 20 90080, 30 90044, 40 90052, 42 90079, 50 90146, 60 90151, 70 90071, 80 89876, 90 90112
        num_fold = 10
        # num_fold = 5
        kf = KFold(n_samples, k=num_fold, indices=True)
        # Stratified is not valid in our case here, due to limited number of training sample in the smaller sample
        # kf = StratifiedKFold(y, k=num_fold, indices=True)

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

            # # Original
            f1_score = metrics.f1_score(y_test, pred)
            f5_score = metrics.fbeta_score(y_test, pred, beta=0.5)
            acc_score = metrics.zero_one_score(y_test, pred)
            pre_score = metrics.precision_score(y_test, pred)
            rec_score = metrics.recall_score(y_test, pred)

            f1_all.append(f1_score)
            f5_all.append(f5_score)
            acc_all.append(acc_score)
            pre_all.append(pre_score)
            rec_all.append(rec_score)

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

    # put the total k*n lists into numpy array for calculating the overall results
    f1_total_array  = np.asarray(f1_total)
    f5_total_array  = np.asarray(f5_total)
    acc_total_array  = np.asarray(acc_total)
    pre_total_array  = np.asarray(pre_total)
    rec_total_array  = np.asarray(rec_total)

    # Report f1 and f0.5 using the final precision and recall for consistancy
    # print "Overall precision:  %0.5f (+/- %0.2f)" % ( pre_total_array.mean(), pre_total_array.std() / 2 )
    # print "Overall recall:     %0.5f (+/- %0.2f)" % ( rec_total_array.mean(), rec_total_array.std() / 2 )
    # print "Overall f1-score:   %0.5f (+/- %0.2f)" % ( (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean()), f1_total_array.std() / 2 )
    # print "Overall f0.5-score: %0.5f (+/- %0.2f)" % ( (1.25*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+0.25*pre_total_array.mean()), f1_total_array.std() / 2 )
    print (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean())
    # print (1.25*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+0.25*pre_total_array.mean())

print (time() - t0)