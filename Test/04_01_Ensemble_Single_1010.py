"""
======================================================
Ensemble: Stacking
======================================================
Ensemble learning
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


def filter_negative(array_2d):
    for i, row in enumerate(array_2d):
        for j, item in enumerate(row):
            if item < 0:
                array_2d[i,j] = 0

t0 = time()

###############################################################################
# Preprocessing

# Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention', 'Linkout', 'California']

# Load data
data_set = load_files('../Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
y = data_set.target

# Extract features
vectorizer = Vectorizer(max_features=10000)
vectorizer.analyzer.stop_words = set([])

X = vectorizer.fit_transform(data_set.data)
# X = Normalizer(norm="l2", copy=False).transform(X)

# # Feature selection
ch2 = SelectKBest(chi2, k = 1150)
X = ch2.fit_transform(X, y)


# X = X.toarray()
X_den = X.toarray()
n_samples, n_features = X.shape

###############################################################################
# Test classifier using n run of K-fold Cross Validation

X_den_orig = X_den
X_orig = X
y_orig = y

num_run = 50

# lists to hold all n*k data
f1_total = []
f5_total = []
acc_total = []
pre_total = []
rec_total = []


# clf = DecisionTreeClassifier(max_depth=14, min_split=5)
# clf = BernoulliNB(alpha=.1) # used for grading classification
# clf = MultinomialNB(alpha=.01)
# clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_split=1, random_state=42)
# clf = OneVsRestClassifier(LogisticRegression(penalty='l1'))
# nn_num = math.ceil(n_samples/30)
# clf = KNeighborsClassifier(n_neighbors=19)
# clf = RidgeClassifier(tol=1e-1)
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="l1")
# clf = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
# clf = SVC(C=1000, kernel='rbf', degree=3, gamma=0.001)
clf = SVC(C=32, gamma=0.0625)
# print clf


# 10 run of Kfold
for i in range(num_run):

    X, X_den, y = shuffle(X_orig, X_den_orig, y_orig, random_state=(i+60)) 

    fold_num_l1 = 10
    kf1 = KFold(n_samples, k=fold_num_l1, indices=True)

    # Initialize variables for couting the average
    # f1_all = 0.0
    # acc_all = 0.0
    # pre_all = 0.0
    # rec_all = 0.0
    f1_all = []
    f5_all = []
    acc_all = []
    pre_all = []
    rec_all = []

    # level 1 evaluation
    for train_index, test_index in kf1:

        X_den_train, X_den_test = X_den[train_index], X_den[test_index]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # clf.fit(X_den_train, y_train)
        # pred = clf.predict(X_den_test)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        # metrics
        acc_score = metrics.zero_one_score(y_test, pred)
        pre_score = metrics.precision_score(y_test, pred)
        rec_score = metrics.recall_score(y_test, pred)

        acc_all.append(acc_score)
        pre_all.append(pre_score)
        rec_all.append(rec_score)

    # put the lists into numpy array for calculating the results
    acc_all_array  = np.asarray(acc_all)
    pre_all_array  = np.asarray(pre_all)
    rec_all_array  = np.asarray(rec_all)

    # add to the total k*n data set
    acc_total += acc_all
    pre_total += pre_all
    rec_total += rec_all

    # Result for each run
    # print  "%f\t%f" % ( (1.25*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+0.25*pre_total_array.mean()), (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean()) )
    print  "%f\t%f" % ( (1.25*pre_all_array.mean()*rec_all_array.mean())/(rec_all_array.mean()+0.25*pre_all_array.mean()), (2*pre_all_array.mean()*rec_all_array.mean())/(rec_all_array.mean()+pre_all_array.mean()) )

# put the total k*n lists into numpy array for calculating the overall results
acc_total_array  = np.asarray(acc_total)
pre_total_array  = np.asarray(pre_total)
rec_total_array  = np.asarray(rec_total)

# Report f1 and f0.5 using the final precision and recall for consistancy
print "Overall precision:  %0.5f (+/- %0.5f)" % ( pre_total_array.mean(), pre_total_array.std() / 2 )
print "Overall recall:     %0.5f (+/- %0.5f)" % ( rec_total_array.mean(), rec_total_array.std() / 2 )
# print (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean())
print "Overall f1-score:   %0.5f" % ( (2*pre_total_array.mean()*rec_total_array.mean()) / (rec_total_array.mean()+pre_total_array.mean()) )
print "Overall f0.5-score: %0.5f" % ( (1.25*pre_total_array.mean()*rec_total_array.mean()) / (rec_total_array.mean()+0.25*pre_total_array.mean()) )

print (time() - t0)