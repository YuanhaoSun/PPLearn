"""
======================================================
Tuning parameters for classifiers
======================================================
Tuning parameters for classifiers using grid search
"""

print __doc__

from time import time
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV


###############################################################################
# Preprocessing

# Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor', 'Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention']

# Load data
print "Loading privacy policy dataset for categories:"
print categories if categories else "all"
data_set = load_files('Privacypolicy_balance/raw', categories = categories,
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

# X = X.toarray()

y = data_set.target
n_samples, n_features = X.shape
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print

# split the dataset in two equal part respecting label proportions
train, test = iter(StratifiedKFold(y, 10, indices=True)).next()

################################################################################
# Grid search with k-fold cross validation to tune parameters for SVC

# parameters for SVC
# right way to tune SVC - 'A Practical Guide to SVC'
# loose search
# Results -- gamma=2**-5, C=2**5
# tuned_parameters = [{'kernel': ['rbf'],
#                      'gamma': [2**-15, 2**-11, 2**-7, 2**-5, 2**-3, 2**-1],
#                      'C': [2**-3, 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]}]
# fine search
# tuned_parameters = [{'kernel': ['rbf'], 
#                      'gamma': [2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4],
#                      'C': [2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10]}]

# parameters for LinearSVC
# tuned_parameters = [{'loss': ['l2'],
#                      'penalty': ['l1', 'l2'],
#                      'C': [2**-3, 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]}]

# parameters for Tree
# tuned_parameters = [{'max_depth': ['8', '10', '12', '15', '18'],
#                      'min_split': ['3', '4', '5', '6', '7', '8', '9']}]

# parameters for kNN
# tuned_parameters = [{'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 
#                                      23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
#                                      43, 45, 47, 49, 51, 53, 55]}]

# rNN's not working
# parameters for rNN
# tuned_parameters = [{'radius': [1.0]}]

# parameters for Ridge
# loose search
# tuned_parameters = [{'alpha': [2**-7, 2**-5, 2**-3, 2**-1, 1, 2, 2**3, 2**5, 2**7]}]
# fine search
# tuned_parameters = [{'alpha': [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]}]

# Not yet working. Need to figure out how to pass para into Logistic through 1vsRest wrapper
# parameters for Logistic Regression
# tuned_parameters = [{'penalty': ['l1', 'l2'],
#                      'C': [2**-3, 2**-1, 2, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]}]



# used metrics for fine tuning
scores = [
    # ('precision', metrics.precision_score),
    # ('recall', metrics.recall_score),
    ('f1', metrics.f1_score),
    # ('accuracy', metrics.zero_one_score)
]

# carry out grid search with CV
for score_name, score_func in scores:
    
    t0 = time()

    # grid search
    # Tree is not working correctly, given strangely small numbers
    # clf = GridSearchCV(DecisionTreeClassifier(),tuned_parameters, score_func=score_func)
    # clf = GridSearchCV(SVC(C=1), tuned_parameters, score_func=score_func)
    # clf = GridSearchCV(LinearSVC(C=1, dual=False), tuned_parameters, score_func=score_func)
    # clf = GridSearchCV(KNeighborsClassifier(n_neighbors=1), 
    #                     tuned_parameters, score_func=score_func)
    # rNN's not working
    # clf = GridSearchCV(RadiusNeighborsClassifier(radius=1.0), 
    #                     tuned_parameters, score_func=score_func)
    # clf = GridSearchCV(RidgeClassifier(alpha=1.0), tuned_parameters, score_func=score_func)
    # clf = GridSearchCV(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l1')),
    #                     tuned_parameters, score_func=score_func)

    clf.fit(X[train], y[train], cv=StratifiedKFold(y[train], 5))
    
    # report
    print "one run of grid search done in %fs" % (time() - t0)
    
    y_true, y_pred = y[test], clf.predict(X[test])

    print "Classification report for the best estimator: "
    print clf.best_estimator
    print "Tuned for '%s' with optimal value: %0.3f" % (
        score_name, score_func(y_true, y_pred))
    print
    print metrics.classification_report(y_true, y_pred)
    print "Grid scores:"
    for item in clf.grid_scores_:
        print item
    print



# ###############################################################################
# # Test a classifier using K-fold Cross Validation

# # Setup 10 fold cross validation
# # kf = KFold(n_samples, k=10)
# kf = KFold(n_samples, k=10, indices=True)

# # Make a classifier list
# clfs = []
# # Note: NBs are not working
# clfs.append(BernoulliNB(alpha=.01))
# clfs.append(MultinomialNB(alpha=.01))
# clfs.append(KNeighborsClassifier(n_neighbors=n_neighb))
# clfs.append(RidgeClassifier(tol=1e-1))
# clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l1"))
# clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="l2"))
# clfs.append(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet"))
# clfs.append(LinearSVC(loss='l2', penalty='l1', C=1000, dual=False, tol=1e-3))
# clfs.append(LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3))
# clfs.append(SVC(C=1000))

# # Treat logistic regression specially due to requirement for array input
# clfs_reg = []
# clfs_reg.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l1')))
# clfs_reg.append(OneVsRestClassifier(LogisticRegression(C=1000,penalty='l2')))


# print "parameters:"
# pprint(parameters)
# t0 = time()
# grid_search.fit(text_docs, data.target)
# print "done in %0.3fs" % (time() - t0)
# print