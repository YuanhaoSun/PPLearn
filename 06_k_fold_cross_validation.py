"""
======================================================
Classificatio with CrossValidation
======================================================
Using CrossValidation to get the fair accuracy of algorithms.
"""

print __doc__

from time import time
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer

from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.svm.sparse import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics
from sklearn.utils import check_arrays

from sklearn.cross_validation import KFold


###############################################################################
# Axiliary function for the function below

def unique_labels(*list_of_labels):
    """Extract an ordered integer array of unique labels

    This implementation ignores any occurrence of NaNs.

    """
    list_of_labels = [np.unique(labels[np.isfinite(labels)].ravel())
                      for labels in list_of_labels]
    list_of_labels = np.concatenate(list_of_labels)
    return np.unique(list_of_labels)


###############################################################################
# Function to compute accuracy which is not provided in API from scikit-learn
# Derived from 'sklearn.metrics.precision_recall_fscore_support'
# Usage: class_metrics(y_true, y_pred, [acc=1] or [pre=1] or [rec=1])

def class_metrics(y_true, y_pred, acc=None, pre=None, rec=None, beta=1.0, labels=None):

    y_true, y_pred = check_arrays(y_true, y_pred)
    assert(beta > 0)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    n_labels = labels.size
    # add true_neg
    true_neg = np.zeros(n_labels, dtype=np.double)
    true_pos = np.zeros(n_labels, dtype=np.double)
    false_pos = np.zeros(n_labels, dtype=np.double)
    false_neg = np.zeros(n_labels, dtype=np.double)
    support = np.zeros(n_labels, dtype=np.long)

    for i, label_i in enumerate(labels):
        # add true_neg[i]
        true_neg[i] = np.sum(y_pred[y_true != label_i] != label_i)
        true_pos[i] = np.sum(y_pred[y_true == label_i] == label_i)
        false_pos[i] = np.sum(y_pred[y_true != label_i] == label_i)
        false_neg[i] = np.sum(y_pred[y_true == label_i] != label_i)
        support[i] = np.sum(y_true == label_i)

    try:
        # oddly, we may get an "invalid" rather than a "divide" error here
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        # accuracy
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        # handle division by 0.0 in precision and recall
        # also for accuracy
        accuracy[(true_pos + true_neg + false_pos + false_neg) == 0.0] = 0.0
        precision[(true_pos + false_pos) == 0.0] = 0.0
        recall[(true_pos + false_neg) == 0.0] = 0.0

        # Just a test when modifying this function (the returns below)
        # print accuracy.shape[0]

    finally:
        np.seterr(**old_err_settings)

    if acc == 1:
        # As defined in fbeta_score() in sklearn-metrics source code
        if accuracy.shape[0] == 2:
            return accuracy[1]
        else:
            return np.average(accuracy, weights=support)
    elif pre == 1:
        if precision.shape[0] == 2:
            return precision[1]
        else:
            return np.average(precision, weights=support)
    elif rec == 1:
        if recall.shape[0] == 2:
            return recal[1]
        else:
            return np.average(recall, weights=support)     



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
print "%d documents" % len(data_set.data)
print "%d categories" % len(data_set.target_names)
print

# Extract features
print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_features=10000)
X = vectorizer.fit_transform(data_set.data)
X = Normalizer(norm="l2", copy=False).transform(X)
print type(X)
X = X.todense()
print type(X)
y = data_set.target
n_samples, n_features = X.shape
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print


###############################################################################
# Test a classifier using K-fold Cross Validation

# Setup 10 fold cross validation
kf = KFold(n_samples, k=10)

# Note: NBs are not working
# clf = BernoulliNB(alpha=.1)
# clf = MultinomialNB(alpha=.01)
# clf = OneVsRestClassifier(LogisticRegression(penalty='l2'))
# clf = KNeighborsClassifier(n_neighbors=13)
clf = RidgeClassifier(tol=1e-1)
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")
# clf = LinearSVC(loss='l2', penalty='l1', C=1000, dual=False, tol=1e-3)

# Initialize variables for couting the average
f1_all = 0.0
acc_all = 0.0
pre_all = 0.0
rec_all = 0.0

# Test for 10 rounds using the results from 10 fold cross validations
for train_index, test_index in kf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print "Training: "
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    #Scores
    f1_score = metrics.f1_score(y_test, pred)
    f1_all += f1_score
    print "f1-score:   %0.3f" % f1_score
    acc_score = class_metrics(y_test, pred, acc=1)
    acc_all += acc_score
    print "accuracy:   %0.3f" % acc_score

    # Could also use the scikit-learn API for precision and recall
    # pre_s = metrics.precision_score(y_test, pred)
    # print "precision api:   %0.3f" % pre_s
    # rec_s = metrics.recall_score(y_test, pred)
    # print "precision api:   %0.3f" % rec_s
    pre_score = class_metrics(y_test, pred, pre=1)
    pre_all += pre_score
    print "precision:   %0.3f" % pre_score
    rec_score = class_metrics(y_test, pred, rec=1)
    rec_all += rec_score
    print "recall:   %0.3f" % rec_score

f1_all = f1_all/10.0
acc_all = acc_all/10.0
pre_all = pre_all/10.0
rec_all = rec_all/10.0

print
print "average f1-score:   %0.5f" % f1_all
print "average accuracy:   %0.5f" % acc_all
print "average precision:  %0.5f" % pre_all
print "averege recall:     %0.5f" % rec_all