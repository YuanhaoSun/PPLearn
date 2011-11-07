"""
======================================================
EM Classifier
======================================================
EM
"""

print __doc__

# Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
# License: BSD Style.

# $Id$

from time import time
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer

from sklearn.cross_validation import StratifiedKFold
from sklearn.mixture import GMM


# Load categories
# categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
#             'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
#             'Process', 'Retention']
# categories = ['Change', 'Location', 'Children', 'Contact', 'Process', 'Retention']
categories = ['CA', 'Collect']

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
print "Extracting features from the training dataset..."
vectorizer = Vectorizer(max_features=10000)
X = vectorizer.fit_transform(data_set.data)
X = Normalizer(norm="l2", copy=False).transform(X)

X = X.toarray()

y = data_set.target

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(y, k=4, indices=True)
# Only take the first fold.
train_index, test_index = skf.__iter__().next()

# datasets
X_train = X[train_index]
y_train = y[train_index]
X_test = X[test_index]
y_test = y[test_index]

print X_train.shape
print len(y_train)
print X_test.shape
print len(y_test)

n_classes = len(np.unique(y_train))
print n_classes

# Try GMMs using different types of covariances.
classifiers = dict((x, GMM(n_components=n_classes, cvtype=x))
                    for x in ['spherical', 'full'])
                    # for x in ['spherical', 'tied', 'full'])
                    # for x in ['spherical', 'tied'])
                    # for x in ['spherical', 'diag', 'tied', 'full'])

n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.iteritems()):
    
    print name

    t0 = time()
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    classifier.means = [X_train[y_train == i, :].mean(axis=0)
                        for i in xrange(n_classes)]

    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train, init_params='wc', n_iter=20)

    y_train_pred = classifier.predict(X_train)
    train_accuracy  = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print 'Train accuracy: %.2f' % train_accuracy

    y_test_pred = classifier.predict(X_test)
    test_accuracy  = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    print 'Test accuracy: %.2f' % test_accuracy

    print "done in %fs" % (time() - t0)

print 'done'