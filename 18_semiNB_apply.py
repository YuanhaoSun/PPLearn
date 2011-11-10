import numpy as np
from time import time

from Class_semiNB import MultinomialNaiveBayes
from Class_semiNB import ComplementNaiveBayes
from Class_semiNB import generate_classifier, to_1_of_K

from sklearn.naive_bayes import MultinomialNB


from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer


from sklearn import metrics
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

# load unlabeled data
data_set_unlabel = load_files('Privacypolicy/unlabeled', shuffle = True, random_state = 30)


# Extract features
print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_features=10000)
X = vectorizer.fit_transform(data_set.data)
X = Normalizer(norm="l2", copy=False).transform(X)
X = X.toarray()

X_unlabel = vectorizer.transform(data_set_unlabel.data)
X_unlabel = X_unlabel.toarray()

y = data_set.target

n_samples, n_features = X.shape
print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % (n_samples, n_features)
print


def _test_semi(NaiveBayes):


    clf = MultinomialNB(alpha=.01)
    clf.fit(X_train, y_train)
    # clf.fit(X_train[:n], Y_train[:n])
    pred = clf.predict(X_test)
    f1_score = metrics.f1_score(y_test, pred)
    print "Supervised Learning - scikits-learn 0.10"
    # print "Accuracy: %d/%d" % (np.sum(pred == y_test), len(y_test))
    accuracy =  float( np.sum(pred == y_test) )  / float(len(y_test))
    print "Accuracy: %0.4f" % accuracy
    print "F1: %0.4f" % f1_score


    clf = NaiveBayes()
    clf.fit(X_train, y_train)
    # clf.fit(X_train[:n], Y_train[:n])
    pred = clf.predict(X_test)
    f1_score = metrics.f1_score(y_test, pred)
    print "Supervised Learning"
    # print "Accuracy: %d/%d" % (np.sum(pred == y_test), len(y_test))
    accuracy =  float( np.sum(pred == y_test) )  / float(len(y_test))
    print "Accuracy: %0.4f" % accuracy
    print "F1: %0.4f" % f1_score

    clf = NaiveBayes()
    clf.fit_semi(X_train, Y_train, X_unlabel)
    # clf.fit_semi(X_train[:n], Y_train[:n], X_train[n:])
    pred = clf.predict(X_test)
    f1_score = metrics.f1_score(y_test, pred)
    accuracy =  float( np.sum(pred == y_test) )  / float(len(y_test))
    print "Semi-Supervised Learning"
    print "Accuracy: %0.4f" % accuracy
    print "F1: %0.4f" % f1_score
        
    print "-----"

def test_semi_multinomial():
    print "Multinomial Naive Bayes"
    _test_semi(MultinomialNaiveBayes)

def test_semi_complement():
    print "Complement Naive Bayes"
    _test_semi(ComplementNaiveBayes)



kf = KFold(n_samples, k=10, indices=True)

for train_index, test_index in kf:

    print len(train_index)
    print len(test_index)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    Y_train = to_1_of_K(y_train)

    test_semi_multinomial()