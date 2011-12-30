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
from sklearn.externals import joblib
from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.lda import LDA
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier

###############################################################################
# Preprocessing


# # Load from raw data
# # Load categories
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention', 'Linkout', 'California']
# # Load data
data_set = load_files('../Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
y = data_set.target

# Extract features
vectorizer = Vectorizer(max_features=10000)

# Engineering stopword
vectorizer.analyzer.stop_words = set([])

X = vectorizer.fit_transform(data_set.data)
# X = Normalizer(norm="l2", copy=False).transform(X)

# X = X.toarray()

# Note: NBs are not working
# clf = MultinomialNB(alpha=.01)
# clf = KNeighborsClassifier(n_neighbors=19)
# clf = RidgeClassifier(tol=1e-1)
clf = LinearSVC(loss='l2', penalty='l2', C=0.5, dual=False, tol=1e-3)
# clf = SVC(C=32, gamma=0.0625)
# clf = OneVsRestClassifier(LogisticRegression(penalty='l1'))
# clf = DecisionTreeClassifier(max_depth=10, min_split=2)
# clf = SGDClassifier(alpha=.0001, n_iter=50, penalty="l1")
# Add Random Forest from treelearn library
# clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_split=1, random_state=42)
print clf


num_total_features = X.shape[1]
X_real_origin = X
y_real_origin = y

# Iterate for steps while selected feature number increasing (from 2% to 100% in 50 steps)
for i in range(50):

    # Engineering feature selection
    ch2 = SelectKBest(chi2, k = num_total_features*(i+1)*0.02)
    X = ch2.fit_transform(X_real_origin, y_real_origin)

    n_samples, n_features = X.shape
    # print n_features

    ###############################################################################
    # Test classifier using n run of K-fold Cross Validation
    X_orig = X
    y_orig = y_real_origin

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

            # metrics
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
    # print (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean())
    # print (1.25*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+0.25*pre_total_array.mean())
    print  "%f\t%f" % ( (1.25*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+0.25*pre_total_array.mean()), (2*pre_total_array.mean()*rec_total_array.mean())/(rec_total_array.mean()+pre_total_array.mean()) )