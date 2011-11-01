"""
======================================================
Classification - Multilabel
======================================================
Implementing the multilabel classifiers
"""

import logging
import numpy as np
from operator import itemgetter
from optparse import OptionParser
import sys
from time import time

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.utils.extmath import density

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn import metrics

# Deaclear categories
# categories = ['SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
#             'Process', 'Retention']
categories = ['Advertising','CA', 'Collect', 'Cookies']
# categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
#             'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
#             'Process', 'Retention']

print __doc__

print '# categories: %d' % len(categories)


# Load all training datasets
print "Loading privacy policy datasets..."
data_train = load_files('Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
print 'Data loaded!'
print

y = data_train.target
print "length of y: %d" % len(y)

# A primary thought on implementing multi-label classifier
# Aborted later due to functions provided by most classifiers
# Method: Transform y to one-else and use loops to learn binary classifiers
# y_0 = y.copy()
# for i in range(len(y_0)):
#       if y_0[i] == 1:
#             y_0[i] = 2


# Extract features
print "Extracting features from Layer 1 training set using a sparse vectorizer..."
t0 = time()
vectorizer = Vectorizer()
X = vectorizer.fit_transform(data_train.data)
print "Done in %0.3fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape
print
# to dense array for logistic regression which does not work on sparse
X_den = X.toarray()


# # Feature selection for the L1 dataset
# select_chi2 = 1000
# print ("Extracting %d best features by a chi-squared test" % select_chi2)
# t0 = time()
# ch2 = SelectKBest(chi2, k = select_chi2)
# X = ch2.fit_transform(X, y)
# print "Done in %fs" % (time() - t0)
# print "L1:      n_samples: %d, n_features: %d" % X.shape
# print


# Testset - X_new
docs_new = ["Eli Lilly and Company complies with the U.S.-EU Safe Harbor Framework and the U.S.-Swiss Safe Harbor Framework as set forth by the U.S. Department of Commerce regarding the collection, use, and retention of personal information from European Union member countries and Switzerland. Eli Lilly and Company has certified that it adheres to the Safe Harbor Privacy Principles of notice, choice, onward transfer, security, data integrity, access, and enforcement. To learn more about the Safe Harbor program, and to view Eli Lilly and Company's certification, please visit", 
            'Through this website, you may be asked to voluntarily provide to us information that can identify or locate you, such as your name, address, telephone number, e-mail address, and other similar information ("Personal Information"). You may always refuse to provide Personal Information to us, but this may lead to our inability to provide you with certain information, products or services.',
            'We may also collect information that does not identify you ("Other Information"), such as the date and time of website session, movements across our website, and information about your browser, We do not tie Other Information to Personal Information. This Other Information may be used and stored by Lilly or its agents. Other Information we collect may include your IP Address (a number used to identify your computer on the Internet) and other information gathered through our weblogs, cookies or by other means (see below). We use Other Information to administer and update our website and for other everyday business purposes. Lilly reserves the right to maintain, update, disclose or otherwise use Other Information, without limitation.',
            'This website may use a technology called a "cookie". A cookie is a piece of information that our webserver sends to your computer (actually to your browser file) when you access a website. Then when you come back our site will detect whether you have one of our cookies on your computer. Our cookies help provide additional functionality to the site and help us analyze site usage more accurately. For instance, our site may set a cookie on your browser that keeps you from needing to remember and then enter a password more than once during a visit to the site. With most Internet browsers or other software, you can erase cookies from your computer hard drive, block all cookies or receive a warning before a cookie is stored. Please refer to your browser instructions to learn more about these functions. If you reject cookies, functionality of the site may be limited, and you may not be able to take advantage of many of the site features',
            'This website collects and uses Internet Protocol (IP) Addresses. An IP Address is a number assigned to your computer by your Internet service provider so you can access the Internet. Generally, an IP address changes each time you connect to the Internet (it is a "dynamic" address). Note, however, that if you have a broadband connection, depending on your individual circumstance, it is possible that your IP Address that we collect, or even perhaps a cookie we use, may contain information that could be deemed identifiable. This is because with some broadband connections your IP Address does not change (it is "static") and could be associated with your personal computer. We use your IP address to report aggregate information on use and to help improve the website.',
            'Except as provided below, Lilly will not share your Personal Information with third parties unless you have consented to the disclosure',
            'We may share your Personal Information with our agents, contractors or partners in connection with services that these individuals or entities perform for, or with, Lilly, such as sending email messages, managing data, hosting our databases, providing data processing services or providing customer service. These agents, contractors or partners are restricted from using this data in any way other than to provide services for Lilly, or services for the collaboration in which they and Lilly are engaged (for example, some of our products are developed and marketed through joint agreements with other companies)',
            'Lilly will share Personal Information to respond to duly authorized subpoenas or information requests of governmental authorities, to provide Internet security or where required by law. In exceptionally rare circumstances where national, state or company security is at issue, Lilly reserves the right to share our entire database of visitors and customers with appropriate governmental authorities',
            'We may also provide Personal Information and Other Information to a third party in connection with the sale, assignment, or other transfer of the business of this website to which the information relates, in which case we will require any such buyer to agree to treat Personal Information in accordance with this Privacy Statement',
            'Lilly will respect your marketing and communications preferences. You can opt-out of receiving commercial emails from us at any time by following the opt-out instructions in our commercial emails. You can also request removal from all our contact lists, by writing to us at the following address',
            'Lilly encourages parents and guardians to be aware of and participate in their children online activities. You should be aware that this site is not intended for, or designed to attract, individuals under the age of 18. We do not collect Personal Information from any person we actually know to be under the age of 18',
            'Lilly protects all Personal Information with reasonable administrative, technical and physical safeguards. For example, areas of this website that collect sensitive Personal Information use industry standard secure socket layer encryption (SSL); however, to take advantage of this your browser must support encryption protection (found in Internet Explorer release 3.0 and above).',
            'As a convenience to our visitors, this Website currently contains links to a number of sites owned and operated by third parties that we believe may offer useful information. The policies and procedures we describe here do not apply to those sites. Lilly is not responsible for the collection or use of Personal Information or Other Information at any third party sites. Therefore, Lilly disclaims any liability for any third party use of Personal Information or Other Information obtained through using the third party web site. We suggest contacting those sites directly for information on their privacy, security, data collection, and distribution policies.',
            'We may update this Privacy Statement from time to time. When we do update it, for your convenience, we will make the updated statement available on this page. We will always handle your Personal Information in accordance with the Privacy Statement in effect at the time it was collected. We will not make any materially different use or new disclosure of your Personal Information unless we notify you and give you an opportunity to object',
            'If you have any questions or comments about this Privacy Statement, please contact us by writing to:',
            ]
X_new = vectorizer.transform(docs_new)


# Train classifiers
print "Training Classifiers..."
t0 = time()

clf_nb = MultinomialNB()
clf_lsvc = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
clf_svc = SVC(C=1024, kernel='rbf', degree=3, gamma=0.001, probability=True)
clf_rdg = RidgeClassifier(tol=1e-1)
clf_sgd = SGDClassifier(alpha=.0001, n_iter=50, penalty="l2")

# Logistic regression requires OneVsRestClassifier which hides
# its methods such as decision_function
# It will require extra implementation efforts to use it as a candidate
# for multilabel classification
# clf_lgr = OneVsRestClassifier(LogisticRegression(C=1000,penalty='l1'))
# kNN does not have decision function due to its nature
# clf = KNeighborsClassifier(n_neighbors=13)

# train
clf_nb.fit(X, y)
clf_lsvc.fit(X, y)
clf_rdg.fit(X, y)
clf_svc.fit(X, y)
clf_sgd.fit(X, y)

print "Train time: %0.3fs" % (time() - t0)
print


# predict use a classifier
predicted = clf_rdg.predict(X_new)
print predicted

for doc, category in zip(docs_new, predicted):
    print '%r => %s' % (doc, data_train.target_names[int(category)])
    print


# decision_function and predict_proba
print clf_nb
pred_prob = clf_nb.predict_proba(X_new)
print pred_prob
print

print clf_lsvc
pred_decisio = clf_lsvc.decision_function(X_new)
print pred_decisio
print 

print clf_svc
# SVC should have the decision_function method, but got error:
# error - ValueError: setting an array element with a sequence
# pred_decisio = clf_svc.decision_function(X_new)
pred_prob = clf_svc.predict_proba(X_new)
print pred_prob
print

print clf_rdg
pred_decisio = clf_rdg.decision_function(X_new)
print pred_decisio
print

print clf_sgd
pred_decisio = clf_sgd.decision_function(X_new)
# Mentioned in scikit learn's API class manual
# that SGDClassifier should have menthod predict_proba
# but in test, none of the three loss modes of SGD supports predict_proba
# pred_prob = clf_sgd.predict_proba(X_new)
print pred_decisio
print