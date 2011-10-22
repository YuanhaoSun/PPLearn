"""
======================================================
Classification - Two-Layer
======================================================
Implementing the two-layer classifiers strcuture
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
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm.sparse import LinearSVC
from sklearn.utils.extmath import density
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

# Deaclear categories
# categories = ['SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
#             'Process', 'Retention']
# categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share']
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention']

ca_categories = ['ca', 'ca.subscription']
collect_categories = ['collect.automatic', 'collect.other', 'collect.user']
cookies_categories = ['cookies', 'cookies.choice']
share_categories = ['share.law', 'share.ma', 'share.statement', 'share.third.party']


print __doc__


# Load all training datasets
print "Loading privacy policy datasets..."
data_train = load_files('Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
ca_train = load_files('Privacypolicy/raw_ca', categories = ca_categories,
                        shuffle = True, random_state = 42)
collect_train = load_files('Privacypolicy/raw_collect', categories = collect_categories,
                        shuffle = True, random_state = 42)
cookies_train = load_files('Privacypolicy/raw_cookies', categories = cookies_categories,
                        shuffle = True, random_state = 42)
share_train = load_files('Privacypolicy/raw_share', categories = share_categories,
                        shuffle = True, random_state = 42)
print 'Data loaded!'
print


# Split datasets
y_L1 = data_train.target
y_L2_ca = ca_train.target
y_L2_collect = collect_train.target
y_L2_cookies = cookies_train.target
y_L2_share = share_train.target


# Extract features
print "Extracting features from Layer 1 training set using a sparse vectorizer..."
t0 = time()
vectorizer = Vectorizer()
X_L1 = vectorizer.fit_transform(data_train.data)
print "Done in %0.3fs" % (time() - t0)
print "L1:      n_samples: %d, n_features: %d" % X_L1.shape
print

print "Extracting features from Layer 2 training sets using the same vectorizer..."
t0 = time()
X_L2_ca = vectorizer.transform(ca_train.data)
X_L2_collect = vectorizer.transform(collect_train.data)
X_L2_cookies = vectorizer.transform(cookies_train.data)
X_L2_share = vectorizer.transform(share_train.data)
print "Done in %0.3fs" % (time() - t0)
print "CA:      n_samples: %d, n_features: %d" % X_L2_ca.shape
print "Collect: n_samples: %d, n_features: %d" % X_L2_collect.shape
print "Cookies: n_samples: %d, n_features: %d" % X_L2_cookies.shape
print "Share:   n_samples: %d, n_features: %d" % X_L2_share.shape
print


# # Feature selection for the L1 dataset
# select_chi2 = 1000
# print ("Extracting %d best features by a chi-squared test" % select_chi2)
# t0 = time()
# ch2 = SelectKBest(chi2, k = select_chi2)
# X_L1 = ch2.fit_transform(X_L1, y_L1)
# print "Done in %fs" % (time() - t0)
# print "L1:      n_samples: %d, n_features: %d" % X_L1.shape
# print


# Train L1 classifier
print "Training L1 Classifier..."
t0 = time()
clf = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
print clf
clf.fit(X_L1, y_L1)
train_time = time() - t0
print "Train time: %0.3fs" % train_time
print


# Train L2 classifiers
print "Training L2 Classifiers..."
t0 = time()

# comment out all linearSVC
# clf_ca = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
# clf_collect = LinearSVC(loss='l2', penalty='l2', C=256, dual=False, tol=1e-2)
# clf_cookies = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
# clf_share = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)

# v0.2 adjusted the classifiers after test results
# Unfixed bug: SGDClassifier returns Numpy integer in predict method...
# clf_ca = SGDClassifier(alpha=.0001, n_iter=50, penalty='l1')
clf_ca = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)
clf_collect = KNeighborsClassifier(n_neighbors=13)
clf_cookies = KNeighborsClassifier(n_neighbors=3)
clf_share = LinearSVC(loss='l2', penalty='l2', C=1000, dual=False, tol=1e-3)

clf_ca.fit(X_L2_ca, y_L2_ca)
clf_collect.fit(X_L2_collect, y_L2_collect)
clf_cookies.fit(X_L2_cookies, y_L2_cookies)
clf_share.fit(X_L2_share, y_L2_share)

train_time = time() - t0
print "Train time: %0.3fs" % train_time
print

# Test: Predict on new texts
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
predicted = clf.predict(X_new)

for doc, category in zip(docs_new, predicted):
        if data_train.target_names[category] == 'CA':
            # Process the text into vector
            doc_li = [doc]
            X_new_l2 = vectorizer.transform(doc_li)
            # Predict using the trained L2 classifier
            L2_predicted = clf_ca.predict(X_new_l2)
            for category_l2 in L2_predicted:
                print '%r => %s => %s' % (doc, data_train.target_names[category], 
                                        ca_train.target_names[category_l2])
                print

        elif data_train.target_names[category] == 'Collect':
            doc_li = [doc]
            X_new_l2 = vectorizer.transform(doc_li)
            L2_predicted = clf_collect.predict(X_new_l2)
            for category_l2 in L2_predicted:
                print '%r => %s => %s' % (doc, data_train.target_names[category], 
                                        collect_train.target_names[category_l2])
                print                                        

        elif data_train.target_names[category] == 'Cookies':
            doc_li = [doc]
            X_new_l2 = vectorizer.transform(doc_li)
            L2_predicted = clf_cookies.predict(X_new_l2)
            for category_l2 in L2_predicted:
                print '%r => %s => %s' % (doc, data_train.target_names[category], 
                                        cookies_train.target_names[category_l2])
                print

        elif data_train.target_names[category] == 'Share':
            doc_li = [doc]
            X_new_l2 = vectorizer.transform(doc_li)
            L2_predicted = clf_share.predict(X_new_l2)
            for category_l2 in L2_predicted:
                print '%r => %s => %s' % (doc, data_train.target_names[category], 
                                        share_train.target_names[category_l2])
                print

        else:
            print '%r => %s' % (doc, data_train.target_names[category])
            print


#Predict
# pred = clf.predict(X_test)
