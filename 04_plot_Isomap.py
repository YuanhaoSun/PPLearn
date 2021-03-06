"""
======================================================
Plot - all instances using Isomap for dimen-reduction
======================================================
Plot any categorized instances.
Using Isomap to reduce feature dimensions to 2D
"""

print __doc__

from time import time
import logging
import numpy as np
import pylab as pl

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer

from sklearn import manifold
from sklearn.decomposition import PCA, RandomizedPCA, MiniBatchSparsePCA, SparsePCA
from sklearn.lda import LDA


###############################################################################
# Preprocess - load data, etc

# categories = ['Cookies','CA', 'Collect']
categories = ['SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention']
# categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
#             'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
#             'Process', 'Retention']
# categories = ['Collect', 'Cookies',]
# categories = ['comp.graphics', 'misc.forsale', 'talk.politics.guns',]

print "Loading privacy policy dataset for categories:"
print categories
data_set = load_files('Privacypolicy/raw', categories = categories,
                        shuffle = True, random_state = 42)
print 'data loaded'

print "%d documents" % len(data_set.data)
print "%d categories" % len(data_set.target_names)
print

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_features=10000)

X = vectorizer.fit_transform(data_set.data)
# X = Normalizer(norm="l2", copy=False).transform(X)
y = data_set.target

print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape
print

X_den = X.todense()

n_samples, n_features = X.shape
n_neighbors = 15


###############################################################################
# Apply Isomap


# print "Computing PCA"
# t0 = time()
# X_iso = RandomizedPCA(n_components=2).fit(X).transform(X)
# print "Train time: %0.3fs" % (time() - t0)
# print

print "Computing Isomap embedding"
t0 = time()
# X_iso = manifold.Isomap(n_neighbors, out_dim=2).fit_transform(X)
X_iso = manifold.Isomap(n_neighbors, out_dim=2, eigen_solver='arpack').\
                        fit(X_den).transform(X_den)
print "Train time: %0.3fs" % (time() - t0)
print

###############################################################################
# Plot

# auxiliary lists for ploting
num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12 ,13, 14 ,15]
# Generate n distinct colors would be an improvement here
colors = ['Chocolate', 'Gold', 'g', 'r', 'b', 'c', 'm', 'y', 'k', 'Indigo', \
            'Lime', 'MidnightBlue', 'OrangeRed', 'Purple', 'DarkGrey']

# Plot
pl.figure()
for c, i, target_name in zip(colors, num, data_set.target_names):
    pl.scatter(X_iso[y == i, 0], X_iso[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('Isomap of Privacy Policy dataset')
pl.show()