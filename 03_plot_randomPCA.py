"""
======================================================
Plot - all instances using PCA for dimen reduction
======================================================
Plot any categorized instances.
Using Random PCA to reduce feature dimensions to 2D
"""

print __doc__

from time import time
import logging
import numpy as np
import pylab as pl

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import RandomizedPCA, MiniBatchSparsePCA, SparsePCA
from sklearn.lda import LDA



###############################################################################
# Preprocess - load data, etc

# categories = ['Cookies','CA', 'Collect']
# categories = ['SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
#             'Process', 'Retention']
categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
            'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
            'Process', 'Retention']
# categories = ['Collect', 'Cookies',]

print "Loading privacy policy dataset for categories:"
print categories

data_train = load_files('Privacypolicy/train', categories = categories,
						shuffle = True, random_state = 42)
data_test = load_files('Privacypolicy/test', categories = categories, 
						shuffle = True, random_state = 42)

print 'data loaded'

# documents = data_train.data + data_test.data
# combine train and test to be used for this plot
documents = data_train.data + data_test.data
target_names = set(data_train.target_names + data_test.target_names)

print "%d documents" % len(documents)
print "%d categories" % len(target_names)
print

print "Extracting features from the training dataset using a sparse vectorizer"
t0 = time()
vectorizer = Vectorizer(max_features=10000)
X = vectorizer.fit_transform(documents)
X = Normalizer(norm="l2", copy=False).transform(X)

# labels, for ploting with categories
y = np.concatenate((data_train.target, data_test.target))

print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape
print

###############################################################################
# Apply RandomizedPCA - Principal Component Analysis using randomized SVD

# pca = RandomizedPCA(n_components=50)

# MiniBatchSparsePCA not working due to error in recogizing X.shape
# pca = MiniBatchSparsePCA(n_components=50)
# SparsePCA not working due to X.shape[1] out of range
# pca = SparsePCA(n_components=50)

X_r = pca.fit(X).transform(X)


# Percentage of variance explained for each components
print 'Explained variance ratio:', \
    pca.explained_variance_ratio_


###############################################################################
# Plot

# auxiliary lists for ploting
num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12 ,13, 14 ,15]
# Generate n distinct colors would be an improvement here
colors = ['Chocolate', 'Gold', 'g', 'r', 'b', 'c', 'm', 'y', 'k', 'Indigo', \
			'Lime', 'MidnightBlue', 'OrangeRed', 'Purple', 'DarkGrey']

# Plot
pl.figure()
for c, i, target_name in zip(colors, num, target_names):
    pl.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
pl.legend()
pl.title('PCA of Privacy Policy dataset')
pl.show()