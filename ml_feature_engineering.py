from time import time
import re
from pprint import pprint
import numpy as np

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import Vectorizer, CountVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.externals import joblib

from nltk.corpus import wordnet as wn 
from nltk import word_tokenize as wt
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic


def select_by_pos(line_list):
    """
    Input: line_list (list of strings(sentences/documents))

    Iterates over all terms in lines, select terms with meaningful type of POS

    Return: POSed_list (list of strings(terms that meets the POS criteria))
    """
    POSed_list = []
    for i, line in enumerate(line_list):
        # linercase
        line = line.lower()
        # remove punctuation
        # below method will simply remove punctuation, but mistakes such as amazon.com => amazoncom
        # nopunct_line = ''.join([c for c in line 
                                            # if re.match("[a-z\-\' \n\t]", c)])
        # this solve the problem above:
        nopunct_line = re.sub('[^A-Za-z0-9]+', ' ', line)                                            
        # tokenize
        line_token = wt(nopunct_line)
        # POS 
        pos_line = pos_tag(line_token)
        # filter line using POS info
        # only remain verbs, nouns, adverbs, adjectives
        filtered_line = []
        for tagged_tuple in pos_line:
            term = tagged_tuple[0]
            tag  = tagged_tuple[1]
            # find out all verbs, nouns, adverbs, adjectives
            if tag.startswith('V') or tag.startswith('N') or tag.startswith('R') or tag.startswith('J'):
                filtered_line.append(term)
        # back to sentence as a string
        POSed_sentence = ' '.join(filtered_line)
        POSed_list.append(POSed_sentence)
    return POSed_list


def pos_tagging(line_list):
    """
    Input: line_list (list of strings(sentences/documents))

    Iterates over all terms in lines, add POS tag to words.
    E.g. 'said' -> ('said', 'VD') -> saidVD

    Return: tagged_list (list of strings(terms that meets the POS criteria))
    """
    tagged_list = []
    for i, line in enumerate(line_list):
        # linercase
        line = line.lower()
        # remove punctuation
        # below method will simply remove punctuation, but mistakes such as amazon.com => amazoncom
        # nopunct_line = ''.join([c for c in line 
                                            # if re.match("[a-z\-\' \n\t]", c)])
        # this solve the problem above:
        nopunct_line = re.sub('[^A-Za-z0-9]+', ' ', line)                                            
        # tokenize
        line_token = wt(nopunct_line)
        # POS 
        pos_line = pos_tag(line_token)
        # filter line using POS info
        # only remain verbs, nouns, adverbs, adjectives
        tagged_line = []
        for tagged_tuple in pos_line:
            term = tagged_tuple[0]
            tag  = tagged_tuple[1]
            tagged_line.append(term+tag)
        # back to sentence as a string
        tagged_sentence = ' '.join(tagged_line)
        tagged_list.append(tagged_sentence)
    return tagged_list









# # # Save original dataset
# # 
# categories = ['Advertising','CA', 'Collect', 'Cookies', 'Security', 'Share', 
#             'SafeHarbor','Truste', 'Change', 'Location', 'Children', 'Contact', 
#             'Process', 'Retention', 'Linkout', 'California']

# # Load data
# print "Loading privacy policy dataset for categories:"
# print categories if categories else "all"
# data_set = load_files('Privacypolicy/raw', categories = categories,
#                         shuffle = True, random_state = 42)
# print 'data loaded'
# print
# # Save dataset
# joblib.dump(data_set, 'models/data_set_origin.pkl')
# print 'data_set_origin saved'
# print



# Load dataset
data_set = joblib.load('models/data_set_origin.pkl')
categories = data_set.target_names
y = data_set.target


# # Feature selection using POS types
# # Only keep verbs, nouns, adverbs, adjectives
# # These lines only need to be used once, and save the POS-selected data_set
# POSed_data = select_by_pos(data_set.data)
# data_set.data = POSed_data
# joblib.dump(data_set, 'models/data_set_pos_selected.pkl')

# Feature engineering using POS tags
# Add POS tags to all words, e.g. 'said' -> 'saidVD'
# These lines only need to be used once, and save the POS-selected data_set
tagged_data = pos_tagging(data_set.data)
data_set.data = tagged_data
joblib.dump(data_set, 'models/data_set_pos_tagged.pkl')


# # Extract features
# vectorizer = Vectorizer(max_features=10000)

# # Engineering nGram
# # vectorizer.analyzer.min_n = 1
# # vectorizer.analyzer.max_n = 2

# # Engineering stopword
# # vectorizer.analyzer.stop_words = set([])
# vectorizer.analyzer.stop_words = set(["amazon", "com", "inc", "emc", "alexa", "realnetworks", "google", "linkedin",
#                                     "fox", "zynga", "ea", "yahoo", "travelzoo", "kaltura", "2co", "ign", "blizzard",
#                                     "jobstreetcom", "surveymonkey"])
# # vectorizer.analyzer.stop_words = set(["we", "do", "you", "your", "the", "that", "this", 
# #                                     "is", "was", "are", "were", "being", "be", "been",
# #                                     "for", "of", "as", "in",  "to", "at", "by",
# #                                     # "or", "and",
# #                                     "ve",
# #                                     "amazon", "com", "inc", "emc", "alexa", "realnetworks", "google", "linkedin",
# #                                     "fox", "zynga", "ea", "yahoo", "travelzoo", "kaltura", "2co", "ign", "blizzard",
# #                                     "jobstreetcom", "surveymonkey"])

# X = vectorizer.fit_transform(data_set.data)
# # X = Normalizer(norm="l2", copy=False).transform(X)

# # # get back the terms of all training samples from Vectorizor
# # terms = vectorizer.inverse_transform(X)
# # print terms[0]

# # # Build dictionary after vectorizer is fit
# # print vectorizer.vocabulary
# # vocabulary = np.array([t for t, i in sorted(vectorizer.vocabulary.iteritems(), key=itemgetter(1))])

# # Engineering feature selection
# ch2 = SelectKBest(chi2, k = 90)
# X = ch2.fit_transform(X, y)

# # X = X.toarray()
# # X = X.todense()

# n_samples, n_features = X.shape
# print "n_samples: %d, n_features: %d" % (n_samples, n_features)