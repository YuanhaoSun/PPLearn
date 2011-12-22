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
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic
from nltk import word_tokenize as wt
from nltk import pos_tag
from nltk import PorterStemmer 
from nltk import WordNetLemmatizer



###############################################################################
# Negation bigram construction

def pos_negation_bigram(line_list):
    """
    Input: 
        line_list (list of strings(sentences/documents)) - e.g. dataset.data

    POS tag the line, match patterns of negations to form bigram terms

    Return: pos_neg_bigram_list (list of strings(terms that meets the POS criteria))
    """
    neg_verb_set = ['not', 'never', 'neither']
    neg_noun_set = ['without']
    verb_window = 10
    noun_window = 3

    pos_neg_bigram_list = []
    for i, line in enumerate(line_list):
        # linercase
        line = line.lower()
        # tokenize
        line_token = wt(line)
        # base for return
        pos_neg_bigram_line = []
        # POS 
        pos_line = pos_tag(line_token)

        # =========
        # POS part
        for tagged_tuple in pos_line:
            term = tagged_tuple[0]
            tag  = tagged_tuple[1]
            pos_neg_bigram_line.append(term+tag)
        # back to sentence as a string
        
        # =========
        # Then negation bigram construction part
        # first iteration to find flag words
        neg_verb = None
        neg_verb_flag = None
        neg_noun_flag = None
        for i, tagged_tuple in enumerate(pos_line):
            term = tagged_tuple[0]
            if term in neg_verb_set:
                neg_verb_flag = i
                neg_verb = term
            elif term in neg_noun_set:
                neg_noun_flag = i

        # second iteration to find neg_verb match and form bigram
        if neg_verb_flag != None:
            for i, tagged_tuple in enumerate(pos_line):
                term = tagged_tuple[0]
                tag  = tagged_tuple[1]
                if (i-neg_verb_flag)<=verb_window and (i-neg_verb_flag)>0 and tag.startswith('V'):
                    pos_neg_bigram_line.append(neg_verb+term)

        # third iteration to find neg_noun match and form bigram
        if neg_noun_flag != None:
            for i, tagged_tuple in enumerate(pos_line):
                term = tagged_tuple[0]
                tag  = tagged_tuple[1]
                if (i-neg_noun_flag)<=noun_window and (i-neg_noun_flag)>0 and tag.startswith('N'):
                    pos_neg_bigram_line.append("without"+term)

        # back to sentence as a string
        neg_bigram_sentence = ' '.join(pos_neg_bigram_line)
        pos_neg_bigram_list.append(neg_bigram_sentence)
    return pos_neg_bigram_list



# data_set = joblib.load('../Dataset/train_datasets_3/data_set_origin.pkl')
data_set = joblib.load('../Dataset/trouble_datasets_3/data_set_origin.pkl')

# Timing experiments
print data_set.data[2:5]
t0 = time()
processed_data = pos_negation_bigram(data_set.data)
print "Done in %fs" % (time() - t0)
data_set.data = processed_data
print data_set.data[2:5]
print len(data_set.data)

# joblib.dump(data_set, '../Dataset/train_datasets_3/data_set_pos_negation_bigram.pkl')
joblib.dump(data_set, '../Dataset/trouble_datasets_3/data_set_pos_negation_bigram.pkl')