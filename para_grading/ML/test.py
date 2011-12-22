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

from topia.termextract import extract

from ml_feature_engineering import pos_tagging


###############################################################################
# Negation bigram construction

def negation_bigram(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    POS tag the line, match patterns of negations to form bigram terms

    Return: neg_bigram_list (list of strings(terms that meets the POS criteria))
    """
    neg_verb_set = ['not', 'never', 'neither']
    neg_noun_set = ['without']
    verb_window = 10
    noun_window = 3

    neg_bigram_list = []
    for i, line in enumerate(line_list):
        # linercase
        line = line.lower()
        
        # Having punctuation removal before POS seems to be a bad idea
        # # remove punctuation
        # # below method will simply remove punctuation, but mistakes such as amazon.com => amazoncom
        # # line = ''.join([c for c in line 
        #                                     # if re.match("[a-z\-\' \n\t]", c)])
        # # this solve the problem above:
        # line = re.sub('[^A-Za-z0-9]+', ' ', line)                                            
        
        # tokenize
        line_token = wt(line)
        # base for return
        neg_bigram_line = line_token
        # POS 
        pos_line = pos_tag(line_token)
        
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
                    neg_bigram_line.append(neg_verb+term)

        # third iteration to find neg_noun match and form bigram
        if neg_noun_flag != None:
            for i, tagged_tuple in enumerate(pos_line):
                term = tagged_tuple[0]
                tag  = tagged_tuple[1]
                if (i-neg_noun_flag)<=noun_window and (i-neg_noun_flag)>0 and tag.startswith('N'):
                    neg_bigram_line.append("without"+term)
        
        # back to sentence as a string
        neg_bigram_sentence = ' '.join(neg_bigram_line)
        neg_bigram_list.append(neg_bigram_sentence)
    return neg_bigram_list





def term_extraction(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    extract the terms using topia package

    Return: term_list (list of strings(terms that meets the POS criteria))
    """
    term_list = []
    extractor = extract.TermExtractor()
    extractor.filter = extract.permissiveFilter
    for line in line_list:
        # using topia extractor to extract terms
        terms = extractor(line)
        # reformat from topia output (term, freq, # of words in term) to single terms
        term_line = []
        for (term, i, j) in terms:
            # deal with terms like "third-party", reformat into "thirdparty"
            term = ''.join(term.split("-"))
            # if multi-word phrase, remove space to form a unique word
            if j > 1:
                term = ''.join(term.split(" "))
            term_line.append(term)
        # back to sentence as a string
        term_sentence = ' '.join(term_line)
        term_list.append(term_sentence)
    return term_list




# extractor = extract.TermExtractor()
# extractor.filter = extract.permissiveFilter

text = "The lazy dogs can not jump over the quick brown fox's tail."
text2 = "I am reading that privacy policy"

texts = ["We do not rent, sell, or share any of this information with third party companies.", 
        "We do not rent, sell, or share any information about the user with any third-parties. ",
        "We do not, under any circumstances, share, sell or rent your information to anyone. ",
        "We never share or sell your personal information", "We neither rent nor sell your Personal Information to anyone", 
        "As a general rule, Blizzard will not forward your information to a third party without your permission."]
    
    # print extractor(text)
    # tokens = wt(text)
    # print tokens
    # pos_line = pos_tag(tokens)
    # print pos_line
    # tagged = pos_tagging([text])
    # print tagged

print term_extraction(texts)

# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# # terms = ["best", "better", "goods"]
# terms = ['go', 'went', 'going', "goes", "gone"]
# # terms = ["cars", "feet"]

# for term in terms:
#   print term, stemmer.stem_word(term), lemmatizer.lemmatize(term, "v")
