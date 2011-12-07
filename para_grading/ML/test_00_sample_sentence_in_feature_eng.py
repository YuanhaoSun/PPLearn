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

from ml_feature_engineering import stemming
from ml_feature_engineering import pos_lemmatizing
from ml_feature_engineering import select_by_pos
from ml_feature_engineering import negation_bigram
from ml_feature_engineering import term_extraction
from ml_feature_engineering import pos_bagging
from ml_feature_engineering import pos_tagging
from ml_feature_engineering import sem_firstsense
from ml_feature_engineering import sem_wsd_sentence


# extractor = extract.TermExtractor()
# extractor.filter = extract.permissiveFilter

text = "These lazy geese and dogs can not jump over the quick brown fox."
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
print wt(text)
print pos_tag(wt(text))
print stemming([text])
print pos_lemmatizing([text])
print select_by_pos([text])
print negation_bigram([text])
print term_extraction([text])
print pos_bagging([text])
print pos_tagging([text])
print sem_firstsense([text])
print sem_wsd_sentence([text])

# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()

# # terms = ["best", "better", "goods"]
# terms = ['go', 'went', 'going', "goes", "gone"]
# # terms = ["cars", "feet"]

# for term in terms:
#   print term, stemmer.stem_word(term), lemmatizer.lemmatize(term, "v")
