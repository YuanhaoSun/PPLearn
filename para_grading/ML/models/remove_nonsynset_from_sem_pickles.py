"""
In ml_feaature_engineering.py, when the data_sets for "bag-of-synset" are generated,
nonsynset terms are also saved.

This program removes all nonsynsets in these saved data_sets and save them as new data_sets
"""

from sklearn.externals import joblib
import re


# pickles = ['sem_pickles_with_nonsynsets/data_set_sem_firstsense.pkl', 
#           'sem_pickles_with_nonsynsets/data_set_sem_corpus_sentence_wsd.pkl',
#           'sem_pickles_with_nonsynsets/data_set_sem_corpus_word_wsd.pkl',
#           'sem_pickles_with_nonsynsets/data_set_sem_internal_word_wsd.pkl',
#           'sem_pickles_with_nonsynsets/data_set_sem_internal_sentence_wsd.pkl']

# for pickle in pickles:

# Load datasets
# data_set = joblib.load('sem_pickles_with_nonsynsets/data_set_sem_firstsense.pkl')
# data_set = joblib.load('sem_pickles_with_nonsynsets/data_set_sem_corpus_sentence_wsd.pkl')
# data_set = joblib.load('sem_pickles_with_nonsynsets/data_set_sem_corpus_word_wsd.pkl')
# data_set = joblib.load('sem_pickles_with_nonsynsets/data_set_sem_internal_word_wsd.pkl')
data_set = joblib.load('sem_pickles_with_nonsynsets/data_set_sem_internal_sentence_wsd.pkl')

data = data_set.data

new_data = []
for line in data:
    list_terms = line.split()
    # Three possible ways to remove the nonsynset terms by judging if it ends with digits
    # new_list = [x for x in list_terms if re.match("^[a-zA-Z_]+\d+$", x)]
    # new_list = [x for x in list_terms if x[-1].isdigit()]
    new_list = [x for x in list_terms if re.match(r".*\d", x)]
    # change list back to a string
    sentence = ' '.join(new_list)
    new_data.append(sentence)

data_set.data = new_data

# save pickles
# joblib.dump(data_set, './data_set_sem_firstsense.pkl')
# joblib.dump(data_set, './data_set_sem_corpus_sentence_wsd.pkl')
# joblib.dump(data_set, './data_set_sem_corpus_word_wsd.pkl')
# joblib.dump(data_set, './data_set_sem_internal_word_wsd.pkl')
joblib.dump(data_set, './data_set_sem_internal_sentence_wsd.pkl')