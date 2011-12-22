from sklearn.datasets import load_files
from sklearn.externals import joblib

import numpy as np

def transfer_8to3(orignal_list):
	'''
	Input: ndarray
	Return: ndarray where 8-class categories changed into 3-class
	'''
	replace = {2: 1, 3: 2, 4: 0, 5: 2, 6: 2, 7: 2}
	data = orignal_list

	mp = np.arange(0,max(data)+1)
	mp[replace.keys()] = replace.values()
	data = mp[data]

	return data


# load from pickle
# load data and initialize classification variables
# data_set = joblib.load('../Dataset/train_datasets/data_set_origin.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_stemmed.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized_pos.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_negation_bigram.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_pos_selected.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_term_extracted.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_pos_tagged.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_pos_bagged.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_firstsense.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_internal_sentence_wsd.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_internal_word_wsd.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_corpus_sentence_wsd.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_sem_corpus_word_wsd.pkl')
data_set = joblib.load('../Dataset/train_datasets/data_set_sem_corpus_sentence_wsd_jiang.pkl')

data_set.category_names = ['bad', 'good', 'neutral']

target_8 = data_set.target
target_3 = transfer_8to3(target_8)

data_set.target = target_3

# joblib.dump(data_set, '../Dataset/train_datasets_3/data_set_sem_firstsense.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets_3/data_set_sem_internal_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/train_datasets_3/data_set_sem_internal_sentence_wsd.pkl') # when using internal_sentence_max_WSD
# joblib.dump(data_set, '../Dataset/train_datasets_3/data_set_sem_corpus_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/train_datasets_3/data_set_sem_corpus_sentence_wsd.pkl') # when using internal_sentence_max_WSD
joblib.dump(data_set, '../Dataset/train_datasets_3/data_set_sem_corpus_sentence_wsd_jiang.pkl') # when using internal_sentence_max_WSD