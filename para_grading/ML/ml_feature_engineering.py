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
# Stemming and lemmatizing

def stemming(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Iterates over all terms in lines, stem them

    Return: stemmed_list (list of strings(terms that stemmed))
    """
    stemmed_list = []
    stemmer = PorterStemmer()
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
        # list to store stemmed terms
        stemmed_line = []
        for term in line_token:
            term = stemmer.stem_word(term)
            stemmed_line.append(term)
        # back to sentence as a string
        stemmed_sentence = ' '.join(stemmed_line)
        stemmed_list.append(stemmed_sentence)
    return stemmed_list


def lemmatizing(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Iterates over all terms in lines, lemmatize them using WordNetLemmatizer()

    Return: lemmatized_list (list of strings(terms that stemmed))
    """
    lemmatized_list = []
    lemmatizer = WordNetLemmatizer()
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
        # stemming
        lemmatized_line = []
        for term in line_token:
            term = lemmatizer.lemmatize(term)
            lemmatized_line.append(term)
        # back to sentence as a string
        lemmatized_sentence = ' '.join(lemmatized_line)
        lemmatized_list.append(lemmatized_sentence)
    return lemmatized_list


def wn_lemmatize(lemma):
    """
    Auxiliary function for pos_lemmatizing (below)

    Lemmatize the supplied (word, pos) pair using
    nltk.stem.WordNetLemmatizer. If the tag corresponds to a
    WordNet tag, then we convert to that one and use it, else we
    just use the strong for lemmatizing.
    """        
    string, tag = lemma
    string = string.lower()
    tag = tag.lower()
    wnl = WordNetLemmatizer()
    if tag.startswith('v'):    tag = 'v'
    elif tag.startswith('n'):  tag = 'n'
    elif tag.startswith('j'):  tag = 'a'
    elif tag.startswith('rb'): tag = 'r'
    if tag in ('a', 'n', 'r', 'v'):        
        return wnl.lemmatize(string, tag)
    else:
        return wnl.lemmatize(string) 


def pos_lemmatizing(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Iterates over all terms in lines, lemmatize them using WordNetLemmatizer()
    Terms are pre-processed using POS tagging to improve accuracy

    Return: lemmatized_list (list of strings(terms that stemmed))
    """
    lemmatized_list = []
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
        # POS 
        pos_line = pos_tag(line_token)
        # list for all lemmatized terms
        lemmatized_line = []
        for lemma in pos_line:
            term = wn_lemmatize(lemma)
            lemmatized_line.append(term)
        # back to sentence as a string
        lemmatized_sentence = ' '.join(lemmatized_line)
        lemmatized_list.append(lemmatized_sentence)
    return lemmatized_list



###############################################################################
# POS tagging and bagging

def select_by_pos(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Iterates over all terms in lines, select terms with meaningful type of POS

    Return: POSed_list (list of strings(terms that meets the POS criteria))
    """
    POSed_list = []
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
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Iterates over all terms in lines, add POS tag to words.
    E.g. 'said' -> ('said', 'VD') -> saidVD

    Return: tagged_list (list of strings(terms that meets the POS criteria))
    """
    tagged_list = []
    for i, line in enumerate(line_list):
        # linercase
        line = line.lower()

        # Having punctuation removal before POS seems to be a bad idae
        # # remove punctuation
        # # below method will simply remove punctuation, but mistakes such as amazon.com => amazoncom
        # # line = ''.join([c for c in line 
        #                                     # if re.match("[a-z\-\' \n\t]", c)])
        # # this solve the problem above:
        # line = re.sub('[^A-Za-z0-9]+', ' ', line)
        # tokenize
        line_token = wt(line)

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


def pos_bagging(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Use POS tags to replace all words

    Return: tagged_list (list of strings(terms that meets the POS criteria))
    """
    bagged_list = []
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
        # POS 
        pos_line = pos_tag(line_token)
        # filter line using POS info
        # only remain verbs, nouns, adverbs, adjectives
        bagged_line = []
        for tagged_tuple in pos_line:
            term = tagged_tuple[0]
            tag  = tagged_tuple[1]
            bagged_line.append(tag)
        # back to sentence as a string
        bagged_sentence = ' '.join(bagged_line)
        bagged_list.append(bagged_sentence)
    return bagged_list



###############################################################################
# Semantic tagging - bag of synsets

def sem_firstsense(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Get a list of synsets or terms, synsets for the terms whic have synsets, term for the ones don't
    Use first senses

    Return: synset_list (list of strings(terms that meets the POS criteria))
    """
    total_synset_sentence_list = []
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
        # list of first-sense synsets
        synset_list = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in line_token if wn.synsets(x) ])
        # format synset into term, e.g. Synset.share.v.1 -> sharev1
        synset_formatted_list = []
        for synset in synset_list:
            formatted_term = re.sub('[^A-Za-z0-9]+', '', str(synset))
            formatted_term = formatted_term.lstrip('Synset')
            synset_formatted_list.append(formatted_term)
        # list of terms without synset defination
        nonsynset_list = [ x for x in line_token if not wn.synsets(x)]
        # add synset list and nonsynset list together
        total_synset_list = synset_formatted_list + nonsynset_list
        # back to sentence as a string
        total_synset_sentence = ' '.join(total_synset_list)
        total_synset_sentence_list.append(total_synset_sentence)
    return total_synset_sentence_list



def internal_word_max_WSD(sentence, word):
    """
    Auxiliary function for sem_wsd()

    Input: a sentence and a word in the sentence,
            sentence is a list of words, not a string

    Return: synset(sense) of the word that maximize one similarity with another word in the sentence

    Derived from code at http://www.jaist.ac.jp/~s1010205/sitemap-2/styled-7/
    """    
    wordsynsets = wn.synsets(word)
    bestScore = 0.0
    result = None
    for synset in wordsynsets:
        for w in sentence:
            score = 0.0
            for wsynset in wn.synsets(w):
                sim = wn.path_similarity(wsynset, synset)
                if(sim == None):
                    continue
                else:
                    score += sim
            if (score > bestScore):
                bestScore = score
                result = synset
    return result


def internal_sentence_max_WSD(sentence, word):
    """
    Auxiliary function for sem_wsd()

    Input: a sentence and a word in the sentence,
            sentence is a list of words, not a string
    
    Return: synset(sense) of the word that maximize similarity with all other synsets in the sentence
    """    
    wordsynsets = wn.synsets(word)
    bestScore = 0.0
    result = None
    for synset in wordsynsets:
        score = 0.0
        for w in sentence:
            for wsynset in wn.synsets(w):
                sim = wn.path_similarity(wsynset, synset)
                if(sim == None):
                    continue
                else:
                    score += sim
        if (score > bestScore):
            bestScore = score
            result = synset
    return result



def sem_wsd_sentence(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Get a list of synsets or terms, synsets for the terms whic have synsets, term for the ones don't
    Use internal maximization on sentence either internal word max or internal sentence max

    Return: synset_list (list of strings(terms that meets the POS criteria))
    """
    total_synset_sentence_list = []
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
        # list of wsd synsets
        synset_list = reduce(lambda x,y:x+y, [ [internal_sentence_max_WSD(line_token, x)] for x in line_token if wn.synsets(x) ])
        # synset_list = reduce(lambda x,y:x+y, [ [internal_word_max_WSD(line_token, x)] for x in line_token if wn.synsets(x) ])
        # format synset into term, e.g. from Synset.share.v.1 -> sharev1
        synset_formatted_list = []
        for synset in synset_list:
            formatted_term = re.sub('[^A-Za-z0-9]+', '', str(synset))
            formatted_term = formatted_term.lstrip('Synset')
            synset_formatted_list.append(formatted_term)
        # list of terms without synset defination
        nonsynset_list = [ x for x in line_token if not wn.synsets(x)]
        # add synset list and nonsynset list together
        total_synset_list = synset_formatted_list + nonsynset_list
        # back to sentence as a string
        total_synset_sentence = ' '.join(total_synset_list)
        total_synset_sentence_list.append(total_synset_sentence)
    return total_synset_sentence_list


def sem_wsd_corpus(line_list):
    """
    Input: line_list (list of strings(sentences/documents)) - e.g. dataset.data

    Get a list of synsets or terms, synsets for the terms whic have synsets, term for the ones don't
    Use internal maximization on corpus either internal word max or internal sentence max

    Return: synset_list (list of strings(terms that meets the POS criteria))
    """
    # get a term based corpus list for compute internal corpus maximization WSD
    corpus_list = []
    for line in line_list:
        line = line.lower()
        nopunct_line = re.sub('[^A-Za-z0-9]+', ' ', line)
        line_token = wt(nopunct_line)
        corpus_list = corpus_list+line_token
    corpus_list = list(set(corpus_list))

    # start
    total_synset_sentence_list = []
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
        # list of wsd synsets
        synset_list = reduce(lambda x,y:x+y, [ [internal_sentence_max_WSD(corpus_list, x)] for x in line_token if wn.synsets(x) ])
        # synset_list = reduce(lambda x,y:x+y, [ [internal_word_max_WSD(corpus_list, x)] for x in line_token if wn.synsets(x) ])
        # format synset into term, e.g. from Synset.share.v.1 -> sharev1
        synset_formatted_list = []
        for synset in synset_list:
            formatted_term = re.sub('[^A-Za-z0-9]+', '', str(synset))
            formatted_term = formatted_term.lstrip('Synset')
            synset_formatted_list.append(formatted_term)
        # list of terms without synset defination
        nonsynset_list = [ x for x in line_token if not wn.synsets(x)]
        # add synset list and nonsynset list together
        total_synset_list = synset_formatted_list + nonsynset_list
        # back to sentence as a string
        total_synset_sentence = ' '.join(total_synset_list)
        total_synset_sentence_list.append(total_synset_sentence)
    return total_synset_sentence_list





###############################################################################
# # Save original dataset
# 
# # Used only once, otherwise if there is change in data set
# # Load categories
# categories = ['nolimitshare', 'notsell', 'notsellnotshare', 'notsharemarketing', 'sellshare', 
            # 'shareforexception', 'shareforexceptionandconsent', 'shareonlyconsent']
# # Load data
# print "Loading privacy policy dataset for categories:"
# print categories if categories else "all"
# data_train = load_files('../Dataset/ShareStatement/raw', categories = categories,
#                         shuffle = True, random_state = 42)
# data_test = load_files('../Dataset/ShareStatement/test', categories = categories,
#                         shuffle = True, random_state = 42)
# data_trouble = load_files('../Dataset/ShareStatement/trouble', categories = categories,
                        # shuffle = True, random_state = 42)                        
# print 'data loaded'
# print len(data_train.data)
# print len(data_test.data)
# print len(data_trouble.data)
# print
# # Save dataset
# joblib.dump(data_train, '../Dataset/train_datasets/data_set_origin.pkl')
# joblib.dump(data_test, '../Dataset/test_datasets/data_set_origin.pkl')
# joblib.dump(data_trouble, '../Dataset/trouble_datasets/data_set_origin.pkl')
# print 'data_set saved'
# print



###############################################################################
# # Feature Engineering and save to pickles

# 
# Load original datasets
# data_set = joblib.load('../Dataset/train_datasets/data_set_origin.pkl')
# data_set = joblib.load('../Dataset/test_datasets/data_set_origin.pkl')
# data_set = joblib.load('../Dataset/trouble_datasets/data_set_origin.pkl')

# data_set = joblib.load('../Dataset/train_datasets/data_set_stemmed.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized.pkl')
# data_set = joblib.load('../Dataset/train_datasets/data_set_lemmatized_pos.pkl')


# # Timing experiments
# print data_set.data[2:5]
# t0 = time()
# processed_data = lemmatizing(data_set.data)
# print "Done in %fs" % (time() - t0)
# data_set.data = processed_data
# print data_set.data[2:5]
# print len(data_set.data)

# stemming
# lemmatizing
# pos_lemmatizing
# select_by_pos
# pos_tagging
# pos_bagging
# sem_firstsense
# sem_wsd_sentence
# sem_wsd_corpus


# # Train datasets
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_stemmed.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_lemmatized.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_lemmatized_pos.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_pos_selected.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_pos_tagged.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_pos_bagged.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_sem_firstsense.pkl')
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_sem_internal_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_sem_internal_sentence_wsd.pkl') # when using internal_sentence_max_WSD
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_sem_corpus_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/train_datasets/data_set_sem_corpus_sentence_wsd.pkl') # when using internal_sentence_max_WSD
#
# 
# # Test datasets
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_stemmed.pkl')
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_lemmatized.pkl')
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_lemmatized_pos.pkl')
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_pos_selected.pkl')
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_pos_tagged.pkl')
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_pos_bagged.pkl')
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_sem_firstsense.pkl')
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_sem_internal_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_sem_internal_sentence_wsd.pkl') # when using internal_sentence_max_WSD
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_sem_corpus_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/test_datasets/data_set_sem_corpus_sentence_wsd.pkl') # when using internal_sentence_max_WSD
# 
# 
# # Trouble datasets
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_stemmed.pkl')
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_lemmatized.pkl')
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_lemmatized_pos.pkl')
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_pos_selected.pkl')
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_pos_tagged.pkl')
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_pos_bagged.pkl')
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_sem_firstsense.pkl')
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_sem_internal_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_sem_internal_sentence_wsd.pkl') # when using internal_sentence_max_WSD
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_sem_corpus_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, '../Dataset/trouble_datasets/data_set_sem_corpus_sentence_wsd.pkl') # when using internal_sentence_max_WSD
# 




###############################################################################
# Individual dumping (not necessary if use lines above)

# # Feature selection using POS types
# # Only keep verbs, nouns, adverbs, adjectives
# # These lines only need to be used once, and save the POS-selected data_set
# POSed_data = select_by_pos(data_set.data)
# data_set.data = POSed_data
# joblib.dump(data_set, 'models/data_set_pos_selected.pkl')

# # Feature engineering using POS tags
# # Add POS tags to all words, e.g. 'said' -> 'saidVD'
# # These lines only need to be used once, and save the POS-tagged data_set
# tagged_data = pos_tagging(data_set.data)
# data_set.data = tagged_data
# joblib.dump(data_set, 'models/data_set_pos_tagged.pkl')

# # Feature engineering using POS bag
# # Use POS tags to replace all words
# # These lines only need to be used once, and save the POS-bagged data_set
# tagged_data = pos_bagging(data_set.data)
# data_set.data = tagged_data
# joblib.dump(data_set, 'models/data_set_pos_bagged.pkl')


# # Feature engineering using WordNet first sense
# # These lines only need to be used once, and save the POS-bagged data_set
# tagged_data = sem_firstsense(data_set.data)
# data_set.data = tagged_data
# joblib.dump(data_set, 'models/data_set_sem_firstsense.pkl')


# # Feature engineering using WordNet with internal sentence maximization WSD
# # These lines only need to be used once, and save the POS-bagged data_set
# tagged_data = sem_wsd_sentence(data_set.data)
# data_set.data = tagged_data
# joblib.dump(data_set, 'models/data_set_sem_internal_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, 'models/data_set_sem_internal_sentence_wsd.pkl') # when using internal_sentence_max_WSD

# # Feature engineering using WordNet with internal corpus maximization WSD
# # These lines only need to be used once, and save the POS-bagged data_set
# tagged_data = sem_wsd_corpus(data_set.data)
# data_set.data = tagged_data
# joblib.dump(data_set, 'models/data_set_sem_corpus_word_wsd.pkl') # when using internal_word_max_WSD
# joblib.dump(data_set, 'models/data_set_sem_corpus_sentence_wsd.pkl') # when using internal_sentence_max_WSD

