import numpy as np
import re

from nltk.corpus import wordnet as wn 
from nltk import word_tokenize as wt
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic

from similarity_utils import load_sentences


# auxiliary function to calculate max_similarity between synsets from two words
def max_similarity(term1, term2, metric, ic):
    # get synset set for two terms
    synset_list1 = wn.synsets(term1)
    synset_list2 = wn.synsets(term2)

    # if no synset exists, return 0
    if (synset_list1==[]) or (synset_list2==[]):
        return 0

    # score_list to save all scores
    score_list = []
    # calculate similarity scores for all possible combination, and store into the score_list
    for synset1 in synset_list1:
        for synset2 in synset_list2:
            if ic is not None:
                try:
                    mark = metric(synset1, synset2, ic)
                    if mark is None:
                        mark = 0.0
                except: 
                    mark = 0.0
                # handle infinitity mark for jcn measure
                if mark == 1e+300:
                    mark = 1.0
                score_list.append(mark)
            else: 
                try:
                    mark = metric(synset1, synset2)
                    if mark is None:
                        mark = 0.0
                except:
                    mark = 0.0
                score_list.append(mark)
    score_list.sort()
    score_list.reverse()
    # return best similarity score
    return score_list[0]


# given a joint set and a sentence as input
# 
# Parameter:
#   J: a joint list created using two sentences
#   Q: a sentence
# 
# Return: r: the word order vector, but in type of List
# because list can be easily transferred into vector using numpy in Python
def calculate_word_order_vector(J, Q, threshold, metric, ic):
    # preprocess sentence Q
    # from list of words, to list of tuples in form (word, order)
    # put the list of tuples as a list Q_tuple
    Q_tuple = []
    for i, word in enumerate(Q):
        Q_tuple.append((word,i+1))
    # initialize word order vector r
    r = []
    # start building the word order vector
    for word1 in J:
        find_match_flag = 0
        for word_tuple in Q_tuple:
            word2 = word_tuple[0]
            order = word_tuple[1]
            # if exact match, add order to r
            # add '(find_match_flag!=1)' to prevent adding more than once in r[]
            if (word1 == word2) and (find_match_flag!=1):
                # if match, put the order into r[]
                r.append(order)
                # set flag for further steps
                find_match_flag = 1
        # if no exact match is found
        if find_match_flag != 1:
            for word_tuple in Q_tuple:
                word2 = word_tuple[0]
                order = word_tuple[1]
                score = max_similarity(word1, word2, metric, ic)
                # here the first match using threshold will be used
                if (score > threshold) and (find_match_flag!=1):
                    r.append(order)
                    find_match_flag = 1
        # if still no match, put 0 to vector
        if find_match_flag != 1:
            r.append(0)
    return r


# calculate word order similarity
def sim_wordorder(sentence1, sentence2, threshold=0.3, metric=wn.path_similarity, ic=None):
    # lowercase
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()
    # remove punctuation
    nopunct_sentence1 = ''.join([c for c in sentence1
                                        if re.match("[a-z\-\' \n\t]", c)])
    nopunct_sentence2 = ''.join([c for c in sentence2 
                                        if re.match("[a-z\-\' \n\t]", c)])
    # tokenize
    line1 = wt(nopunct_sentence1)
    line2 = wt(nopunct_sentence2)
    
    # joint list
    # # Note: set() method is not inplace, 
    # #       however, the calculate of word order vector does not 
    # #       require inplace set of J due to nature of vector modulus
    # J = list(set(line1).union(set(line2)))
    # print J
    # an inplace way to get the joint set:
    combined = line1 + line2
    J = []
    [J.append(x) for x in combined if x not in J]

    r1 = calculate_word_order_vector(J, line1, threshold, metric, ic)
    r2 = calculate_word_order_vector(J, line2, threshold, metric, ic)

    # Similarity calculation given word order vector r1 and r2
    # transfer to array
    x = np.array(r1)
    y = np.array(r2)
    # difference and sum
    diff = x - y
    summ = x + y
    # modulus
    diff_modulus = np.sqrt((diff*diff).sum())
    summ_modulus = np.sqrt((summ*summ).sum())
    # final similarity
    sim = 1 - (diff_modulus/summ_modulus)

    return sim


# Test
# T1 = 'A quick brown dog jumps over the lazy fox.'
# T2 = 'A quick blue fox jumps over the lazy dog.'
# score = sim_wordorder(T1,T2)
# print score

# Test
list1 = load_sentences('data_not_sell')
list2 = load_sentences('data_sell_share')

sentence1 = list1[0]
sentence2 = list2[1]

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

# sim_wordorder(sentence1, sentence2)

score = sim_wordorder(sentence1, sentence2)
print 'path: ', score
score = sim_wordorder(sentence1, sentence2, metric=wn.lch_similarity)
print 'lch : ', score
score = sim_wordorder(sentence1, sentence2, metric=wn.wup_similarity)
print 'wup : ', score
score = sim_wordorder(sentence1, sentence2, metric=wn.res_similarity, ic=brown_ic)
print 'res - brown  : ', score
score = sim_wordorder(sentence1, sentence2, metric=wn.res_similarity, ic=semcor_ic)
print 'res - semcor : ', score
score = sim_wordorder(sentence1, sentence2, metric=wn.jcn_similarity, ic=brown_ic)
print 'jcn : ', score
score = sim_wordorder(sentence1, sentence2, metric=wn.lin_similarity, ic=brown_ic)
print 'lin : ', score

# Sample results:
# sentence1 = list1[0]
# sentence2 = list2[1]
# path:  0.19306769657
# lch :  0.250567652287
# wup :  0.256855951338
# res - brown  :  0.252420961067
# res - semcor :  0.252420961067
# jcn :  0.180138258853
# lin :  0.310539825618