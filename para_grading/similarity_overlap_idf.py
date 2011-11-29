import re
from nltk import word_tokenize as wt
from nltk.text import TextCollection

from similarity_utils import load_sentences

# create the textcollection for calculation of IDF
list_all_sentences = load_sentences('train_all')
text_collection = TextCollection(list_all_sentences)


# Calculate sentence similarity base on overlap_idf, i.e.
# Sim = ( |Q intersect R| / |Q| ) * Sum(idf_w) for w in (Q intersect R)
def sim_overlap_idf(sentence1, sentence2): 
    # remove punctuation
    nopunct_sentence1 = ''.join([c for c in sentence1 
                                        if re.match("[a-z\-\' \n\t]", c)])
    nopunct_sentence2 = ''.join([c for c in sentence2 
                                        if re.match("[a-z\-\' \n\t]", c)])                                         
    # tokenize
    line1 = wt(nopunct_sentence1)
    line2 = wt(nopunct_sentence2)
    
    # intersection: (Q intersect R)
    intersection = set(line1) & set(line2)
    # calculate sum of idfs: Sum(idf_w) for w in (Q intersect R)
    sum_idf = 0.0
    for item in intersection:
        idf = text_collection.idf(item)
        sum_idf += idf
    
    # Calculate element numbers of intersection and sentence1
    intersection_num = len(intersection)
    sentence1_num = len(set(line1))
    # sim = |Q intersect R| / |Q|
    sim = float(intersection_num) / float(sentence1_num)
    # sim = ( |Q intersect R| / |Q| ) * Sum(idf_w) for w in (Q intersect R)
    sim = sim * sum_idf
    return sim


# # Test
# list1 = load_sentences('data_not_sell')
# list2 = load_sentences('data_sell_share')

# sentence1 = list1[0]
# sentence2 = list2[0]

# score = sim_overlap_idf(sentence1, sentence2)
# print score