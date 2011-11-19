import re
from nltk import word_tokenize as wt

import similarity_utils

# Calculate sentence similarity base on overlap, i.e.
# Sim = |Q intersect R| / |Q|
def sim_overlap(sentence1, sentence2): 
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
    # Calculate element numbers of intersection and sentence1
    # combined_line = line1 + line2
    # union_num = len(set(combined_line))
    intersection_num = len(set(line1) & set(line2))
    sentence1_num = len(set(line1))
    # return score = |Q intersect R| / |Q|
    sim = float(intersection_num) / float(sentence1_num)
    return sim


# Test
list1 = similarity_utils.load_sentences('data_not_sell')
list2 = similarity_utils.load_sentences('data_sell_share')

sentence1 = list1[0]
sentence2 = list2[0]

score = sim_overlap(sentence1, sentence2)
print score