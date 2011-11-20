import re
from math import tanh

from nltk import word_tokenize as wt
from nltk import collocations

from similarity_utils import load_sentences

bigram_measures = collocations.BigramAssocMeasures()
trigram_measures = collocations.TrigramAssocMeasures()

# Calculate sentence similarity base on phrasal overlap, i.e.
# Sim = \sum_n m^2
# Then, normalize with lenghts of sentences and tanh function
def sim_overlap_phrasal(sentence1, sentence2): 
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

    # finders for bigram and trigram
    finder_bi_line1 = collocations.BigramCollocationFinder.from_words(line1)
    finder_bi_line2 = collocations.BigramCollocationFinder.from_words(line2)
    finder_tri_line1 = collocations.TrigramCollocationFinder.from_words(line1)
    finder_tri_line2 = collocations.TrigramCollocationFinder.from_words(line2)
    # find bigram / trigram
    scored_bi_line1 = finder_bi_line1.score_ngrams(bigram_measures.raw_freq)
    scored_bi_line2 = finder_bi_line2.score_ngrams(bigram_measures.raw_freq)
    scored_tri_line1 = finder_tri_line1.score_ngrams(bigram_measures.raw_freq)
    scored_tri_line2 = finder_tri_line2.score_ngrams(bigram_measures.raw_freq)
    # generate lists contain all the bigram or trigram for line1 and line2
    list_bi_line1 = sorted(bigram for bigram, score in scored_bi_line1)
    list_bi_line2 = sorted(bigram for bigram, score in scored_bi_line2)
    list_tri_line1 = sorted(trigram for trigram, score in scored_tri_line1)
    list_tri_line2 = sorted(trigram for trigram, score in scored_tri_line2)
    # find the common elements from two sets of bigram in two sentences
    common_set_bi = [i for i in list_bi_line1 if i in list_bi_line2]
    common_set_tri = [i for i in list_tri_line1 if i in list_tri_line2]

    # Calculate element numbers of intersection and sentence1
    # combined_line = line1 + line2
    # union_num = len(set(combined_line))
    intersection_len = len(set(line1) & set(line2))
    sentence1_len = len(set(line1))
    sentence2_len = len(set(line1))

    # Overlap (phrasal) score
    # Note, here we only consider trigram and bigram
    overlap_score = 9*len(common_set_tri) + 4*len(common_set_bi) + intersection_len

    # Normalization as defined in Ponzetto et al. 2007
    sim = float(overlap_score) / (sentence1_len+sentence2_len)
    sim = tanh(sim)

    return sim




# # Test
# list1 = load_sentences('data_not_sell')
# list2 = load_sentences('data_sell_share')

# sentence1 = list1[1]
# sentence2 = list2[3]

# score = sim_overlap_phrasal(sentence1, sentence2)
# print score