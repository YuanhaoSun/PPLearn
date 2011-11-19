import similarity_overlap
import similarity_overlap_idf
import similarity_utils

from pprint import pprint


list1 = similarity_utils.load_sentences('data_not_sell')
list2 = similarity_utils.load_sentences('data_sell_share')
print "len(list1):", len(list1)
print "len(list2):", len(list2)

sentence1 = list1[1]
sentence2 = list2[2]

# test similarity from sentence
score1 = similarity_overlap.sim_overlap(sentence1, sentence2)
print "sim(list1[1], list2[2])  :", score1

# test iterate_combination_2d_sim
score_array = similarity_utils.iterate_combination_2d_sim(list1, list2, similarity_overlap.sim_overlap)
print "score_array[1][2]        :", score_array[1][2]

score_array_idf = similarity_utils.iterate_combination_2d_sim(list1, list2, similarity_overlap_idf.sim_overlap_idf)
pprint(score_array_idf)

normalized = similarity_utils.normalization(score_array_idf)
pprint(normalized)