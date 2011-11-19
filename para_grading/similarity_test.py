import similarity_overlap
import similarity_overlap_idf

import similarity_utils


list1 = similarity_utils.load_sentences('data_not_sell')
list2 = similarity_utils.load_sentences('data_sell_share')

sentence1 = list1[0]
sentence2 = list2[1]

# test two overlap similarity measures
score1 = similarity_overlap.sim_overlap(sentence1, sentence2)
score2 = similarity_overlap_idf.sim_overlap_idf(sentence1, sentence2)



# test  iterate_combination_2d_sim
similarity_utils.iterate_combination_2d_sim(list1, list2, similarity_overlap.sim_overlap)


# sentence1 = list1[0]
# sentence2 = list2[0]

# score = sim_overlap_idf(sentence1, sentence2)
# print score