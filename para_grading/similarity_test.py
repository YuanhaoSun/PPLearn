import similarity_overlap
import similarity_overlap_idf
import similarity_utils

from pprint import pprint


# log file
logFile = open('mylogfile.txt', 'wb')

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


# Test for combined_list
combined_list = list1+list2

# Sim_overlap
score_array_overlap = similarity_utils.iterate_combination_2d_sim(combined_list, combined_list, similarity_overlap.sim_overlap)
# pretty print
score_array_overlap = [[similarity_utils.PrettyFloat(n) for n in row] for row in score_array_overlap]
pprint(score_array_overlap, logFile)
logFile.write('\n')

# Sim_overlap_idf
score_array_idf = similarity_utils.iterate_combination_2d_sim(combined_list, combined_list, similarity_overlap_idf.sim_overlap_idf)
normalized = similarity_utils.normalization(score_array_idf)
normalized_sym = similarity_utils.normalization_symmetric(score_array_idf)
# pretty print
score_array_idf = [[similarity_utils.PrettyFloat(n) for n in row] for row in score_array_idf]
pprint(score_array_idf, logFile)
logFile.write('\n')
normalized = [[similarity_utils.PrettyFloat(n) for n in row] for row in normalized]
pprint(normalized, logFile)
logFile.write('\n')
normalized_sym = [[similarity_utils.PrettyFloat(n) for n in row] for row in normalized_sym]
pprint(normalized_sym, logFile)
logFile.write('\n')

logFile.close()