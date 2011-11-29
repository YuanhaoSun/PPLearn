import re

from nltk.corpus import wordnet as wn 
from nltk import word_tokenize as wt
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic

from similarity_utils import load_sentences




# Auxiliary function to compute list of max scores
# 
# Parameters: 
#   filtered_line1  (POS filtered list of terms of sentence 1)
#   synset_list2    (All synsets of all terms from filtered_line2)
# 
# Returns: the list of max scores for each term in sentence 1

def inter_sentence_max(filtered_line1, synset_list2, metric, ic):

    # iterate to get a list of all max similarity for all words in sentence 1
    max_scores_list1 = []
    for term in filtered_line1:
        # for each term, create a list to contain all scores in order to get the max
        score_list1 = []
        term_synsets = wn.synsets(term)
        # a term may have more than one synsets, so need to compute all
        for term_synset in term_synsets:
            # compute similarity using current synset of term and iterate over all synsets in sentence 2
            for synset2 in synset_list2:
                if ic is not None:
                    try:
                        mark = metric(term_synset, synset2, ic)
                        if mark is None:
                            mark = 0.0
                    except: 
                        mark = 0.0
                    # handle infinitity mark for jcn measure
                    if mark == 1e+300:
                        mark = 1.0
                else: 
                    try:
                        mark = metric(term_synset, synset2)
                        if mark is None:
                            mark = 0.0
                    except:
                        mark = 0.0
                score_list1.append(mark)
        # now, all scores is in sort to get the max on top
        score_list1.sort()
        score_list1.reverse()
        # put the one max score into the max list for this one term
        max_scores_list1.append(score_list1[0])

    # # need to consider if this removal is legitimate...
    # # eliminate 0 from max score list
    # max_scores_list1 = [x for x in max_scores_list1 if x != 0.0]
    return max_scores_list1




# Calculate sentence semantic similarity base on first sense heuristic without alpha
def sim_sem_intermax(sentence1, sentence2, metric=wn.path_similarity, ic=None):
    
    # Bug fix: lower
    sentence1 = sentence1.lower()
    sentence2 = sentence2.lower()
    # import stopwords 
    sw = stopwords.words('english')
    # remove punctuation
    nopunct_sentence1 = ''.join([c for c in sentence1 
                                        if re.match("[a-z\-\' \n\t]", c)])
    nopunct_sentence2 = ''.join([c for c in sentence2 
                                        if re.match("[a-z\-\' \n\t]", c)])                                         
    # tokenize
    line1 = wt(nopunct_sentence1)
    line2 = wt(nopunct_sentence2)
    # POS 
    pos_line1 = pos_tag(line1)
    pos_line2 = pos_tag(line2)

    # filter line1 and line2 using POS info
    # only remain verbs, nouns, adverbs, adjectives
    filtered_line1 = []
    filtered_line2 = []
    for tagged_tuple in pos_line1:
        term = tagged_tuple[0]
        tag  = tagged_tuple[1]
        # find out all verbs, nouns, adverbs, adjectives
        # in the meanwhile get rid of terms that do not appear in WordNet
        if (tag.startswith('V') or tag.startswith('N') or tag.startswith('R') or tag.startswith('J')) and wn.synsets(term):
            filtered_line1.append(term)
    for tagged_tuple in pos_line2:
        term = tagged_tuple[0]
        tag  = tagged_tuple[1]
        # find out all verbs, nouns, adverbs, adjectives
        # in the meanwhile get rid of terms that do not appear in WordNet
        if (tag.startswith('V') or tag.startswith('N') or tag.startswith('R') or tag.startswith('J')) and wn.synsets(term):
            filtered_line2.append(term)

    # get the synset list for each sentence, containing all WordNet senses
    # without stopword elimination
    synset_list1 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in filtered_line1])
    synset_list2 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in filtered_line2])
    # # # with stopword elimination
    # # synset_list1 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in filtered_line1 if x not in sw])
    # # synset_list2 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in filtered_line2 if x not in sw])

    # get max score lists using the inter max function defined above
    max_score_list1 = inter_sentence_max(filtered_line1, synset_list2, metric=metric, ic=ic)
    max_score_list2 = inter_sentence_max(filtered_line2, synset_list1, metric=metric, ic=ic)

    sim = (sum(max_score_list1) + sum(max_score_list2)) / (len(max_score_list1) + len(max_score_list2))
    return sim


# Test
list1 = load_sentences('data_not_sell')
list2 = load_sentences('data_sell_share')

sentence1 = list1[0]
sentence2 = list2[1]

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

# sim_sem_intermax(sentence1, sentence2)

score = sim_sem_intermax(sentence1, sentence2)
print 'path: ', score
score = sim_sem_intermax(sentence1, sentence2, metric=wn.lch_similarity)
print 'lch : ', score
score = sim_sem_intermax(sentence1, sentence2, metric=wn.wup_similarity)
print 'wup : ', score
score = sim_sem_intermax(sentence1, sentence2, metric=wn.res_similarity, ic=brown_ic)
print 'res - brown  : ', score
score = sim_sem_intermax(sentence1, sentence2, metric=wn.res_similarity, ic=semcor_ic)
print 'res - semcor : ', score
score = sim_sem_intermax(sentence1, sentence2, metric=wn.jcn_similarity, ic=brown_ic)
print 'jcn : ', score
score = sim_sem_intermax(sentence1, sentence2, metric=wn.lin_similarity, ic=brown_ic)
print 'lin : ', score

# Sample results:
# sentence1 = list1[0]
# sentence2 = list2[1]
# path:  0.511742424242
# lch :  2.37924751823
# wup :  0.715648844878
# res - brown  :  5.9252699315
# res - semcor :  6.82379313536
# jcn :  0.693656881745
# lin :  0.662626674403