import re

from nltk.corpus import wordnet as wn 
from nltk import word_tokenize as wt
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic

from similarity_utils import load_sentences

# Calculate sentence semantic similarity base on first sense heuristic using POS
def sim_sem_firstsense_pos(sentence1, sentence2, metric=wn.path_similarity, ic=None):
    
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
        if tag.startswith('V') or tag.startswith('N') or tag.startswith('R') or tag.startswith('J'):
            filtered_line1.append(term)
    for tagged_tuple in pos_line2:
        term = tagged_tuple[0]
        tag  = tagged_tuple[1]
        # find out all verbs, nouns, adverbs, adjectives
        if tag.startswith('V') or tag.startswith('N') or tag.startswith('R') or tag.startswith('J'):
            filtered_line2.append(term)

    # get list of synsets only using first senses, without stopword elimination
    # synset_list1 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in filtered_line1 if wn.synsets(x) ])
    # synset_list2 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in filtered_line2 if wn.synsets(x) ])
    # # get the synset list for each sentence, containing all WordNet senses
    # # with stopword elimination
    # synset_list1 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in line1 if x not in sw])
    # synset_list2 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in line2 if x not in sw])
    # # get list of synsets only using first senses, with stopword elimination
    synset_list1 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in line1 if ((x not in sw) and wn.synsets(x)) ])
    synset_list2 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in line2 if ((x not in sw) and wn.synsets(x)) ])

    runningscore = 0.0
    runningcount = 0
    # get Wordnet similarity score for <metric> for each pair created from both synset lists 
    for synset1 in set(synset_list1): 
        for synset2 in set(synset_list2): 
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
                runningscore += mark
            else: 
                try:
                    mark = metric(synset1, synset2)
                    if mark is None:
                        mark = 0.0
                except:
                    mark = 0.0
                runningscore += mark
            runningcount += 1
    
    # add up individual scores, divide by number of individual scores 
    sim = runningscore/runningcount 
    return sim


# Test
list1 = load_sentences('data_not_sell')
list2 = load_sentences('data_sell_share')

sentence1 = list1[0]
sentence2 = list2[1]

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

# sim_sem_firstsense_pos(sentence1, sentence2)

score = sim_sem_firstsense_pos(sentence1, sentence2)
print 'path: ', score
score = sim_sem_firstsense_pos(sentence1, sentence2, metric=wn.lch_similarity)
print 'lch : ', score
score = sim_sem_firstsense_pos(sentence1, sentence2, metric=wn.wup_similarity)
print 'wup : ', score
score = sim_sem_firstsense_pos(sentence1, sentence2, metric=wn.res_similarity, ic=brown_ic)
print 'res - brown  : ', score
score = sim_sem_firstsense_pos(sentence1, sentence2, metric=wn.res_similarity, ic=semcor_ic)
print 'res - semcor : ', score
score = sim_sem_firstsense_pos(sentence1, sentence2, metric=wn.jcn_similarity, ic=brown_ic)
print 'jcn : ', score
score = sim_sem_firstsense_pos(sentence1, sentence2, metric=wn.lin_similarity, ic=brown_ic)
print 'lin : ', score

# Sample results:
# sentence1 = list1[0]
# sentence2 = list2[1]
# 
# Without stopword:
# path:  0.0940466778868
# lch :  0.887929847504
# wup :  0.219807728866
# res - brown  :  0.848695269318
# res - semcor :  0.865603927843
# jcn :  0.0836914661059
# lin :  0.10223230793
# 
# With stopword:
# path:  0.119400470224
# lch :  1.0397796603
# wup :  0.261290110119
# res - brown  :  1.08493178733
# res - semcor :  1.07913738215
# jcn :  0.107552064251
# lin :  0.129823958399