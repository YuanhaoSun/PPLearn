import re

from nltk.corpus import wordnet as wn 
from nltk import word_tokenize as wt 
from nltk.corpus import stopwords
from nltk.corpus import wordnet_ic

from similarity_utils import load_sentences

# Calculate sentence semantic similarity base on first sense heuristic without alpha
def sim_sem_firstsense(sentence1, sentence2, metric=wn.path_similarity, ic=None):
    
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

    # get list of synsets only using first senses, without stopword elimination
    synset_list1 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in line1 if wn.synsets(x) ])
    synset_list2 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in line2 if wn.synsets(x) ])
    # # get the synset list for each sentence, containing all WordNet senses
    # # with stopword elimination
    # synset_list1 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in line1 if x not in sw])
    # synset_list2 = reduce(lambda x,y:x+y,[wn.synsets(x) for x in line2 if x not in sw])
    # # get list of synsets only using first senses, with stopword elimination
    # synset_list1 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in line1 if ((x not in sw) and wn.synsets(x)) ])
    # synset_list2 = reduce(lambda x,y:x+y, [ [wn.synsets(x)[0]] for x in line2 if ((x not in sw) and wn.synsets(x)) ])

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
sentence2 = list2[0]

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')

score = sim_sem_firstsense(sentence1, sentence2)
print 'path: ', score
score = sim_sem_firstsense(sentence1, sentence2, metric=wn.lch_similarity)
print 'lch : ', score
score = sim_sem_firstsense(sentence1, sentence2, metric=wn.wup_similarity)
print 'wup : ', score
score = sim_sem_firstsense(sentence1, sentence2, metric=wn.res_similarity, ic=brown_ic)
print 'res - brown  : ', score
score = sim_sem_firstsense(sentence1, sentence2, metric=wn.res_similarity, ic=semcor_ic)
print 'res - semcor : ', score
score = sim_sem_firstsense(sentence1, sentence2, metric=wn.jcn_similarity, ic=brown_ic)
print 'jcn : ', score
score = sim_sem_firstsense(sentence1, sentence2, metric=wn.lin_similarity, ic=brown_ic)
print 'lin : ', score