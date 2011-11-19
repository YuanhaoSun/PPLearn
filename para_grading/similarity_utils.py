from pprint import pprint

# Read lines from a pre-defined txt file for one category of 
# share statements and convert the lines into a list
# 
# Preprocessing: lowercase, strip '\r\n' at the end
# 
# Return the list
def load_sentences(file_name):
    sentence_list = []
    f = open('./Dataset/'+file_name+'.txt', 'rb')
    for line in f.readlines():
        line = line.rstrip("\r\n")  # remove '\r\n' at the end of a sentence
        line = line.lower()         # lowercase
        sentence_list.append(line)
    f.close()
    return sentence_list


def iterate_combination_2d_sim(sentence_list_p, sentence_list_q, similarity_measure):
    result = []
    for i, sentence_p in enumerate(sentence_list_p):
        result.append([])
        for j, sentence_q in enumerate(sentence_list_q):
            result[i].append(similarity_measure(sentence_p, sentence_q))
    print len(result)
    print len(result[0])
    pprint(result)


# Iterate over two given lists, find all combinations
# Derived from http://docs.python.org/library/itertools.html#itertools.product
# 
# Return a 2d array with [setence_p_n, setence_q_m] tuples
# 
def iterate_combination_2darray(sentence_list_p, sentence_list_q):
    result = []
    for i, sentence_p in enumerate(sentence_list_p):
        result.append([])
        for j, sentence_q in enumerate(sentence_list_q):
        	result[i].append([sentence_p, sentence_q])
    print len(result)
    print len(result[0])
    print result[0][1]


# Iterate over two given lists, find all combinations
# Derived from http://docs.python.org/library/itertools.html#itertools.product
# 
# Return a generator function, with next() method to iterate all lists
# 
def iterate_combination(*sentence_list_p, **sentence_list_q):
    pools = map(tuple, sentence_list_p) * sentence_list_q.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
    	yield prod