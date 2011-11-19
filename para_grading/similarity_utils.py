import itertools


# Read lines from a pre-defined txt file for one category of 
# share statements and convert the lines into a list
# 
# Preprocessing: lowercase, strip '\r\n' at the end
# 
# Return the list
# 
def load_sentences(file_name):
    sentence_list = []
    f = open('./Dataset/'+file_name+'.txt', 'rb')
    for line in f.readlines():
        line = line.rstrip("\r\n")  # remove '\r\n' at the end of a sentence
        line = line.lower()         # lowercase
        sentence_list.append(line)
    f.close()
    return sentence_list


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


# Iterate over two given lists, find all combinations
# 
# Return a 2d array with [setence_p_n, setence_q_m] tuples
# 
def iterate_combination_2darray(sentence_list_p, sentence_list_q):
    result = []
    for i, sentence_p in enumerate(sentence_list_p):
        result.append([])
        for j, sentence_q in enumerate(sentence_list_q):
        	result[i].append([sentence_p, sentence_q])
    return result


# Iterate over two given lists, len(list1)=m, len(list2)=n,
# find all combinations, then calculate the similarity using given measure
# 
# Return a 2d m-by-n array with scores - sim[setence_p_n, setence_q_m]
# 
def iterate_combination_2d_sim(sentence_list_p, sentence_list_q, similarity_measure):
    result = []
    for i, sentence_p in enumerate(sentence_list_p):
        result.append([])
        for j, sentence_q in enumerate(sentence_list_q):
            result[i].append(similarity_measure(sentence_p, sentence_q))
    return result


# Normalize array to range [0,1]
# normalized value = (x-min)/(max-min)
# 
# Return a normalized 2d score array
# 
def normalization(score_2d_array):
    # generate a 1d merged list from the 2d array
    # for getting max and min
    merged = list(itertools.chain(*score_2d_array))
    min_value = min(merged)
    max_value = max(merged)
    print 'min: ', min_value
    print 'max: ', max_value
    max_min_diff = max_value - min_value

    # put normalized value back into a new 2d array
    normalized_2d_array = [[((item-min_value)/max_min_diff) for item in row] for row in score_2d_array]
    # # The above one-line list comprehension equals to the five lines below
    # normalized_2d_array = []
    # for i, row in enumerate(score_2d_array):
    #     normalized_2d_array.append([])
    #     for j, item in enumerate(row):
    #         normalized_2d_array[i].append((item-min_value)/max_min_diff)

    return normalized_2d_array