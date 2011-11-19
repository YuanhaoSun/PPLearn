

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