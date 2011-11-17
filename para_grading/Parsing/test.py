import sys
sys.path.append('./stanford-parser.jar') # path to stanford-parser.jar
 
from java.io import CharArrayReader
from edu.stanford.nlp import *

# PrintWriter is used to write TreePrint
# result to file
from java.io import PrintWriter, FileWriter, BufferedWriter

# Initialize the LexicalizedParser
lp = parser.lexparser.LexicalizedParser('./englishPCFG.ser.gz') # path to englishPCFG.ser.gz
tlp = trees.PennTreebankLanguagePack()
lp.setOptionFlags(["-maxLength", "80", "-retainTmpSubcategories"])
 
sentence = "Yahoo! does not rent, sell, or share personal information \
			about you with other people or non-affiliated companies except \
			to provide products or services you've requested, when we have \
			your permission, or under the following circumstances"
 
toke = tlp.getTokenizerFactory().getTokenizer(CharArrayReader(sentence));
wordlist = toke.tokenize()

# parse and get the tree
if (lp.parse(wordlist)):
	parse = lp.getBestParse()

# ###########################
# use trees.TreePrint Class to get three formats of results
# 
# Possible options:
# http://nlp.stanford.edu/nlp/javadoc/javanlp/edu/stanford/nlp/parser/lexparser/package-summary.html
# See section Output formatting options
# 
tp = trees.TreePrint("wordsAndTags,penn,typedDependenciesCollapsed")
# tp = trees.TreePrint("wordsAndTags,penn,typedDependencies")

output = PrintWriter(BufferedWriter(FileWriter("parsed.txt")))
tp.printTree(parse, output)


# Using separate methods to get the penn tree
# and typedDependenciesCollapsed
# 
# gsf = tlp.grammaticalStructureFactory()
# gs = gsf.newGrammaticalStructure(parse)
# tdl = gs.typedDependenciesCollapsed()

# # print parse
# f = open("parsed.txt","wb")
# f.write(str(parse))
# f.write('\n')
# # print str(parse)
# # print tdl
# f.write(str(tdl))
# f.close()


