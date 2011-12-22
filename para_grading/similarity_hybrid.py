import numpy as np
from time import time

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.linear_model.sparse import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.lda import LDA
from sklearn.svm.sparse import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.externals import joblib
from sklearn.datasets import load_files
from sklearn import metrics

import similarity_utils
import similarity_overlap
import similarity_overlap_idf
import similarity_overlap_phrasal
import similarity_sem_firstsense_alpha
import similarity_sem_firstsense_pos
import similarity_sem_intermax
import similarity_syntactic_wordorder



# sim_result_array_train = joblib.load('./Sim_results/pickles/intermax_3class_train.pkl')
# sim_result_array_train = joblib.load('./Sim_results/pickles/intermax_8class_train.pkl')
# sim_result_array_train = joblib.load('./Sim_results/pickles/alpha_3class_train.pkl')
# sim_result_array_train = joblib.load('./Sim_results/pickles/alpha_8class_train.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/intermax_3class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/intermax_8class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/alpha_3class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/alpha_8class_test.pkl')




##########
# Hybrid
# 
# Sim = Beta*Semantic + (1-Beta)*Syntactic


# sim_result_array_1 = joblib.load('./Sim_results/pickles/pos_3class_test.pkl')
# sim_result_array_1 = joblib.load('./Sim_results/pickles/pos_8class_test.pkl')
# sim_result_array_2 = joblib.load('./Sim_results/pickles/wordorder_3class_test.pkl')
# sim_result_array_2 = joblib.load('./Sim_results/pickles/wordorder_8class_test.pkl')

# sim_result_array_1 = joblib.load('./Sim_results/pickles/pos_3class_train.pkl')
sim_result_array_1 = joblib.load('./Sim_results/pickles/pos_8class_train.pkl')
# sim_result_array_2 = joblib.load('./Sim_results/pickles/wordorder_3class_train.pkl')
sim_result_array_2 = joblib.load('./Sim_results/pickles/wordorder_8class_train.pkl')

# print sim_result_array_1
# print sim_result_array_2
# print

# Beta = 0.8
Beta = 0.5

newList_1 = [[item*Beta for item in innerlist] for innerlist in sim_result_array_1]
newList_2 = [[item*(1-Beta) for item in innerlist] for innerlist in sim_result_array_2]
# print newList_1
# print newList_2


finalList = [[ x+y for x, y in zip(innerlist_1, innerlist_2)] for innerlist_1, innerlist_2 in zip(newList_1, newList_2)]
# print finalList

# joblib.dump(finalList, './Sim_results/pickles/hybrid_0.8_3class_test.pkl')
# joblib.dump(finalList, './Sim_results/pickles/hybrid_0.8_8class_test.pkl')
# joblib.dump(finalList, './Sim_results/pickles/hybrid_0.8_3class_train.pkl')
# joblib.dump(finalList, './Sim_results/pickles/hybrid_0.8_8class_train.pkl')

# joblib.dump(finalList, './Sim_results/pickles/hybrid_0.5_3class_test.pkl')
# joblib.dump(finalList, './Sim_results/pickles/hybrid_0.5_8class_test.pkl')
# joblib.dump(finalList, './Sim_results/pickles/hybrid_0.5_3class_train.pkl')
joblib.dump(finalList, './Sim_results/pickles/hybrid_0.5_8class_train.pkl')