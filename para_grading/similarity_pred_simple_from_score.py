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



# sim_result_array = joblib.load('./Sim_results/pickles/wordorder_3class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/wordorder_8class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/intermax_3class_test.pkl')
sim_result_array = joblib.load('./Sim_results/pickles/intermax_8class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/pos_3class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/pos_8class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/alpha_3class_test.pkl')
# sim_result_array = joblib.load('./Sim_results/pickles/alpha_8class_test.pkl')


def predict(Dataset_train, Dataset_test):
    """
    Generate predict results, given two datasets and similarity_measure

    This is the first predict scheme, without using any machine learning techniques

    Return a list of predicted labels, same as pred from scikit-learn classifier.predict(X)
    """
    data_train = Dataset_train
    data_test = Dataset_test
    # placeholder for final prediction results
    pred = []
    # get the sizes for further usages
    train_size = len(data_train.data)
    test_size  = len(data_test.data)
    # using utility to get a n*m array of similarity scores
    sim_result = sim_result_array
    # sim_result = similarity_utils.iterate_combination_2d_sim(data_test.data, data_train.data, similarity_measure)

    for i in range(test_size):
        # list for holding sum of scores on each category
        # note: scores in this list are not averaged but sum
        category_scores = [0] * len(set(data_train.target))
        # get a list of sim results from ith row in the sim_result 2d array 
        # i.e. sim scores for one item in the test set compared to all train sentencs
        score_list = sim_result[i]

        # iterate over the score_list and sum up all scores by their categories into category_scores
        for j in range(train_size):
            # j's category
            j_cate = data_train.target[j]
            # j's sim score
            j_score = score_list[j]
            # add score by category back to the score sum list
            category_scores[j_cate] += j_score
        
        # after adding all scores to category score list, first average
        for category in set(data_train.target):
            # numpy array to list, otherwise cannot count
            # train_targets_list = map(None, data_train.target)
            train_targets_list = data_train.target.tolist()
            # occurance of this category in training set
            occurance = train_targets_list.count(category)
            # average score in the list
            category_scores[category] = category_scores[category] / float(occurance)

        # get the category of highest sim score
        highest_cate = category_scores.index(max(category_scores))
        # append to the pred[]
        pred.append(highest_cate)
    
    # convert to numpy array
    pred_array = np.array(pred)
    return pred_array


# Load categories
categories = ['nolimitshare','notsell', 'notsellnotshare', 'notsharemarketing', 'sellshare', 
            'shareforexception', 'shareforexceptionandconsent','shareonlyconsent']
categories3 = ['good','neutral', 'bad']
# Load data
data_train = load_files('./Dataset/ShareStatement/raw', categories = categories,
                        shuffle = True, random_state = 42)
data_test = load_files('./Dataset/ShareStatement/test', categories = categories, 
                        shuffle = True, random_state = 42)
# data_train = load_files('./Dataset/ShareStatement3/raw', categories = categories3,
                        # shuffle = True, random_state = 42)
# data_test = load_files('./Dataset/ShareStatement3/test', categories = categories3, 
                        # shuffle = True, random_state = 42)
y_test = data_test.target

# Construct training from train data
pred = predict(data_train, data_test)

print y_test
print pred

# acc_score = metrics.zero_one_score(y_test, pred)
pre_score = metrics.precision_score(y_test, pred)
rec_score = metrics.recall_score(y_test, pred)
f1_score = ((2*pre_score*rec_score) / (pre_score+rec_score))
f5_score = ((1.25*pre_score*rec_score) / (0.25*pre_score+rec_score))
print "f1-score  :   %0.5f" % f1_score
print "f0.5-score:   %0.5f" % f5_score
print "precision :   %0.5f" % pre_score
print "recall    :   %0.5f" % rec_score