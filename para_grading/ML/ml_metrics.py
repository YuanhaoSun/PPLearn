import numpy as np
from sklearn.utils import check_arrays
from sklearn import metrics

def unique_labels(*lists_of_labels):
    """Extract an ordered array of unique labels"""
    labels = set()
    for l in lists_of_labels:
        if hasattr(l, 'ravel'):
            l = l.ravel()
        labels |= set(l)
    return np.unique(sorted(labels))



######################################################
# Weighted

def precision_weighted_score(y_true, y_pred, pos_label=1):

    p, _, _, s = precision_recall_fscore_support_weighted(y_true, y_pred)
    if p.shape[0] == 2:
        return p[pos_label]
    else:
        return np.average(p, weights=s)

def recall_weighted_score(y_true, y_pred, pos_label=1):

    _, r, _, s = precision_recall_fscore_support_weighted(y_true, y_pred)
    if r.shape[0] == 2:
        return r[pos_label]
    else:
        return np.average(r, weights=s)

def fbeta_weighted_score(y_true, y_pred, beta, pos_label=1):

    _, _, f, s = precision_recall_fscore_support_weighted(y_true, y_pred, beta=beta)
    if f.shape[0] == 2:
        return f[pos_label]
    else:
        return np.average(f, weights=s)

def f1_weighted_score(y_true, y_pred, pos_label=1):
 
    return fbeta_weighted_score(y_true, y_pred, 1, pos_label=pos_label)


def precision_recall_fscore_support_weighted(y_true, y_pred, beta=1.0, labels=None):

    y_true, y_pred = check_arrays(y_true, y_pred)
    assert(beta > 0)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    n_labels = labels.size
    true_pos = np.zeros(n_labels, dtype=np.double)
    false_pos = np.zeros(n_labels, dtype=np.double)
    false_neg = np.zeros(n_labels, dtype=np.double)
    support = np.zeros(n_labels, dtype=np.long)

    for i, label_i in enumerate(labels):
        true_pos[i] = np.sum(y_pred[y_true == label_i] == label_i)
        false_pos[i] = np.sum(y_pred[y_true != label_i] == label_i)
        false_neg[i] = np.sum(y_pred[y_true == label_i] != label_i)
        support[i] = np.sum(y_true == label_i)

    try:
        # oddly, we may get an "invalid" rather than a "divide" error here
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        # precision and recall
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)

        # handle division by 0.0 in precision and recall
        precision[(true_pos + false_pos) == 0.0] = 0.0
        recall[(true_pos + false_neg) == 0.0] = 0.0

        # fbeta score
        beta2 = beta ** 2
        fscore = (1 + beta2) * (precision * recall) / (
            beta2 * precision + recall)

        # handle division by 0.0 in fscore
        fscore[(precision + recall) == 0.0] = 0.0
    finally:
        np.seterr(**old_err_settings)

    return precision, recall, fscore, support



######################################################
# micro-averaging

def precision_score(y_true, y_pred, pos_label=1):

    p, _, _, s = precision_recall_fscore_support(y_true, y_pred)
    # if p.shape[0] == 2:
    #     return p[pos_label]
    # else:
    #     return p
    return p

def recall_score(y_true, y_pred, pos_label=1):

    _, r, _, s = precision_recall_fscore_support(y_true, y_pred)
    # if r.shape[0] == 2:
    #     return r[pos_label]
    # else:
    #     return r
    return r

def fbeta_score(y_true, y_pred, beta, pos_label=1):

    _, _, f, s = precision_recall_fscore_support(y_true, y_pred, beta=beta)
    # if f.shape[0] == 2:
    #     return f[pos_label]
    # else:
    #     return f
    return f

def f1_score(y_true, y_pred, pos_label=1):
 
    return fbeta_score(y_true, y_pred, 1, pos_label=pos_label)

def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None):

    y_true, y_pred = check_arrays(y_true, y_pred)
    assert(beta > 0)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    n_labels = labels.size
    true_pos = np.zeros(n_labels, dtype=np.double)
    false_pos = np.zeros(n_labels, dtype=np.double)
    false_neg = np.zeros(n_labels, dtype=np.double)
    support = np.zeros(n_labels, dtype=np.long)

    for i, label_i in enumerate(labels):
        true_pos[i] = np.sum(y_pred[y_true == label_i] == label_i)
        false_pos[i] = np.sum(y_pred[y_true != label_i] == label_i)
        false_neg[i] = np.sum(y_pred[y_true == label_i] != label_i)
        support[i] = np.sum(y_true == label_i)

    try:
        # oddly, we may get an "invalid" rather than a "divide" error here
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        # precision and recall
        # Micro-averaging is used
        precision = true_pos.sum() / (true_pos.sum() + false_pos.sum())
        recall = true_pos.sum() / (true_pos.sum() + false_neg.sum())

        # print false_pos
        # print false_neg
        # print false_pos.sum()
        # print false_neg.sum()

        # # handle division by 0.0 in precision and recall
        # precision[(true_pos + false_pos) == 0.0] = 0.0
        # recall[(true_pos + false_neg) == 0.0] = 0.0

        # fbeta score
        beta2 = beta ** 2
        fscore = (1 + beta2) * (precision * recall) / (
            beta2 * precision + recall)
        # print (beta2 * precision + recall)

        # handle division by 0.0 in fscore
        if (precision + recall) == 0.0:
            fscore = 0.0
        # fscore[(precision + recall) == 0.0] = 0.0
    finally:
        np.seterr(**old_err_settings)

    return precision, recall, fscore, support




######################################################
# macro-averaging

def precision_macro_score(y_true, y_pred, pos_label=1):

    p, _, _, s = precision_recall_fscore_support_macro(y_true, y_pred)
    # if p.shape[0] == 2:
    #     return p[pos_label]
    # else:
    #     return p
    return p

def recall_macro_score(y_true, y_pred, pos_label=1):

    _, r, _, s = precision_recall_fscore_support_macro(y_true, y_pred)
    # if r.shape[0] == 2:
    #     return r[pos_label]
    # else:
    #     return r
    return r

def fbeta_macro_score(y_true, y_pred, beta, pos_label=1):

    _, _, f, s = precision_recall_fscore_support_macro(y_true, y_pred, beta=beta)
    # if f.shape[0] == 2:
    #     return f[pos_label]
    # else:
    #     return f
    return f

def f1_macro_score(y_true, y_pred, pos_label=1):

    return fbeta_macro_score(y_true, y_pred, 1, pos_label=pos_label)

def precision_recall_fscore_support_macro(y_true, y_pred, beta=1.0, labels=None):

    y_true, y_pred = check_arrays(y_true, y_pred)
    assert(beta > 0)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels, dtype=np.int)

    n_labels = labels.size
    true_pos = np.zeros(n_labels, dtype=np.double)
    false_pos = np.zeros(n_labels, dtype=np.double)
    false_neg = np.zeros(n_labels, dtype=np.double)
    support = np.zeros(n_labels, dtype=np.long)

    for i, label_i in enumerate(labels):
        true_pos[i] = np.sum(y_pred[y_true == label_i] == label_i)
        false_pos[i] = np.sum(y_pred[y_true != label_i] == label_i)
        false_neg[i] = np.sum(y_pred[y_true == label_i] != label_i)
        support[i] = np.sum(y_true == label_i)

    try:
        # oddly, we may get an "invalid" rather than a "divide" error here
        old_err_settings = np.seterr(divide='ignore', invalid='ignore')

        # precision and recall
        # Macro-averaging is used
        precision_in_process = (true_pos / (true_pos + false_pos))
        recall_in_process = (true_pos / (true_pos + false_neg))

        # handle division by 0.0 in precision and recall
        precision_in_process[(true_pos + false_pos) == 0.0] = 0.0
        recall_in_process[(true_pos + false_neg) == 0.0] = 0.0

        precision = np.mean(precision_in_process)
        recall = np.mean(recall_in_process)
        
        # fbeta score
        beta2 = beta ** 2
        fscore = (1 + beta2) * (precision * recall) / (
            beta2 * precision + recall)

        # handle division by 0.0 in fscore
        if (precision + recall) == 0.0:
            fscore = 0.0
        # fscore[(precision + recall) == 0.0] = 0.0

    finally:
        np.seterr(**old_err_settings)

    return precision, recall, fscore, support




# # Test using the example in LinePipe tutorial Fig 8.1-8.3
# y_list    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
# # pred_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2, 2]
# pred_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2]
# y = np.array(y_list)
# pred = np.array(pred_list)

# print metrics.confusion_matrix(y, pred)
# print

# print "Wei Pre:", precision_weighted_score(y, pred)
# print "Wei Rec:", recall_weighted_score(y, pred)
# print "Wei F1 :", f1_weighted_score(y, pred)
# print "Wei F5 :", fbeta_weighted_score(y, pred, 0.5)
# print

# print "Mic Pre:", precision_score(y, pred)
# print "Mic Rec:", recall_score(y, pred)
# print "Mic F1 :", f1_score(y, pred)
# print "Mic F5 :", fbeta_score(y, pred, 0.5)
# print

# print "Mac Pre:", precision_macro_score(y, pred)
# print "Mac Rec:", recall_macro_score(y, pred)
# print "Mac F1 :", f1_macro_score(y, pred)
# print "Mac F5 :", fbeta_macro_score(y, pred, 0.5)

