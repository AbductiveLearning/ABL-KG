import torch
import numpy as np
from torchmetrics import F1Score
from sklearn.metrics import f1_score, average_precision_score, precision_recall_fscore_support

def compute_f1(targets, model_results, threshold = 0):
    # Ensure model_results>0 means positive, otherwise negative
    y_pred = np.array(model_results).copy()
    cond1, cond2 = y_pred>threshold, y_pred<=threshold
    y_pred[cond1] = 1
    y_pred[cond2] = 0
    targets = np.array(targets)
    micro_f1_score = f1_score(y_true=targets, y_pred=y_pred, average='micro')
    macro_f1_score = f1_score(y_true=targets, y_pred=y_pred, average='macro')
    f1_score_list = f1_score(y_true=targets, y_pred=y_pred, average=None)
    precision_list, recall_list, _, _ = precision_recall_fscore_support(y_true=targets, y_pred=y_pred)
    return micro_f1_score, macro_f1_score, f1_score_list, precision_list, recall_list
