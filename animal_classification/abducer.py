from checkrules import CheckRules
import numpy as np
from itertools import combinations
import time 
from multiprocessing import Pool

# Invert the value of label whose positions in pos_list
# Ensure indexs in pos_list have the same number of element
# Ensuure the value of label is 0 or 1
def get_invert_label(label, pos_list):
    
    # for pos in pos_list:
    #     idx_val = label[list(pos)]
    #     abduced_label = label.copy()
    #     abduced_label[list(pos)] = 1 - idx_val
    #     print(pos, abduced_label)
    invert_labels = np.array([label]).repeat(len(pos_list), axis=0)
    idxs_0 = np.expand_dims(np.arange(len(invert_labels)), axis=1)
    idxs_1 = np.array(pos_list)
    invert_labels[idxs_0, idxs_1] = 1 - invert_labels[idxs_0, idxs_1]
    return invert_labels

# Generate candidate labels of label, where the number of revised positions between min_address_num and max_address_num
# Ensure the value of label is 0 or 1
def gen_cand_labels(label, max_address_num, min_address_num = 0):
    cand_label_list = []
    for address_num in range(min_address_num, max_address_num+1):
        if address_num == 0:
            cand_label_list.append(label)
            continue
        pos_list = list(combinations(range(len(label)), address_num))
        invert_labels = get_invert_label(label, pos_list)
        cand_label_list.extend(invert_labels)
    return np.array(cand_label_list)

class Abducer():
    def __init__(self, check, alpha = 1):
        self.check = check
        self.alpha = alpha
    
    # Ensure:
    #     1. Rules only include relevant atoms (e.g., by forgetting)
    #     2. No subsumption relation for each rule pair (otherwise, the weights of rules are inaccurate)
    # Consistency: (sum of prob) - alpha * (weights of instantiated violated rules)
    # Or: (sum of prob) + alpha * (weights of not violated rules)
    def abduce_multiclass(self, x, y_prob, class_names):
        if class_names is None:
            class_names = list(range(len(y_prob)))
        abduced_class = -1
        max_score = -1
        for i in range(len(y_prob)): # For every abduced result
            onehot = np.eye(len(y_prob))[i]
            model_score = y_prob[i]
            rule_score = self.check.judge([x], [onehot])[0]
            score = model_score + self.alpha * rule_score
            if score > max_score:
                max_score = score
                abduced_class = class_names[i]
        return abduced_class

    # Sum of confidence of 0-1 labels
    # y_prob shape: n_labels
    # label shape: n_samples, n_labels
    # output shape: n_samples
    def get_model_scores(self, y_prob, cand_labels):
        y_prob = np.array(y_prob)
        # score_1_old = np.sum(y_prob[label==1])
        # score_0_old = np.sum(1-y_prob[label==0])
        score_1 = np.matmul(cand_labels, y_prob)
        score_0 = np.matmul(1-cand_labels, 1-y_prob)
        return score_1 + score_0

    # Ensure:
    #     1. Rules only include relevant atoms (e.g., by forgetting)
    #     2. No subsumption relation for each rule pair (otherwise, the weights of rules are inaccurate)
    # Consistency: (sum of prob) - alpha * (weights of instantiated violated rules)
    # Or: (sum of prob) + alpha * (weights of not violated rules)
    def abduce_multilabel(self, x, y_prob, y_pred, max_address_num = 3):
        # print("x, y_prob, y_pred", x, y_prob, y_pred)
        max_address_num = min(len(y_pred), max_address_num)
        abduced_label = None
        cand_labels = gen_cand_labels(label=y_pred, max_address_num=max_address_num)
        model_scores = self.get_model_scores(y_prob=y_prob, cand_labels=cand_labels)
        rule_scores = self.check.judge([x for _ in range(len(cand_labels))], cand_labels)
        scores = model_scores + self.alpha * rule_scores
        abduced_label = cand_labels[np.argmax(scores)]
        # for cand_label, model_score, rule_score, score in zip(cand_labels, model_scores, rule_scores, scores): # For every abduced result
        #     print(y_prob, cand_label, model_score, rule_score, score)
        return abduced_label

    def abduce_multilabel_wrapper(self, args):
        x, y_prob, y_pred, max_address_num = args[0], args[1], args[2], args[3]
        return self.abduce_multilabel(x=x, y_prob=y_prob, y_pred=y_pred, max_address_num = max_address_num)

    def abduce_batch(self, task, X, y_probs, class_names = None, y_preds = None, max_address_num = 3):
        ret = []
        assert(task=='multiclass' or task=='multilabel')
        if task == 'multiclass':
            if class_names is None:
                class_names = list(range(len(y_probs[0])))
            for x, y_prob in zip(X, y_probs):
                ret.append(self.abduce_multiclass(x=x, y_prob=y_prob, class_names=class_names))
        elif task=='multilabel': # Ensure y_preds is not None
            assert(y_preds is not None)
            args = zip(X, y_probs, y_preds, [max_address_num for _ in range(len(X))])
            pool = Pool(processes=64)
            ret = pool.map(self.abduce_multilabel_wrapper, args)
            pool.close()
            pool.join()
            # for x, y_prob, y_pred in zip(X, y_probs, y_preds):
            #     ret.append(self.abduce_multilabel(x=x, y_prob=y_prob, y_pred=y_pred, max_address_num=max_address_num))
        return ret

if __name__ == "__main__":
    # # Multiclass
    # check = CheckRules("zoo/rule_format.txt")
    # abducer = Abducer(check, alpha = 1)
    # x_unlabel = [1,0,1,0,1,0,0,0,0,1,1,0,6,0,1,0]
    # y_unlabel_pred_proba = [0.49,0,0,0,0,0,0.51]
    # Y_unlabel_abduce = abducer.abduce_batch(task ='multiclass', X = [x_unlabel], y_probs = [y_unlabel_pred_proba], class_names=list(range(1,8)))
    # print(Y_unlabel_abduce)

    # # Multilabel
    # label = [0,1,1]
    # invert_labels = get_invert_label(label, [(0,1),(1,2)])
    # print(label, invert_labels)
    # cand_label_list = gen_cand_labels(label=label, max_address_num=2)
    # print(cand_label_list)

    # check = CheckRules("ade20k/rule_test.txt")
    check = CheckRules("ade20k/ade_rule.txt")
    abducer = Abducer(check, alpha = 1)

    # x_unlabel = [1,0,1,1]
    # y_unlabel_proba = [0.3,0.6,0.8]#,0.9]
    # y_pred = np.array(y_unlabel_proba)
    # y_pred[y_pred>0.5] = 1
    # y_pred[y_pred<=0.5] = 0
    # y_pred = y_pred.astype(int)
    # Y_unlabel_abduce = abducer.abduce_batch(task ='multilabel', X = [x_unlabel], y_probs = [y_unlabel_proba], y_preds = [y_pred], max_address_num = 3)
    # print(Y_unlabel_abduce)

    # Time test
    n_samples = 1800
    n_scenes, n_classes = 1055, 20
    scene_list = np.random.choice(n_scenes, size=n_samples)
    X = np.eye(n_scenes)[scene_list]
    y_probs = np.random.uniform(size=(n_samples,n_classes))
    y_preds = np.array(y_probs)
    y_preds[y_preds>0.5] = 1
    y_preds[y_preds<=0.5] = 0
    y_preds = y_preds.astype(int)

    time_start=time.time()
    # Eval unlabel
    # conf_list, vio_list = check.eval_rules(X, y_preds)
    # print(np.average(conf_list), conf_list.shape, np.average(vio_list), vio_list.shape)
    # Abduce
    Y_unlabel_abduce = abducer.abduce_batch(task ='multilabel', X = X, y_probs = y_probs, y_preds = y_preds, max_address_num = 3)

    time_end=time.time()
    print('total time cost',time_end-time_start,'s')
