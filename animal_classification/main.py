# import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import sys 
import os
import json
from args import arg_init
from checkrules import CheckRules
from abducer import Abducer

if __name__ == "__main__":
    args = arg_init()
    # Load data
    data = np.loadtxt(args.data_file, delimiter=',', dtype=str)
    data = data[:, 1:].astype('int')
    data_x = data[:, :-1]
    data_y = data[:, -1]
    n_classes = 7

    # Load rules
    rules_folder = args.rules_folder
    check = CheckRules(args.forgetting_rule_file%(1))
    # Validate rules
    if False:
        for i in range(len(data_y)):
            x, y = data_x[i], data_y[i]
            y_onehot = np.eye(n_classes)[y-1]
            print(x,y)
            print(check.judge(x, y_onehot))
    # exit()

    test_size = 0.3
    # Split data (one shot)
    # For every class: 1 : (1-test_size)*(len-1) : test_size*(len-1)
    label_index, unlabel_index, test_index = [], [], []
    for i in np.unique(data_y):
        idxs_all = np.arange(len(data_y))
        idxs = idxs_all[data_y==i]
        np.random.shuffle(idxs)
        label_index.append(idxs[0]) # one shot
        split_idx = int(1 + (1-test_size)*(len(idxs)-1))
        unlabel_index.extend(idxs[1:split_idx])
        test_index.extend(idxs[split_idx:])
        # print(idxs)
        # print(idxs[1:split_idx])
        # print(idxs[split_idx:])
        # print()
    # print(label_index, unlabel_index, test_index)
    
    X_label, X_unlabel, X_test = data_x[label_index], data_x[unlabel_index], data_x[test_index]
    Y_label, Y_unlabel, Y_test = data_y[label_index], data_y[unlabel_index], data_y[test_index]

    # Adjust KB rules and confidence
    conf_threshold = args.filter_conf_thres
    conf_list, vio_list = check.eval_rules(X_label, np.eye(n_classes)[Y_label-1])
    org_conf_list = check.get_rule_conf()
    avg_conf_list = np.nanmean(np.array((conf_list, org_conf_list)), axis=0)
    select_rule_idxs = np.where((avg_conf_list>conf_threshold) | np.isnan(conf_list))[0] # if no data, then use default confidence
    check.update_rule_by_idxs(select_rule_idxs, avg_conf_list[select_rule_idxs])
    conf_list, vio_list = check.eval_rules(X_unlabel, np.eye(n_classes)[Y_unlabel-1])
    print(len(select_rule_idxs), select_rule_idxs+1)
    print(conf_list)

    rf = RandomForestClassifier()

    # Pre train
    print("---- Pre training ----")
    rf = RandomForestClassifier()
    rf.fit(X_label, Y_label)
    # Test
    test_pred_list = rf.predict(X_test)
    labeled_test_acc = accuracy_score(Y_test, test_pred_list)
    print(labeled_test_acc)

    # Predict unlabeled data
    Y_unlabel_pred = rf.predict(X_unlabel).copy()
    Y_unlabel_pred_proba = rf.predict_proba(X_unlabel).copy()
    print("\nUnlabel acc:", accuracy_score(Y_unlabel, Y_unlabel_pred))
    
    # Abductive learning
    print("---- Abductive learning ----")
    abducer = Abducer(check, alpha = args.alpha)
    Y_unlabel_abduce = abducer.abduce_batch(task='multiclass', X=X_unlabel, y_probs=Y_unlabel_pred_proba, class_names=list(range(1,n_classes+1)))
    print("Abduced acc:", accuracy_score(Y_unlabel, Y_unlabel_abduce))
    rf = RandomForestClassifier()
    rf.fit(np.concatenate((X_label,X_unlabel)), np.concatenate((Y_label,Y_unlabel_abduce)))
    # Analyze abduction
    if False:
        with open(args.ML2IDX_dict_file, "r") as f:
            ml2idx_dict = json.load(f)
            idx2ml_dict = {value:key for key,value in ml2idx_dict.items()}
        for (x, y, pred, prob, abduce) in zip(X_unlabel, Y_unlabel, Y_unlabel_pred, Y_unlabel_pred_proba, Y_unlabel_abduce):
            if y != abduce: # wrong abduction
                print(x, "y:", y, prob, "pred:", pred, "abduce:", abduce)
                y_onehot = np.eye(n_classes)[y-1]
                print("y consistency: ", check.judge([x], [y_onehot], log=True, idx2ml_dict=idx2ml_dict))
                pred_onehot = np.eye(n_classes)[pred-1]
                print("pred consistency: ", check.judge([x], [pred_onehot]))
                abduce_onehot = np.eye(n_classes)[abduce-1]
                print("abduce consistency: ", check.judge([x], [abduce_onehot]))
    # Test
    test_pred_list = rf.predict(X_test)
    abl_test_acc = accuracy_score(Y_test, test_pred_list)
    print(abl_test_acc)
    exit()

    # Abductive learning with Oracle
    print("---- Abductive learning with Oracle ----")
    check = CheckRules(os.path.join(rules_folder, "rule_oracle.txt"))
    abducer = Abducer(check, alpha = args.alpha)
    Y_unlabel_abduce = abducer.abduce_batch(task='multiclass', X=X_unlabel, y_probs=Y_unlabel_pred_proba, class_names=list(range(1,n_classes+1)))
    print("Abduced acc:", accuracy_score(Y_unlabel, Y_unlabel_abduce))
    rf = RandomForestClassifier()
    rf.fit(np.concatenate((X_label,X_unlabel)), np.concatenate((Y_label,Y_unlabel_abduce)))
    # Test
    test_pred_list = rf.predict(X_test)
    abl_oracle_test_acc = accuracy_score(Y_test, test_pred_list)
    print(abl_oracle_test_acc)
    abl_oracle_recorder = ResultRecorder("results/abl_oracle_result.txt")
    abl_oracle_recorder.zoo_result2file(abl_oracle_test_acc)
