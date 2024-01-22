import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import time
import pickle as pkl
import os
from info import get_scene_file, get_fig_object_list, get_cand_obj_idxs
from metrics import compute_f1
from NN_model import TorchDataset, ResNet50
from info import get_one_hot, analyze_bad_cases, get_unique_idx_list, filter_rule_file, filter_class_rule
import sys
from args import arg_init
sys.path.append("..")
from checkrules import CheckRules
from abducer import Abducer

if __name__ == '__main__':
    args = arg_init()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    print(args)

    n_label =  args.n_label# 20210
    n_training = args.n_training
    crop_size = 320
    batch_size, pretrain_epochs, epochs = 128, 6, 10
    rule_file = args.forgetting_rule_file%(1)
    name_rule_file = args.forgetting_MLname_rule_file%(1)
    index_file = args.index_file
    check = CheckRules(rule_file, name_rule_file)
    abducer = Abducer(check, alpha = args.alpha)

    with open(index_file, 'rb') as f:
        index_ade20k = pkl.load(f)
    # Scene info from ADE2016, and object info from 2021
    file_name_list, scene_list = get_scene_file(args.scene_file, shuffle_train=True, n_training=n_training)
    fig_object_list = get_fig_object_list(index_ade20k, file_name_list)
    obj_name_list = args.obj_name_list
    cand_obj_idxs = get_unique_idx_list(obj_name_list, index_ade20k['objectnames'])
    fig_object_list = fig_object_list[:,cand_obj_idxs]
    label_gt, unlabel_gt, test_gt = fig_object_list[:n_label], fig_object_list[n_label:n_training], fig_object_list[n_training:]
    label_scene, unlabel_scene = scene_list[:n_label], scene_list[n_label:n_training]
    unique_scene_list = np.unique(scene_list)
    label_scene_one_hot = get_one_hot(label_scene, unique_scene_list)
    unlabel_scene_one_hot = get_one_hot(unlabel_scene, unique_scene_list)
    num_class = fig_object_list.shape[1]
    print("num_class:", num_class)

    # # Adjust KB rules and confidence
    # conf_threshold = args.filter_conf_thres
    # conf_list, vio_list = check.eval_rules(label_scene_one_hot, label_gt)
    # org_conf_list = check.get_rule_conf()
    # avg_conf_list = np.nanmean(np.array((conf_list, org_conf_list)), axis=0)
    # select_rule_idxs = np.where((avg_conf_list>conf_threshold) | np.isnan(conf_list))[0] # if no data, then use default confidence
    # # select_rule_idxs = np.where((avg_conf_list>conf_threshold))[0] ##########
    # # select_rule_idxs = np.where(np.isnan(conf_list))[0] ########
    # check.update_rule_by_idxs(select_rule_idxs, avg_conf_list[select_rule_idxs])
    # conf_list, vio_list = check.eval_rules(unlabel_scene_one_hot, unlabel_gt)
    # print(select_rule_idxs+1)
    # print(conf_list, np.nanmean(conf_list))
    # # exit()
    
    label_data = TorchDataset(root_dataset="images/training", image_name_list=file_name_list[:n_label], training=True, label=label_gt, scale_size=640, crop_size=crop_size)
    label_loader = DataLoader(label_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    unlabel_data = TorchDataset(root_dataset="images/training", image_name_list=file_name_list[n_label:n_training], training=False, label=unlabel_gt, scale_size=crop_size, crop_size=crop_size)
    unlabel_loader = DataLoader(unlabel_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_data = TorchDataset(root_dataset="images/validation", image_name_list=file_name_list[n_training:], training=False, label=test_gt, scale_size=crop_size, crop_size=crop_size)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = ResNet50(num_class=num_class, loss_criterion=nn.MultiLabelSoftMarginLoss(), freeze_feature=False).cuda() #BCEWithLogitsLoss()
    print("Pre-training...")
    for epoch in range(1, pretrain_epochs + 1):
        model.train_val(1, True, data_loader=label_loader)
        ori_test_loss, ori_test_micro_f1, ori_test_macro_f1, ori_test_f1_score_list, ori_test_precision_list, ori_test_recall_list = model.train_val(1, False, data_loader=test_loader)
    # model.set_freeze_feature(False, lr=5e-5)
    model.set_freeze_feature(False, lr=2e-5)
    test_micro_f1_record_list, test_macro_f1_record_list = [], []
    test_micro_f1_record_list.append(ori_test_micro_f1)
    test_macro_f1_record_list.append(ori_test_macro_f1)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Predict unlabel
        unlabel_pred_prob = model.predict(unlabel_loader)
        unlabel_pred_class = unlabel_pred_prob.copy()
        unlabel_pred_class[unlabel_pred_class>0.5] = 1
        unlabel_pred_class[unlabel_pred_class<=0.5] = 0

        # Unlabel f1
        micro_f1_score, macro_f1_score, unlabel_f1_score_list, unlabel_precision_list, unlabel_recall_list = compute_f1(unlabel_gt, unlabel_pred_class)
        print("### Unlabel f1", micro_f1_score, macro_f1_score, unlabel_f1_score_list)

        # Abduce label
        print("Abducing labels")
        Y_unlabel_abduce = abducer.abduce_batch(task ='multilabel', X = unlabel_scene_one_hot, y_probs = unlabel_pred_prob, y_preds = unlabel_pred_class, max_address_num = 2)
        micro_f1_score, macro_f1_score, abduce_f1_score_list, abduce_precision_list, abduce_recall_list = compute_f1(unlabel_gt, Y_unlabel_abduce)
        
        # Retrain model
        label_abduce_data = TorchDataset(root_dataset="images/training", image_name_list=file_name_list[:n_training], training=False, label=np.concatenate((fig_object_list[:n_label],Y_unlabel_abduce)), scale_size=640, crop_size=crop_size)
        label_abduce_loader = DataLoader(label_abduce_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        model.train_val(1, True, data_loader=label_abduce_loader)
        test_loss, test_micro_f1, test_macro_f1, test_f1_score_list, test_precision_list, test_recall_list = model.train_val(1, False, data_loader=test_loader)
        
        print("### Test f1", test_micro_f1, test_macro_f1, test_f1_score_list)
        print(obj_name_list)
        test_f1_inc = test_f1_score_list-ori_test_f1_score_list
        idxs = np.argsort(test_f1_inc)
        print('### Increment ', np.mean(test_f1_inc), test_f1_inc[idxs])
        print(obj_name_list[idxs])
        print("### Test precision", test_precision_list)
        test_prec_inc = test_precision_list-ori_test_precision_list
        idxs = np.argsort(test_prec_inc)
        print('### Increment ', np.mean(test_prec_inc), test_prec_inc[idxs])
        print(obj_name_list[idxs])
        print("### Test recall", test_recall_list)
        test_prec_inc = test_recall_list-ori_test_recall_list
        idxs = np.argsort(test_prec_inc)
        print('### Increment ', np.mean(test_prec_inc), test_prec_inc[idxs])
        print(obj_name_list[idxs])
        print("\n")

        test_micro_f1_record_list.append(test_micro_f1)
        test_macro_f1_record_list.append(test_macro_f1)