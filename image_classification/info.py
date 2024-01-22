import pickle as pkl
import numpy as np
import sys

def get_unique_idx(name, name_list):
    unique_name_list = np.unique(name_list)
    idxs = np.where(unique_name_list==name)[0]
    assert(len(idxs)>0)
    return idxs[0]

def get_unique_idx_list(name_list, unique_name_list):
    out_list = np.array(name_list.copy())
    for i, name in enumerate(unique_name_list):
        out_list[out_list==name] = i
    return out_list.astype(int)

def get_one_hot(name_list, unique_name_list):
    class_list = get_unique_idx_list(name_list, unique_name_list)
    one_hot_list = np.eye(len(unique_name_list))[class_list]
    return one_hot_list

def mine_scene_object_rule(obj_name_list, scene_list, fig_object_list, confidence_thres = 0.9, left_hold_thres = 1):
    assert(fig_object_list.shape[0]==len(scene_list))
    assert(fig_object_list.shape[1]==len(obj_name_list))
    rule_list = []
    for scene_idx, scene in enumerate(np.unique(scene_list)):
        scene_idxs = np.where(scene_list==scene)
        if len(scene_idxs[0]) < left_hold_thres:
            continue
        avg_object_cnt = np.mean(fig_object_list[scene_idxs], axis=0)
        rule_obj_idxs = np.where(avg_object_cnt >= confidence_thres)[0]
        # rule_obj_idxs = np.where(avg_object_cnt < confidence_thres)[0] ################################################
        for obj_idx in rule_obj_idxs:
            obj_name = obj_name_list[obj_idx]
            # rule_name_list.append("%s -> %s"%(scene, obj_name))
            rule_list.append((scene_idx, obj_idx, avg_object_cnt[obj_idx]))
    return rule_list

def mine_object_scene_rule(obj_name_list, scene_list, fig_object_list, confidence_thres = 0.9):
    assert(fig_object_list.shape[0]==len(scene_list))
    assert(fig_object_list.shape[1]==len(obj_name_list))
    rule_list = []
    for obj_idx in range(fig_object_list.shape[1]):
        obj_idxs = np.where(fig_object_list[:,obj_idx]==1)[0] # is 1
        appear_scene_list = scene_list[obj_idxs]
        for scene in np.unique(appear_scene_list):
            conf = np.sum(appear_scene_list==scene)/len(appear_scene_list)
            if conf > confidence_thres:
                obj_name = obj_name_list[obj_idx]
                scene_idx = get_unique_idx(scene, scene_list)
                # rule_name_list.append("%s -> %s"%(obj_name, scene))
                rule_list.append((obj_idx, scene_idx, conf))
                break
    return rule_list

def mine_object_object_rule(obj_name_list, fig_object_list, confidence_thres = 0.9):
    rule_list = []
    for obj1_idx in range(fig_object_list.shape[1]):
        left_cnt = np.sum(fig_object_list[:,obj1_idx]==1) # is 1
        for obj2_idx in range(fig_object_list.shape[1]):
            if obj1_idx == obj2_idx:
                continue
            all_cnt = np.sum((fig_object_list[:,obj1_idx]==1)&(fig_object_list[:,obj2_idx]==1)) # is 1
            conf = all_cnt/left_cnt
            if conf > confidence_thres:
                obj_name1 = obj_name_list[obj1_idx]
                obj_name2 = obj_name_list[obj2_idx]
                # rule_name_list.append("%s -> %s"%(obj_name1, obj_name2))
                rule_list.append((obj1_idx, obj2_idx, conf))
    return rule_list


def get_cand_obj_idxs(fig_object_list, obj_cnt_thres = 1000):
    obj_cnt = np.sum(fig_object_list, axis=0)
    cand_obj_idxs = np.where(obj_cnt>obj_cnt_thres)[0]
    return cand_obj_idxs

def get_scene_file(filename, shuffle_train = False, n_training = None):
    file_name_list, scene_list = [], []
    with open(filename, "r") as f:
        for line in f.readlines():
            info = line.strip().split(" ")
            file_name_list.append(info[0])
            scene_list.append(info[1])
    file_name_list, scene_list = np.array(file_name_list), np.array(scene_list)
    if shuffle_train: # Shuffle the first n_training examples
        if n_training is None:
            n_training = len(file_name_list)
        p = np.random.permutation(n_training)
        rest = np.arange(n_training, len(file_name_list))
        p = np.concatenate((p, rest))
        file_name_list, scene_list = file_name_list[p], scene_list[p]
    return file_name_list, scene_list

def get_fig_object_list(index_ade20k, file_name_list):
    filename2idx = dict(zip(index_ade20k['filename'], list(range(0, len(index_ade20k['filename'])))))
    idx_list = [filename2idx[file_name+".jpg"] for file_name in file_name_list]
    fig_object_list = index_ade20k['objectPresence'][:, idx_list].T
    fig_object_list[fig_object_list>1] = 1
    return fig_object_list

def gen_rule_file(scene_obj_rule_list = [], obj_scene_rule_list = [], obj_obj_rule_list = [], obj_name_list = None, scene_list = None, rule_file="", name_rule_file=""):
    # Total index: scene index + object index
    unique_scene_list = np.unique(scene_list)
    assert(len(np.unique(obj_name_list))==len(obj_name_list))
    rule_str_list, name_rule_str_list = [], []
    for rule in scene_obj_rule_list:
        scene_idx, obj_idx, conf = rule
        # print(scene_idx, obj_idx, conf, unique_scene_list[scene_idx], obj_name_list[obj_idx])
        rule_str_list.append("([(1, %d, 1)], [(1, %d, 1)], %f)"%(scene_idx, len(unique_scene_list)+obj_idx, conf)) 
        # name_rule_str_list.append("([(1, '%s', 1)], [(1, '%s', 1)], %f)"%(unique_scene_list[scene_idx], obj_name_list[obj_idx], conf)) 
        name_rule_str_list.append("([(1, '%s', 1)], [(0, '%s', 1)], %f)"%(unique_scene_list[scene_idx], obj_name_list[obj_idx], 1-conf)) #################
    for rule in obj_scene_rule_list:
        obj_idx, scene_idx, conf = rule
        # print(obj_idx, scene_idx, conf, obj_name_list[obj_idx], unique_scene_list[scene_idx])
        rule_str_list.append("([(1, %d, 1)], [(1, %d, 1)], %f)"%(len(unique_scene_list)+obj_idx, scene_idx, conf)) 
        name_rule_str_list.append("([(1, '%s', 1)], [(1, '%s', 1)], %f)"%(obj_name_list[obj_idx], unique_scene_list[scene_idx], conf))
    for rule in obj_obj_rule_list:
        obj1_idx, obj2_idx, conf = rule
        # print(obj1_idx, obj2_idx, conf, obj_name_list[obj1_idx], obj_name_list[obj2_idx])
        rule_str_list.append("([(1, %d, 1)], [(1, %d, 1)], %f)"%(len(unique_scene_list)+obj1_idx, len(unique_scene_list)+obj2_idx, conf)) 
        name_rule_str_list.append("([(1, '%s', 1)], [(1, '%s', 1)], %f)"%(obj_name_list[obj1_idx], obj_name_list[obj2_idx], conf))
    if len(rule_file) > 0:
        with open(rule_file, "w") as f:
            f.write("\n".join(rule_str_list))
    if len(name_rule_file) > 0:
        with open(name_rule_file, "w") as f:
            f.write("\n".join(name_rule_str_list))

# Filter by indexs
def filter_rule_file(rule_file, name_rule_file, idxs):
    for in_file in [rule_file, name_rule_file]:
        in_rule_list = []
        with open(in_file, "r") as fin:
            for rule in fin:
                in_rule_list.append(rule)
        in_rule_list = np.array(in_rule_list)
        out_rule_list = in_rule_list[idxs]
        out_file = "filter_" + in_file
        with open(out_file, "w") as fout:
            for rule in out_rule_list:
                fout.write(rule)

# Select top k violated rules for each scene
def filter_class_rule(rule_list, vio_list, class_n_rules=10):
    assert(len(rule_list)==len(vio_list))
    ret_idx_list = []
    class_list = np.array([rule[1][0][1] for rule in rule_list])
    class_list, vio_list, sorted_idxs = zip(*sorted(zip(class_list, vio_list, np.arange(len(vio_list))), reverse=True))
    print(class_list, vio_list, sorted_idxs)
    class_list, vio_list, sorted_idxs = np.array(class_list), np.array(vio_list), np.array(sorted_idxs)
    for c in np.unique(class_list):
        class_idxs = sorted_idxs[class_list==c]
        ret_idx_list.extend(class_idxs[:class_n_rules])
    return ret_idx_list





def analyze_bad_cases(unlabel_gt, unlabel_pred, scene_list, obj_name_list):
    fn_cnt_list = []
    conf_list = []
    can_abduce_obj_cnt_list = np.zeros(len(obj_name_list))
    unique_scene_list = np.unique(scene_list)
    assert(len(unlabel_gt)==len(unlabel_pred)==len(scene_list))
    assert(unlabel_gt.shape[1]==unlabel_pred.shape[1]==len(obj_name_list))
    threshold = 3
    fn = (unlabel_gt==1)&(unlabel_pred==0)
    tp = (unlabel_gt==1)&(unlabel_pred==1)
    tn = (unlabel_gt==0)&(unlabel_pred==0)
    fn_total = fn.sum(axis=0)
    tp_total = tp.sum(axis=0)
    tn_total = tn.sum(axis=0)
    # Original f1
    ori_f1_list = tp_total*2/(tp_total-tn_total+len(unlabel_gt))
    for scene in unique_scene_list:
        fn_cnt = fn[scene_list==scene].sum(axis=0)
        fn_cnt_list.append(fn_cnt)
        conf = unlabel_gt[scene_list==scene].mean(axis=0)
        conf_list.append(conf)
    fn_cnt_list = np.array(fn_cnt_list)
    conf_list = np.array(conf_list)
    idxs = np.where(fn_cnt_list>threshold)
    for idx0, idx1 in zip(idxs[0], idxs[1]):
        # print(unique_scene_list[idx0], obj_name_list[idx1], fn_cnt_list[idx0][idx1], conf_list[idx0, idx1])
        if conf_list[idx0,idx1] > 0.6:
            # can_abduce_obj_cnt_list[idx1] += fn_cnt_list[idx0][idx1]
            tp_total[idx1] += fn_cnt_list[idx0][idx1]
    # np.savetxt("fn_cnt.txt", fn_cnt_list, fmt="%.1f", delimiter=",", header="','".join(obj_name_list))
    # Percentage of correct abduced labels in false negative
    # can_abduce_perc = np.array(can_abduce_obj_cnt_list)/np.array(fn_total)
    # print(can_abduce_perc*100)
    # New f1
    new_f1_list = tp_total*2/(tp_total-tn_total+len(unlabel_gt))
    inc = new_f1_list - ori_f1_list
    print(ori_f1_list.mean(), new_f1_list.mean())
    print(ori_f1_list, "\n", new_f1_list, "\n", inc)
    print("top 15 avg inr:", np.average(np.sort(inc, )[-15:]))


if __name__ == "__main__":
    pass