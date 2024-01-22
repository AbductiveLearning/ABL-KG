import argparse
import os
import numpy as np

def arg_init():
    parser = argparse.ArgumentParser()
    # GPU
    parser.add_argument('--cuda_device', default=0, type=int, help='CUDA_VISIBLE_DEVICES')
    # GloVe vector file
    parser.add_argument('--GloVe_file', default="../word2vec.glove_vectors", type=str, help="path of GloVe file")
    # Data
    # parser.add_argument('--data_folder', default="data", type=str, help="path of data folder")
    parser.add_argument('--index_file', default="data/index_ade20k.pkl", type=str, help="ade20k index file name")
    parser.add_argument('--scene_file', default="data/sceneCategoriesADE2016.txt", type=str, help="ade20k scene file name")
    parser.add_argument('--n_training', default=20210, type=int, help='number of samples (labeled+unlabeled) for training')
    parser.add_argument('--n_label', default=1000, type=int, help='number of labeled samples for pre-training')
    # Rules
    parser.add_argument('--rules_folder', default="rules", type=str, help="path of rules folder")
    parser.add_argument('--handwritten_rule_filename', default="handwritten_rules.txt", type=str, help="handwritten rule file name")
    parser.add_argument('--KG_rule_filename', default="KG_rules.txt", type=str, help="KG rule file name")
    # Post-processing rules
    parser.add_argument('--resol_time', default=1, type=int, help='resolution times for rules')
    parser.add_argument('--resol_thres', default=0.5, type=float, help='resolution confidence threshold for rules')
    parser.add_argument('--filter_conf_thres', default=0.3, type=float, help='rule confidence threshold for rule filtering')
    # Rule match
    parser.add_argument('--rule_match_folder', default="rule_name_match", type=str, help="path of rule name match folder")
    parser.add_argument('--ML2IDX_dict_filename', default="ML2IDX_dict.json", type=str, help="machine learning names to index matching file name")
    # Abducer
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha of abducer')
    # Recorder
    parser.add_argument('--record_file', default="results/abl_result.txt", type=str, help='recorder file')
    parser.add_argument('--labeled_record_file', default="results/labeled_result.txt", type=str, help='labeled recorder file')
    
    args = parser.parse_args()
    args.obj_name_list = np.array(["bed", "cushion", "mountain, mount", "grass", "door", "mirror",   "sidewalk, pavement", "signboard, sign", "wall", "painting, picture", "car, auto, automobile, machine, motorcar", "person, individual, someone, somebody, mortal, soul"]) 
    # Input file
    # args.data_file = os.path.join(args.data_folder, args.data_filename)
    args.handwritten_rule_file = os.path.join(args.rules_folder, args.handwritten_rule_filename)
    args.KG_rule_file = os.path.join(args.rules_folder, args.KG_rule_filename)
    # Generated temp matching file
    args.KG2ML_attr_dict_file = os.path.join(args.rule_match_folder, "KG2MLattr_dict.json")
    args.KG2ML_class_dict_file = os.path.join(args.rule_match_folder, "KG2MLclass_dict.json")
    # Output rule file
    args.ML2IDX_dict_file = os.path.join(args.rule_match_folder, args.ML2IDX_dict_filename)
    args.forgetting_KGname_rule_file = os.path.join(args.rules_folder, "%drule_forgetting_KGname.txt")
    args.forgetting_MLname_rule_file = os.path.join(args.rules_folder, "%drule_forgetting_MLname.txt")
    args.forgetting_rule_file = os.path.join(args.rules_folder, "%drule_forgetting.txt")
    return args