import argparse
import os

def arg_init():
    parser = argparse.ArgumentParser()
    # GloVe vector file
    parser.add_argument('--GloVe_file', default="../word2vec.glove_vectors", type=str, help="path of GloVe file")
    # Data
    parser.add_argument('--data_folder', default="data", type=str, help="path of data folder")
    parser.add_argument('--data_filename', default="zoo.data", type=str, help="data file name")
    # Rules
    parser.add_argument('--rules_folder', default="rules", type=str, help="path of rules folder")
    parser.add_argument('--handwritten_rule_filename', default="handwritten_rules.txt", type=str, help="handwritten rule file name")
    parser.add_argument('--KG_rule_filename', default="KG_rules.txt", type=str, help="KG rule file name")
    # Post-processing rules
    parser.add_argument('--resol_time', default=1, type=int, help='resolution times for rules')
    parser.add_argument('--resol_thres', default=0.5, type=float, help='resolution confidence threshold for rules')
    parser.add_argument('--filter_conf_thres', default=0.55, type=float, help='rule confidence threshold for rule filtering')
    # Rule match
    parser.add_argument('--rule_match_folder', default="rule_name_match", type=str, help="path of rule name match folder")
    parser.add_argument('--ML2IDX_dict_filename', default="ML2IDX_dict.json", type=str, help="machine learning names to index matching file name")
    # Abducer
    parser.add_argument('--alpha', default=1.0, type=float, help='alpha of abducer')
    # Recorder
    parser.add_argument('--record_file', default="results/abl_result.txt", type=str, help='recorder file')
    
    args = parser.parse_args()
    args.attr_names = ["hair", "feather", "egg", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fin", "legs", "tail", "domestic", "catsize"]
    args.class_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "mollusk"]
    # Input file
    args.data_file = os.path.join(args.data_folder, args.data_filename)
    args.handwritten_rule_file = os.path.join(args.rules_folder, args.handwritten_rule_filename)
    args.KG_rule_file = os.path.join(args.rules_folder, args.KG_rule_filename)
    args.ML2IDX_dict_file = os.path.join(args.rule_match_folder, args.ML2IDX_dict_filename)
    # Generated temp matching file
    args.KG2ML_attr_dict_file = os.path.join(args.rule_match_folder, "KG2MLattr_dict.json")
    args.KG2ML_class_dict_file = os.path.join(args.rule_match_folder, "KG2MLclass_dict.json")
    # Output rule file
    args.forgetting_KGname_rule_file = os.path.join(args.rules_folder, "%drule_forgetting_KGname.txt")
    args.forgetting_MLname_rule_file = os.path.join(args.rules_folder, "%drule_forgetting_MLname.txt")
    args.forgetting_rule_file = os.path.join(args.rules_folder, "%drule_forgetting.txt")
    return args