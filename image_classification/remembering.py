from args import arg_init
import sys
from info import get_scene_file
import numpy as np
from word2vec import get_glove_vectors, matching, replace_rules, gen_ML2IDX_dict
from process_rules import get_rule_list, get_rule_names, process_scene_name, reserve, filter_rules
from checkrules import CheckRules


if __name__ == "__main__":
    args = arg_init()
    glove_vectors = get_glove_vectors(glove_file = args.GloVe_file)

    # Get KG names and ML names
    rules_folder = args.rules_folder
    handwritten_rule_file = args.handwritten_rule_file
    KG_rule_file = args.KG_rule_file
    handwritten_rule_list = get_rule_list(handwritten_rule_file)
    KG_rule_list = get_rule_list(KG_rule_file)
    all_rule_list = handwritten_rule_list + KG_rule_list
    # all_rule_list = handwritten_rule_list###########
    KG_names = get_rule_names(all_rule_list)
    print(len(handwritten_rule_list), "handwritten rules in", handwritten_rule_file)
    print(len(KG_rule_list), "KG rules in", KG_rule_file)
    print(len(KG_names), "entity names in total")

    _, scene_name_list = get_scene_file(args.scene_file)
    scene_name_list = list(np.unique(scene_name_list))
    obj_name_list = list(args.obj_name_list)
    scene_name_list = [process_scene_name(scene_name) for scene_name in scene_name_list]

    # Generating ML name matching file
    ML2IDX_dict_file = args.ML2IDX_dict_file
    gen_ML2IDX_dict(scene_name_list, obj_name_list, ML2IDX_dict_file)

    # Match each KG name
    print("Generating name matching file...")
    KG2ML_attr_dict_file = args.KG2ML_attr_dict_file
    KG2ML_class_dict_file = args.KG2ML_class_dict_file
    attr_matching_dict = matching(glove_vectors, KG_names = KG_names, ML_names = scene_name_list, thres = 0.7, outfile = KG2ML_attr_dict_file)
    class_matching_dict = matching(glove_vectors, KG_names = KG_names, ML_names = obj_name_list, thres = 0.8, outfile = KG2ML_class_dict_file)

    # Reserve the matched KG names
    resol_reserve_names = list(class_matching_dict.keys())
    # Remove person
    resol_reserve_names.remove('person, individual, someone, somebody, mortal, soul')
    rule_reserve_names = resol_reserve_names + list(attr_matching_dict.keys())
    print("Number of reserved names in resolution:", len(resol_reserve_names), "Number of reserved names in rules:", len(rule_reserve_names))

    # First filter the rules: if scene names appear, must on the left, class names on the right 
    all_rule_list = filter_rules(all_rule_list, list(attr_matching_dict.keys()), list(class_matching_dict.keys()))
    print("Number of rules after filtering:", len(all_rule_list))

    resol_time = args.resol_time
    forgetting_KGname_rule_file = args.forgetting_KGname_rule_file
    forgetting_MLname_rule_file = args.forgetting_MLname_rule_file

    ret_rule_list = reserve(all_rule_list, resol_reserve_names, rule_reserve_names, resol_time = resol_time, resol_thres = args.resol_thres, out_rule_filename = forgetting_KGname_rule_file)
    forgetting_rule_file = args.forgetting_rule_file

    for t in range(1, resol_time+1):
        replace_rules(forgetting_KGname_rule_file%t, KG2ML_attr_dict_file, forgetting_MLname_rule_file%t)
        replace_rules(forgetting_MLname_rule_file%t, KG2ML_class_dict_file, forgetting_MLname_rule_file%t)
        replace_rules(forgetting_MLname_rule_file%t, ML2IDX_dict_file, forgetting_rule_file%t)

