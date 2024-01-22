from args import arg_init
import sys
from word2vec import get_glove_vectors, matching, replace_rules, gen_ML2IDX_dict
from process_rules import get_rule_list, get_rule_names, reserve
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
    KG_names = get_rule_names(all_rule_list)
    print(len(handwritten_rule_list), "handwritten rules in", handwritten_rule_file)
    print(len(KG_rule_list), "KG rules in", KG_rule_file)
    print(len(KG_names), "entity names in total")

    # Generating ML name matching file
    attr_names = args.attr_names
    class_names = args.class_names
    ML2IDX_dict_file = args.ML2IDX_dict_file
    gen_ML2IDX_dict(attr_names, class_names, ML2IDX_dict_file)

    # Match each KG name
    KG2ML_attr_dict_file = args.KG2ML_attr_dict_file
    KG2ML_class_dict_file = args.KG2ML_class_dict_file
    attr_matching_dict = matching(glove_vectors, KG_names = KG_names, ML_names = attr_names, thres = 0.7, outfile = KG2ML_attr_dict_file)
    class_matching_dict = matching(glove_vectors, KG_names = KG_names, ML_names = class_names, thres = 0.8, outfile = KG2ML_class_dict_file)

    # (Remove rules that have a low confidence score in data)
    # TODO

    # Forget the KG names not matched
    resol_reserve_names = list(class_matching_dict.keys())
    rule_reserve_names = resol_reserve_names + list(attr_matching_dict.keys())
    print(len(KG_names), len(resol_reserve_names), len(rule_reserve_names))

    resol_time = args.resol_time
    forgetting_KGname_rule_file = args.forgetting_KGname_rule_file
    forgetting_MLname_rule_file = args.forgetting_MLname_rule_file

    ret_rule_list = reserve(all_rule_list, resol_reserve_names, rule_reserve_names, resol_time = resol_time, resol_thres = args.resol_thres, out_rule_filename = forgetting_KGname_rule_file)
    forgetting_rule_file = args.forgetting_rule_file

    for t in range(1, resol_time+1):
        replace_rules(forgetting_KGname_rule_file%t, KG2ML_attr_dict_file, forgetting_MLname_rule_file%t)
        replace_rules(forgetting_MLname_rule_file%t, KG2ML_class_dict_file, forgetting_MLname_rule_file%t)
        replace_rules(forgetting_MLname_rule_file%t, args.ML2IDX_dict_file, forgetting_rule_file%t)

