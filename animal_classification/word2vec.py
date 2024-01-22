import gensim
import gensim.downloader
import json
import os
from process_rules import get_rule_list, write_rules

def gen_ML2IDX_dict(attr_names, class_names, ML2IDX_dict_file):
    all_names = attr_names + class_names
    matching_dict = dict(zip(all_names,list(range(len(all_names)))))
    # Write a json file
    info_json = json.dumps(matching_dict, indent=4)
    with open(ML2IDX_dict_file, 'w') as f:
        print("Generating matching file:", ML2IDX_dict_file)
        f.write(info_json)

# Remove words that do not have vectors
def validate_words(glove_vectors, names):
    for name in names:
        try:  
            vec = glove_vectors[name]
        except:
            names.remove(name)
            # print("Remove word:", name)
    return names

# Output a word matching json
def matching(glove_vectors, KG_names, ML_names, thres = 0.5, outfile = ""):
    new_KG_names = validate_words(glove_vectors, KG_names.copy())
    new_ML_names = validate_words(glove_vectors, ML_names.copy())
    # For every name in KG, find the most similar name larger than thres in new_ML_names
    matching_dict = {}
    for KG_name in new_KG_names:
        try:
            most_similar_ML_name = glove_vectors.most_similar_to_given(KG_name, new_ML_names)
            similarity = glove_vectors.similarity(KG_name, most_similar_ML_name)
            if similarity >= thres:
                matching_dict[KG_name] = most_similar_ML_name# +" "+str(similarity)
        except:
            pass
    same_set = set(KG_names) & set(ML_names)
    for name in same_set:
        matching_dict[name] = name
    # Write a json file
    if len(outfile) > 0:
        info_json = json.dumps(matching_dict, indent=4)
        with open(outfile, 'w') as f:
            print("Generating matching file:", outfile)
            f.write(info_json)
    return matching_dict

# Replace the words in the rule file by matching json
def replace_rules(rule_file, KG2ML_dict_file, outfile):
    # Ensure: words are in the form of 'word'
    with open(KG2ML_dict_file, "r") as f:
        matching_dict = json.load(f)
    rule_list = get_rule_list(rule_file)
    new_rule_list = []
    for rule in rule_list:
        left, right, conf = rule[0], rule[1], rule[2]
        new_left, new_right = [], []
        for literal in left:
            replaced = matching_dict.get(literal[1])
            if replaced is not None:
                new_left.append((literal[0], replaced, literal[2]))
            else:
                new_left.append((literal[0], literal[1], literal[2]))
        for literal in right:
            replaced = matching_dict.get(literal[1])
            if replaced is not None:
                new_right.append((literal[0], replaced, literal[2]))
            else:
                new_right.append((literal[0], literal[1], literal[2]))
        new_rule_list.append((new_left, new_right, conf))
    write_rules(new_rule_list, outfile)
    return new_rule_list
    # with open(rule_file, "r") as f:
    #     rule_str = f.read()
    #     for key in matching_dict:
    #         if type(matching_dict[key]) == int:
    #             rule_str = rule_str.replace("'"+str(key)+"'", str(matching_dict[key]))
    #         else: # str
    #             rule_str = rule_str.replace("'"+str(key)+"'", "'"+matching_dict[key]+"'")
    # with open(outfile, 'w') as f:
    #     f.write(rule_str)
    
    
def get_glove_vectors(glove_name = 'glove-wiki-gigaword-300', glove_file = "word2vec.glove_vectors"):
    if not os.path.exists(glove_file):
        print(list(gensim.downloader.info()['models'].keys()))
        glove_vectors = gensim.downloader.load(glove_name)
        glove_vectors.save(glove_file)

    # Load back with memory-mapping = read-only, shared across processes.
    # https://radimrehurek.com/gensim/models/keyedvectors.html
    glove_vectors = gensim.models.KeyedVectors.load(glove_file, mmap='r')
    return glove_vectors

if __name__ == "__main__":
    glove_vectors = get_glove_vectors()

    print(glove_vectors.similarity('fly','airborne'))
    print(glove_vectors.similarity('tooth','toothed'))
    print(glove_vectors.similarity('breath','breathes'))
    # print(glove_vectors.wmdistance('fish','fishes'))
    KG2ML_dict_file = "KG2ML_dict.json"
    ML2IDX_dict_file = "ML2IDX_dict.json"
    matching_dict = matching(glove_vectors, 
    ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "mammal", "bird", "reptile", "fish", "amphibian", "insect", "mollusk", "cat-sized"], 
    # ["hair", "feather", "egg", "milk", "fly", "aquatic", "predator", "tooth", "backbone", "breath", "venomous", "fin", "leg", "tail", "domestic", "mammal", "bird", "reptile", "fish", "amphibian", "insect", "mollusk", "cat-sized"], 
    ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "mammal", "bird", "reptile", "fish", "amphibian", "insect", "mollusk", "cat-sized"], outfile = KG2ML_dict_file)

    replace_rules("rule_name.txt", KG2ML_dict_file, "rule_replaced.txt")
    replace_rules("rule_replaced.txt", ML2IDX_dict_file, "rule_replaced2.txt")
