from rule import FuncRule, Rule
from collections import defaultdict
from munkres import Munkres
import os 
import numpy as np
from jpype.types import *
    

def str2bool(v):
    # because argparse does not support to parse "true, False" as python
    # boolean directly
    return v.lower() in ("true", "t", "1", "True")


def merge_new_alignment(global_new_alignment, new_alignment):
    rest = set(new_alignment)
    new_source_entities = set([i for i, j in new_alignment])
    new_target_entities = set([j for i, j in new_alignment])
    for i, j in global_new_alignment:
        if i not in new_source_entities and j not in new_target_entities:
            rest.add((i, j))
    print("# new alignment after merging:", len(rest) - len(new_alignment))
    return rest


def obtain_1_to_1_pairs(not_1_to_1_pairs, sim_mat):
    assert sim_mat.shape[0] == sim_mat.shape[1]
    m = Munkres()
    matrix = np.zeros((sim_mat.shape[0], sim_mat.shape[1])) + 100
    for e1_idx, e2_idx in not_1_to_1_pairs:
        matrix[e1_idx][e2_idx] = -sim_mat[e1_idx][e2_idx]
    indexes = m.compute(matrix)
    return indexes

    
def check_left_e1s(kgs, pairs, func_rules, mined_rules):
    # all_e1_list = [e1 for e1, _ in kgs.test_links + kgs.valid_links]
    predict_e1 = [e1 for e1, _ in pairs] 
    predict_e2 = [e2 for _, e2 in pairs]
    
    left_e1 = []
    for e1, e2 in kgs.test_links + kgs.valid_links:
        if e1 not in set(predict_e1) and e2 not in set(predict_e2):
            left_e1.append(e1)
    # left_e1 = list(set(all_e1_list) - set(predict_e1))
    
    related_e1 = 0
    my_kg1_h_to_r_t = {} # key: h, value: {r: t}
    my_kg1_t_to_r_h = {} # key: t, value: {r: h}
    for h, r, t in kgs.kg1.relation_triples_set:
        if h not in my_kg1_h_to_r_t.keys():
            my_kg1_h_to_r_t[h] = {}
        if r not in my_kg1_h_to_r_t[h].keys():
            my_kg1_h_to_r_t[h][r] = set()
        my_kg1_h_to_r_t[h][r].add(t)
        
        if t not in my_kg1_t_to_r_h.keys():
            my_kg1_t_to_r_h[t] = {}
        if r not in my_kg1_t_to_r_h[t].keys():
            my_kg1_t_to_r_h[t][r] = set()
        my_kg1_t_to_r_h[t][r].add(h)
        
    for e1 in left_e1:
        flag = False
        for rule in func_rules + mined_rules:
            if flag:
                break
            for body in rule.rule_body_left:
                if flag:
                    break
                if body[1] in my_kg1_h_to_r_t.get(e1, {}).keys() or body[1] in my_kg1_t_to_r_h.get(e1, {}).keys():
                    flag = True 
                    break
        if flag:
            related_e1 += 1
    print("among {} left e1s, there are {} ({:.2f}) e1s have related rules.".format(len(left_e1), related_e1, related_e1 / len(left_e1)))
    

def create_mined_rules(args, triples_fn, rule_fn):
    import jpype
    import jpype.imports
    jvmPath = jpype.getDefaultJVMPath()
    jarpath="./rule_systems/AMIE/amie-milestone-intKB.jar"
    jpype.startJVM(jvmPath, "-ea", "-Djava.class.path=%s" % jarpath,"-Dfile.encoding=utf-8",convertStrings=True)
    from java.lang import System
    from java.io import PrintStream, File
    Test = jpype.JClass('amie/mining/AMIE')
    System.setOut(PrintStream(File(rule_fn))) 

    Test.main([triples_fn, "--minpca", "0.001", "--htr", "equalTo", "--maxad", "5", "--minhc", "0.001", "--mins", "15", "--nc", "1"])
    jpype.shutdownJVM()
    

def simplify_mined_rules(rule_fn, simplified_rule_fn):
    rules = []
    with open(rule_fn) as f:
        for line in f:
            if not line.startswith("?"):
                continue
            l = line.strip().split()
            assert (len(l) - 7 - 4) % 3 == 0
            rule_body_size = (len(l) - 7 - 4) // 3
            assert rule_body_size >= 1 and rule_body_size <= 4
            rule_body = l[:-11]
            rule_head = l[len(rule_body)+1:-7]
            head_coverage = float(l[-7])
            std_confidence = float(l[-6])
            pca_confidence = float(l[-5])
            rule = (rule_body, rule_head, head_coverage, std_confidence, pca_confidence)
            rules.append(rule)
    with open(simplified_rule_fn, "w") as f:
        for rule in rules:
            rule_body, rule_head, head_coverage, std_confidence, pca_confidence = rule 
            rule_body_size = (len(rule_body)) // 3
            if rule_head[1] == 'equalTo' and pca_confidence >= 0.5:
                e1_var = ""
                e2_var = "" 
                e3_var = "" 
                e4_var = ""
                for i in range(1, len(rule_body), 3):
                    if rule_body[i] == "equalTo":
                        e1_var = rule_body[i-1]
                        e2_var = rule_body[i+1]
                e3_var = rule_head[0]
                e4_var = rule_head[2]
                variable = set()
                for i in range(1, len(rule_body), 3):
                    variable.add(rule_body[i-1])
                    variable.add(rule_body[i+1])
                variable.add(rule_head[0])
                variable.add(rule_head[2])
                if len(set([e1_var, e2_var, e3_var, e4_var])) == 4:
                    # if rule_body_size != len(variable) - 1:
                    # draw_one_rule(rule, "{}.png".format(index))
                    f.write("{}\t=>\t{}\tpca:\t{}\n".format(rule_body, rule_head, pca_confidence))
   

def read_simplified_mined_rules(simplified_rule_fn, kgs):
    rules = []
    with open(simplified_rule_fn) as f:
        for line in f:
            cur = line
            line = line.strip()
            rule_body_origin = eval(line.split("=>")[0])
            rule_body = []
            
            for i in range(0, len(rule_body_origin), 3):
                if rule_body_origin[i+1] != 'equalTo':
                    rule_body.append([rule_body_origin[i+0], rule_body_origin[i+1], rule_body_origin[i+2]])
                else:
                    rule_body.append([rule_body_origin[i+0], rule_body_origin[i+1], rule_body_origin[i+2]])
            line = line.split("=>")[1]
            rule_head = eval(line.split("pca:")[0])
            rule_head = [(rule_head[i+0], rule_head[i+1], rule_head[i+2]) for i in range(0, len(rule_head), 3)]
            conf = eval(line.split("pca:")[1])
            rule_body_left = []
            rule_body_right = []
            rule_body_middle = []
            variable_color = {}   
            for body in rule_body:
                variable_color[body[0]] = -1
                variable_color[body[2]] = -1         
            for body in rule_body:
                if body[1] == "equalTo":
                    rule_body_middle.append(body)
                    variable_color[body[0]] = 1
                    variable_color[body[2]] = 2
            while True:
                for body in rule_body:
                    if variable_color[body[0]] == -1 and variable_color[body[2]] != -1:
                        variable_color[body[0]] = variable_color[body[2]]
                    if variable_color[body[0]] != -1 and variable_color[body[2]] == -1:
                        variable_color[body[2]] = variable_color[body[0]]
                flag = True
                for body in rule_body:
                    if variable_color[body[0]] == -1 or variable_color[body[2]] == -1:
                        flag = False
                        break
                if flag == True:
                    break
            flag = True # whether this rule is ok
            for body in rule_body:
                if body[1] != "equalTo":
                    if variable_color[body[0]] == 1 and variable_color[body[2]] == 1:
                        if body[1] not in kgs.kg1.relations_id_dict.keys():
                            flag = False
                            break
                            # print(cur)
                            # assert 0
                        body[1] = kgs.kg1.relations_id_dict[body[1]]
                        rule_body_left.append(body)
                    elif variable_color[body[0]] == 2 and variable_color[body[2]] == 2:
                        if body[1] not in kgs.kg2.relations_id_dict.keys():
                            flag = False
                            # assert 0
                            break
                        body[1] = kgs.kg2.relations_id_dict[body[1]]
                        rule_body_right.append(body)
                    else:
                        assert 0
            if flag == False:
                continue
            # else:
                # print(cur)
            # if len(rule_body_left) == 1 and len(rule_body_right) == 1:
            # print(rule_body_left)
            # print(rule_body_right)
            this_mined_rule = Rule(rule_body_left, rule_body_right, rule_body_middle, rule_head, conf)
            rules.append(this_mined_rule)
    return rules
