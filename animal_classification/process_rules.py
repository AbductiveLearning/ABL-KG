from tqdm import tqdm
import time
from multiprocessing import Pool
from functools import partial
from nltk.stem import WordNetLemmatizer
import nltk

def process_scene_name(name):
    try:
        wnl = WordNetLemmatizer()
    except:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        wnl = WordNetLemmatizer()
    name = name.lower()
    name = name.replace("_", " ")
    if name.startswith("a "):
        name = name[2:]
    if name.startswith("an "):
        name = name[3:]
    if name.startswith("the "):
        name = name[4:]
    name = wnl.lemmatize(name, 'n')
    return name

def filter_rules(rule_list, scene_list, object_list):
    ret_rule_list = []
    for rule in rule_list:
        left, right = rule[0], rule[1]
        ok = True
        for literal in left:
            if literal[0]==1 and literal[1] in object_list:
                ok = False
                break
        for literal in right:
            if literal[0]==1 and literal[1] in scene_list and not (left[0][0]==1 and left[0][1] in scene_list): # scene->scene is ok
                ok = False
                break
        if ok:
            ret_rule_list.append(rule)
        else:
            pass
            # print("Filter: ", rule)
    return ret_rule_list

# Ensure: the entities are already lower case
def get_rule_list(rule_file):
    rule_list = []
    with open(rule_file, "r") as f:
        for line in f:
            rule = eval(line)
            rule_list.append(rule)
    return rule_list

def get_rule_names(rule_list):
    name_set = set()
    for rule in rule_list:
        for i in range(2):
            for literal in rule[i]:
                name_set.add(literal[1])
    return list(name_set)

def rules_to_clauses(rule_list):
    clause_list = []
    for rule in rule_list:
        clause = []
        left, right, conf = rule[0], rule[1], rule[2]
        assert(len(right)==1)
        for literal in left:
            neg_l = (1-literal[0], literal[1], literal[2])
            clause.append(neg_l)
        for literal in right:
            l = (literal[0], literal[1], literal[2])
            clause.append(l)
        clause.sort()
        clause_list.append((clause, conf))
    return clause_list

def clauses_to_rules(clause_list):
    rule_list = []
    for clause in clause_list:
        rule = []
        left, right, conf = [], [], clause[1]
        for literal in clause[0][:-1]:
            left.append((1-literal[0], literal[1], literal[2]))
        right.append(clause[0][-1])
        rule = (left, right, conf)
        rule_list.append(rule)
    return rule_list

def resolution(c1, c2, resol_reserve_names, all_clause_list = [], threshold = 0.5):
    clause1, conf1 = c1[0], c1[1]
    clause2, conf2 = c2[0], c2[1]
    conf = conf1 * conf2
    if conf < threshold:
        return None
    resolvent = []
    neg_idx, pos_idx = -1, -1
    # Ignore inconsistency
    ok = False
    for i in range(len(clause1)-1, -1, -1):
        for j in range(len(clause2)-1, -1, -1):
            li, lj = clause1[i], clause2[j]
            if li[0]+lj[0] == 1 and li[1] == lj[1] and li[2] == lj[2]:
                neg_idx, pos_idx = i, j
                ok = True
                break
        if ok:
            break
            
    if neg_idx >= 0 and pos_idx >= 0:
        resolvent = clause1[:neg_idx]+clause1[neg_idx+1:]+clause2[:pos_idx]+clause2[pos_idx+1:]
    if len(resolvent)== 0 or is_tautology(resolvent):
        return None
    resolvent.sort()
    # Simplify (for 2CNF)
    if len(resolvent)>=2 and resolvent[0]==resolvent[1]:
        resolvent = resolvent[1:]
    # Contradictory rules, return False
    if len(resolvent)==1 and len(clause1)==2 and len(clause2)==2:
        return False
    # The resolvent must have particular names
    if not has_name((resolvent, conf), resol_reserve_names):
        return None
    # The resolution exists, then return None
    if len(all_clause_list)>0 and contain_clause(all_clause_list, (resolvent, conf)):
        return None
    return (resolvent, conf)

def is_tautology(clause):
    n_l = len(clause)
    for i in range(n_l):
        for j in range(i+1, n_l):
            li, lj = clause[i], clause[j]
            if li[0]+lj[0]==1 and li[1]==lj[1] and li[2]==lj[2]:
                # print("is tautology:", clause)
                return True
    return False

# def forget_name(clause_list, name):
#     ret_clause_list = clause_list.copy()
#     # Find all the clauses that have the name
#     pos_clause_list, neg_clause_list = [], []
#     for clause in ret_clause_list:
#         for literal in clause:
#             if literal[1]==name:
#                 if literal[0]==0:
#                     neg_clause_list.append(clause)
#                 else:
#                     pos_clause_list.append(clause)
#     # print(neg_clause_list)
#     # print(pos_clause_list)
#     # Do forgetting
#     # Resolution
#     for neg_clause in neg_clause_list:
#         for pos_clause in pos_clause_list:
#             resolvent = resolution(neg_clause, pos_clause, name)
#             if len(resolvent) > 0 and resolvent not in ret_clause_list:
#                 ret_clause_list.append(resolvent)
#                 # print("add ", resolvent)
#     # Forget clause
#     for clause in neg_clause_list:
#         ret_clause_list.remove(clause)
#     for clause in pos_clause_list:
#         try:
#             ret_clause_list.remove(clause)
#         except:
#             print("remove ", clause, " failed")
#     return ret_clause_list

# def forget(rule_list, forget_names):
#     ret_rule_list = []
#     clause_list = rules_to_clauses(rule_list)
#     for name in tqdm(forget_names):
#         if name in ['human']:
#             continue
#         print("forgetting ", name)
#         clause_list = forget_name(clause_list, name)
#         print("after forgetting ", name, len(clause_list))
#     ret_rule_list = clauses_to_rules(clause_list)
#     return ret_rule_list

def has_name(clause, name_list):
    for literal in clause[0]:
        if literal[1] in name_list:
            return True
    return False

def get_exist_reserve_clauses(clause_list, reserve_names):
    new_list = []
    for clause in clause_list:
        if has_name(clause, reserve_names):
            new_list.append(clause)
    return new_list

def get_reserve_clauses(clause_list, reserve_names):
    reserve_clause_list = []
    for clause in clause_list:
        flag = True
        for literal in clause[0]: # For all literals in a clause, they must in reserve_names
            if literal[1] not in reserve_names:
                flag = False
                break
        if flag:
            reserve_clause_list.append(clause)
    return reserve_clause_list

def contain_clause(clause_list, target_clause):
    for c in clause_list:
        if c[0] == target_clause[0]:
            return True
    return False

def resol_clause_list(c1, clause_list, resol_reserve_names, resol_thres):
    resolvent_list = [resolution(c1, c2, resol_reserve_names, clause_list, resol_thres) for c2 in clause_list]
    return resolvent_list

def resol_list_list(new_list, clause_list, resol_reserve_names, resol_thres = 0.5):
    partial_resol_clause_list = partial(resol_clause_list, clause_list=clause_list, resol_reserve_names = resol_reserve_names, resol_thres = resol_thres)
    with Pool() as pool:
        resolution_list = pool.map(partial_resol_clause_list, new_list)
        # list(tqdm(pool.imap(partial_resol_clause_list, new_list), total=len(new_list)))
    return resolution_list

def reserve(rule_list, resol_reserve_names, rule_reserve_names, resol_time, resol_thres, out_rule_filename = "rule_forgetting_KGname.txt"):
    clause_list = rules_to_clauses(rule_list)
    new_list = get_exist_reserve_clauses(clause_list, resol_reserve_names)
    for t in range(1, resol_time+1):
        print("Begin resolution...(%dx%d)"%(len(new_list),len(clause_list)))
        time_start=time.time()
        resolution_list = resol_list_list(new_list, clause_list, resol_reserve_names, resol_thres)
        assert(len(resolution_list)==len(new_list) and len(resolution_list[0])==len(clause_list))
        time_end=time.time()
        print('time cost1',time_end-time_start,'s')

        print("Finding contradictory rules...")
        contra_i_idxs, contra_j_idxs = [], []
        cand_resolvent_list, cand_i_list, cand_j_list = [], [], []
        for i in tqdm(range(len(new_list))):
            for j in range(len(clause_list)):
                c1, c2, resolvent = new_list[i], clause_list[j], resolution_list[i][j]
                if resolvent is None:
                    continue
                if resolvent is False: # Contradictory, record the idxs
                    contra_i_idxs.append(i); contra_j_idxs.append(j)
                    print(c1, c2)
                    continue
                # Accepted as candidate
                cand_resolvent_list.append(resolvent)
                cand_i_list.append(i); cand_j_list.append(j)

        print("Generating final rules...")
        resolvent_clause_set, resolvent_list = set(), []
        for cand_resolvent,cand_i,cand_j in zip(cand_resolvent_list,cand_i_list,cand_j_list):
            # If contradictory (a->b and a->not b), then remove both clauses derive from them
            if cand_i in contra_i_idxs or cand_j in contra_j_idxs: 
                continue
            cand_resolvent_str = str(cand_resolvent[0])
            if cand_resolvent_str not in resolvent_clause_set: # remove duplicate
                resolvent_list.append(cand_resolvent)
                resolvent_clause_set.add(cand_resolvent_str)
                if len(get_reserve_clauses([cand_resolvent], rule_reserve_names))>0:
                    print(new_list[cand_i], clause_list[cand_j], cand_resolvent)
                # if len(get_reserve_clauses([cand_resolvent], ['bone', 'mammal']))>0:###debug
                #     print(new_list[cand_i], clause_list[cand_j], cand_resolvent)
                #     input()
        if len(resolvent_list)==0 and t!=1:
            break

        print("Number of resolvent:", len(resolvent_list))
        clause_list.extend(resolvent_list)
        new_list = resolvent_list.copy()

        reserve_clause_list = get_reserve_clauses(clause_list, rule_reserve_names) # Can have rule_reserve_names for all literal
        reserve_clause_list = get_exist_reserve_clauses(reserve_clause_list, resol_reserve_names) # Must exist resol_reserve_names
        ret_rule_list = clauses_to_rules(reserve_clause_list)
        # Write rule file
        write_rules(ret_rule_list, out_rule_filename%t)
        # break########
    return ret_rule_list
                
def write_rules(rule_list, filename):
    with open(filename, "w") as f:
        for rule in rule_list:
            f.write(str(rule)+'\n')
    print("Rule file saved in ", filename)
