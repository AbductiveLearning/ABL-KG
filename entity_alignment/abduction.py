from utils import read_simplified_mined_rules, create_mined_rules, simplify_mined_rules
from rule import Grounding
import os
from collections import defaultdict


def abduction_learning(args, kgs, sup_relation_uri_pairs, pre_abducted_pairs, sim_mat, kg1_id_to_idx, kg2_id_to_idx, whether_use_mined_rule):
    if whether_use_mined_rule:
        # simplified_mined_rules_fn = os.path.join(args.training_data, "rules/simplified_mined_rules")
        simplified_mined_rules_fn = os.path.join(args.training_data, "rules/rule_expert.txt")
        if os.path.exists(simplified_mined_rules_fn):
            mined_rules = read_simplified_mined_rules(simplified_mined_rules_fn, kgs)
        else:
            triples = []
            def readTriples(fn):
                res = []
                with open(fn) as f:
                    for line in f:
                        h, r, t = line.strip().split()
                        res.append([h, r, t])
                return res
            for fn in ["rel_triples_1", "rel_triples_2"]:
                triples.extend(readTriples(os.path.join(args.training_data, fn)))
            def readlinks(fn):
                res = []
                with open(fn) as f:
                    for line in f:
                        h, t = line.strip().split()
                        res.append([h, "equalTo", t])
                return res
            triples.extend(readlinks(os.path.join(args.training_data, args.dataset_division, "train_links")))
            to_mined_triples_fn = os.path.join(args.training_data, "rules/to_mined_triples")
            with open(to_mined_triples_fn, "w") as f:
                for triple in triples:
                    f.write("\t".join(triple) + "\n")     
            print("start mining rules")
            mined_rules_fn = os.path.join(args.training_data, "rules/mined_rules")
            create_mined_rules(self.args, to_mined_triples_fn, mined_rules_fn)
            simplify_mined_rules(mined_rules_fn, simplified_mined_rules_fn)
    else:
        mined_rules = []

    kg1 = {"hr_ts": defaultdict(set), "tr_hs": defaultdict(set), "triples": set()}
    kg2 = {"hr_ts": defaultdict(set), "tr_hs": defaultdict(set), "triples": set()}
    
    def read_triples(kg, entities_id_dict, relations_id_dict, fn):
        with open(fn) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, t = entities_id_dict[h], entities_id_dict[t] 
                r = relations_id_dict[r]
                kg["hr_ts"][(h, r)].add(t)
                kg["tr_hs"][(t, r)].add(h)
                kg["triples"].add((h, r, t))
                
    read_triples(kg1, kgs.kg1.entities_id_dict, kgs.kg1.relations_id_dict, os.path.join(args.training_data, "rel_triples_1"))
    read_triples(kg2, kgs.kg2.entities_id_dict, kgs.kg2.relations_id_dict, os.path.join(args.training_data, "rel_triples_2"))
    ############################################################################################################################
    all_groundings = obtain_all_groundings(mined_rules, pre_abducted_pairs, kgs.train_links, kg1, kg2)
    ###################################################################
    post_abducted_pairs_set = minimize_inconsistency(all_groundings, pre_abducted_pairs, kgs.train_links, sim_mat, kg1_id_to_idx, kg2_id_to_idx)
    ################################
    groundings = obtain_all_groundings(mined_rules, post_abducted_pairs_set, kgs.train_links, kg1, kg2)
    train_links_map = {}
    for train_e1, train_e2 in kgs.train_links:
        train_links_map[train_e1] = train_e2
    inferred_entity_pairs_set = set()
    for grounding in groundings:
        if grounding.inferred_entity_alignment[0] in set(train_links_map.keys()):
            continue
        if grounding.inferred_entity_alignment[1] in set(train_links_map.values()):
            continue
        inferred_entity_pairs_set.add(grounding.inferred_entity_alignment)
    inferred_entity_pairs_set = inferred_entity_pairs_set - set(kgs.train_links) - post_abducted_pairs_set
    ###############################
    
    return post_abducted_pairs_set, inferred_entity_pairs_set, mined_rules

def obtain_all_groundings(mined_rules, pre_abducted_pairs, sup_entity_pairs, kg1, kg2):
    all_observation_pairs = []
    for e1, e2 in pre_abducted_pairs:
        all_observation_pairs.append((e1, e2))
    for e1, e2 in sup_entity_pairs:
        all_observation_pairs.append((e1, e2))
    all_groundings = []
    def instantiate(rule, instantiation, kg1, kg2, rule_body, depth):
        if depth == len(rule_body):
            assert len(instantiation) == len(rule.variables)
            assert instantiation.keys() == rule.variables
            all_groundings.append(Grounding(rule, instantiation.copy()))
            return
        body = rule_body[depth]
        if body[0] in instantiation.keys() and body[2] not in instantiation.keys():

            for e_id in kg1["hr_ts"][(instantiation[body[0]], body[1])]:
                instantiation[body[2]] = e_id
                instantiate(rule, instantiation, kg1, kg2, rule_body, depth+1)
                instantiation.pop(body[2])
            for e_id in kg2["hr_ts"][(instantiation[body[0]], body[1])]:
                instantiation[body[2]] = e_id
                instantiate(rule, instantiation, kg1, kg2, rule_body, depth+1)
                instantiation.pop(body[2])
        elif body[0] not in instantiation.keys() and body[2] in instantiation.keys():

            for e_id in kg1["tr_hs"][(instantiation[body[2]], body[1])]:
                instantiation[body[0]] = e_id
                instantiate(rule, instantiation, kg1, kg2, rule_body, depth+1)
                instantiation.pop(body[0])
            for e_id in kg2["tr_hs"][(instantiation[body[2]], body[1])]:
                instantiation[body[0]] = e_id
                instantiate(rule, instantiation, kg1, kg2, rule_body, depth+1)
                instantiation.pop(body[0])
        elif body[0] in instantiation.keys() and body[2] in instantiation.keys():
            if (instantiation[body[0]], body[1], instantiation[body[2]]) in kg1["triples"]:
                instantiate(rule, instantiation, kg1, kg2, rule_body, depth+1)
            if (instantiation[body[0]], body[1], instantiation[body[2]]) in kg2["triples"]:
                instantiate(rule, instantiation, kg1, kg2, rule_body, depth+1)
        else:
            print(rule.rule_body_left + rule.rule_body_right + rule.rule_body_middle)
            print(instantiation)
            print(body[0])
            print(body[2])
            assert 0
    for rule in mined_rules: 
        for e1, e2 in all_observation_pairs:
            this_instantiation = {}
            this_instantiation[rule.rule_body_middle[0][0]] = e1
            this_instantiation[rule.rule_body_middle[0][2]] = e2

            rule_body = []
            for body in rule.rule_body_left + rule.rule_body_right:
                if body[0] == rule.rule_body_middle[0][0] or body[0] == rule.rule_body_middle[0][2] or body[2] == rule.rule_body_middle[0][0] or body[2] == rule.rule_body_middle[0][2]:
                    rule_body.append(body)
            for body in rule.rule_body_left + rule.rule_body_right:
                if not(body[0] == rule.rule_body_middle[0][0] or body[0] == rule.rule_body_middle[0][2] or body[2] == rule.rule_body_middle[0][0] or body[2] == rule.rule_body_middle[0][2]):
                    rule_body.append(body)
            instantiate(rule, this_instantiation, kg1, kg2, rule_body, 0)
    return all_groundings



def minimize_inconsistency(all_groundings, pre_abducted_pairs, sup_entity_pairs, entity_similarities, kg1_id_to_idx, kg2_id_to_idx):
    graph = {} 
    pre_abducted_pairs_set = set(pre_abducted_pairs)
    sup_entity_pairs_set = set(sup_entity_pairs)
    
    for i, pair1 in enumerate(pre_abducted_pairs_set):
        for j, pair2 in enumerate(pre_abducted_pairs_set):
            if i < j:
                if (pair1[0] == pair2[0] and pair1[1] != pair2[1]) or (pair1[0] != pair2[0] and pair1[1] == pair2[1]):
                    
                    if pair1 not in graph.keys():
                        graph[pair1] = {}
                    if pair2 not in graph[pair1].keys():
                        graph[pair1][pair2] = 0
                    graph[pair1][pair2] += 1
                    
                    if pair2 not in graph.keys():
                        graph[pair2] = {}
                    if pair1 not in graph[pair2].keys():
                        graph[pair2][pair1] = 0
                    graph[pair2][pair1] += 1

    for grounding in all_groundings:
        inferred_entity_alignment = grounding.inferred_entity_alignment
        inferred_by_entity_alignment = grounding.inferred_by_entity_alignment
        for pairs_set in [pre_abducted_pairs_set, sup_entity_pairs_set]:

            for i, pair in enumerate(pairs_set):
                if (pair[0] == inferred_entity_alignment[0] and pair[1] != inferred_entity_alignment[1]) or (pair[0] != inferred_entity_alignment[0] and pair[1] == inferred_entity_alignment[1]):
                    
                    if pair not in graph.keys():
                        graph[pair] = {}
                    if inferred_by_entity_alignment not in graph[pair].keys():
                        graph[pair][inferred_by_entity_alignment] = 0
                    if inferred_by_entity_alignment in sup_entity_pairs_set:
                        graph[pair][inferred_by_entity_alignment] += grounding.rule.conf
                    else:
                        graph[pair][inferred_by_entity_alignment] += grounding.rule.conf * entity_similarities[kg1_id_to_idx[inferred_by_entity_alignment[0]]][kg2_id_to_idx[inferred_by_entity_alignment[1]]]
                    
                    if inferred_by_entity_alignment not in graph.keys():
                        graph[inferred_by_entity_alignment] = {}
                    if pair not in graph[inferred_by_entity_alignment].keys():
                        graph[inferred_by_entity_alignment][pair] = 0
                    if inferred_by_entity_alignment in sup_entity_pairs_set:
                        graph[inferred_by_entity_alignment][pair] += grounding.rule.conf
                    else:
                        graph[inferred_by_entity_alignment][pair] += grounding.rule.conf * entity_similarities[kg1_id_to_idx[inferred_by_entity_alignment[0]]][kg2_id_to_idx[inferred_by_entity_alignment[1]]]
    
    node_score = {}
    for node in graph.keys():
        node_score[node] = 0
        for out_node, edge_score in graph[node].items():
            node_score[node] += edge_score
        if node in sup_entity_pairs_set:
            node_score[node] -= 1
        else:
            node_score[node] -= entity_similarities[kg1_id_to_idx[node[0]]][kg2_id_to_idx[node[1]]]
    
    conflict_node_set = set() 
    
    while len(node_score) > 0:
        max_node = max(node_score, key=node_score.get)
        while max_node in sup_entity_pairs_set:
            node_score.pop(max_node) 
            if len(node_score) == 0:
                break
            max_node = max(node_score, key=node_score.get)
        
        if len(node_score) == 0:
            break
        if node_score[max_node] <= 0:
            break
        
        for max_node_neigh in graph[max_node].keys():
            if max_node_neigh in node_score.keys():
                node_score[max_node_neigh] -= graph[max_node][max_node_neigh]
       
        node_score.pop(max_node)
        conflict_node_set.add(max_node)
        
    post_abducted_pairs_set = set()
    for node in pre_abducted_pairs_set:
        if node not in conflict_node_set:
            post_abducted_pairs_set.add(node)
    return post_abducted_pairs_set
