import math
import random

import data_utils as ut
import os


def get_object_of_entities(entity_list, triples):
    obj_dict = dict()
    object_list = list()
    for s, p, o in triples:
        obj_dict[s] = o
    for entity in entity_list:
        obj = obj_dict.get(entity, "[None]")
        object_list.append(obj)
    assert len(entity_list) == len(object_list)
    return object_list


def get_dict_from_list(items):
    assert len(items) == len(set(items))
    item_list = list(items)
    id_dict = dict()
    for item in item_list:
        id_dict[item] = len(id_dict)
    assert len(item_list) == len(id_dict)
    return item_list, id_dict


def triples2ids(triples, subj_dict, prop_dict, obj_dict):
    id_triples = set()
    for s, p, o in triples:
        id_triples.add((subj_dict[s], prop_dict[p], obj_dict[o]))
    return id_triples


def random_data_split(data_set, train_ratio, test_ratio):
    assert train_ratio + test_ratio < 1.0
    train_num = math.ceil(train_ratio * len(data_set))
    test_num = math.ceil(test_ratio * len(data_set))

    train_data = set(random.sample(data_set, train_num))
    test_data = set(random.sample(data_set - train_data, test_num))
    valid_data = data_set - train_data - test_data

    assert len(train_data) + len(test_data) + len(valid_data) == len(data_set)
    assert len(train_data & test_data & valid_data) == 0
    return train_data, valid_data, test_data


def generate_dbpedia_data(output_path="../data/dataset/",
                          input_path='../data/raw_data/dbpedia201610/',
                          language='en',
                          namespace='http://dbpedia.org/resource'):

    type_file = os.path.join(input_path, "lang/instance_types_lang.ttl").replace("lang", language)
    label_file = os.path.join(input_path, "lang/labels_lang.ttl").replace("lang", language)
    attribute_triple_file = os.path.join(input_path, "lang/mappingbased_literals_lang.ttl").replace("lang", language)
    relation_triple_file = os.path.join(input_path, "lang/mappingbased_objects_lang.ttl").replace("lang", language)
    print("\ngenerating dataset from", relation_triple_file, attribute_triple_file)

    relation_triples = ut.ttl_reader(relation_triple_file)
    head_entities_w_relations, relations, tail_entities_w_relations = ut.parse_statements(relation_triples)
    entities_w_relations = head_entities_w_relations | tail_entities_w_relations
    print("# relation triples before filtering:", len(relation_triples))
    print("# head entities having relations before filtering", len(head_entities_w_relations))
    print("# tail entities having relations before filtering", len(tail_entities_w_relations))
    print("# total entities having relations before filtering:", len(entities_w_relations))
    print("# relations before filtering:", len(relations))

    attribute_triples = ut.ttl_reader(attribute_triple_file)
    entities_w_attributes, attributes, values = ut.parse_statements(attribute_triples)
    print("# attribute triples before filtering:", len(attribute_triples))
    print("# entities having attributes before filtering:", len(entities_w_attributes))
    print("# attributes before filtering", len(attributes))

    total_entities = entities_w_attributes | entities_w_relations
    print("# total entities with at least one relation or attribute before filtering:", len(total_entities))
    print("# total entities with at least one relation and attribute before filtering:", len(entities_w_attributes & entities_w_relations))

    # remove the entities with different namespaces
    print("\n=== remove the entities with different namespaces ===")

    selected_entities = set()
    for ent in total_entities:
        if ent.startswith(namespace):
            selected_entities.add(ent)
    print("# total selected entities with at least one relation or attribute:", len(selected_entities))

    selected_relation_triples = set()
    for h, r, t in relation_triples:
        if h in selected_entities and t in selected_entities:
            selected_relation_triples.add((h, r, t))
    print("# total selected relation triples:", len(selected_relation_triples))

    selected_attribute_triples = set()
    for h, a, v in attribute_triples:
        if h in selected_entities:
            selected_attribute_triples.add((h, a, v))
    print("# total selected attribute triples:", len(selected_attribute_triples))

    _, selected_relations, _ = ut.parse_statements(selected_relation_triples)
    _, selected_attributes, selected_values = ut.parse_statements(selected_attribute_triples)

    type_triples = ut.ttl_reader(type_file)
    entities_w_types, _, _ = ut.parse_statements(type_triples)
    print("# type triples:", len(type_triples))
    print("# entities having types:", len(entities_w_types))

    label_triples = ut.ttl_reader(label_file)
    entities_w_labels, _, _ = ut.parse_statements(label_triples)
    print("# label triples:", len(label_triples))
    print("# entities having labels:", len(entities_w_labels))

    output_path = os.path.join(output_path, language)

    entities, entity_id_dict = get_dict_from_list(selected_entities)
    relations, relation_id_dict = get_dict_from_list(selected_relations)
    attributes, attribute_id_dict = get_dict_from_list(selected_attributes)
    values, value_id_dict = get_dict_from_list(selected_values)

    id_relation_triples = triples2ids(selected_relation_triples, entity_id_dict, relation_id_dict, entity_id_dict)
    id_attribute_triples = triples2ids(selected_attribute_triples, entity_id_dict, attribute_id_dict, value_id_dict)

    entity_labels = get_object_of_entities(entities, label_triples)
    entity_types = get_object_of_entities(entities, type_triples)

    print("\n=== statistics of the final dataset ===")
    print("# entities:", len(entities))
    print("# relations:", len(relations))
    print("# attributes:", len(attributes))
    print("# values:", len(values))
    print("# relation triples:", len(id_relation_triples))
    print("# attribute triples:", len(id_attribute_triples))
    print("=== end ===\n")

    ut.item_writer(entities, os.path.join(output_path, "entities"))
    ut.item_writer(relations, os.path.join(output_path, "relations"))
    ut.item_writer(attributes, os.path.join(output_path, "attributes"))
    ut.item_writer(values, os.path.join(output_path, "values"))

    ut.item_writer(entity_labels, os.path.join(output_path, "entity_labels"))
    ut.item_writer(entity_types, os.path.join(output_path, "entity_types"))

    ut.triple_writer(id_relation_triples, os.path.join(output_path, "relation_triples"))
    ut.triple_writer(id_attribute_triples, os.path.join(output_path, "attribute_triples"))


def alignment_split(source_namespace, target_namespace, source_folder, target_folder, output_folder,
                    source_ill_path, target_ill_path,
                    train_ratio=0.3, test_ratio=0.4):
    print("\nalignment split from", source_folder, target_folder)
    links = ut.ill_reader(source_ill_path, target_namespace)
    print("# total ills:", len(links))
    link_dic = dict()
    for subj, obj in links:
        link_dic[subj] = obj
    print("# total alignment from source:", len(link_dic))

    links = ut.ill_reader(target_ill_path, source_namespace)
    for obj, subj in links:
        if subj not in link_dic.keys():
            link_dic[subj] = obj
    print("# total alignment from source and target:", len(link_dic))

    source_entities = ut.item_reader(os.path.join(source_folder, "entities"), to_list=True)
    target_entities = ut.item_reader(os.path.join(target_folder, "entities"), to_list=True)
    print("# total source entities", len(source_entities), len(set(source_entities)))
    print("# total target entities", len(target_entities), len(set(target_entities)))

    _, source_ent_ids = get_dict_from_list(source_entities)
    _, target_ent_ids = get_dict_from_list(target_entities)
    source_rel_triples = ut.triple_reader(os.path.join(source_folder, "relation_triples"), to_int=True)
    target_rel_triples = ut.triple_reader(os.path.join(target_folder, "relation_triples"), to_int=True)
    heads, _, tails = ut.parse_statements(source_rel_triples)
    source_rel_entities = heads | tails
    heads, _, tails = ut.parse_statements(target_rel_triples)
    target_rel_entities = heads | tails
    print("# total source relational entities", len(source_rel_entities))
    print("# total target relational entities", len(target_rel_entities))

    alignment = set()
    for key, value in link_dic.items():
        key_id = source_ent_ids.get(key, None)
        value_id = target_ent_ids.get(value, None)
        if key_id is not None and value_id is not None:  # the linked entities are in our datasets
            if key_id in source_rel_entities and value_id in target_rel_entities:  # entities have relations
                alignment.add((key_id, value_id))
    print("# selected alignment:", len(alignment))
    ut.pair_writer(alignment, os.path.join(output_folder, "alignment"))

    train_alignment, valid_alignment, test_alignment = random_data_split(alignment, train_ratio, test_ratio)
    print("\n=== statistics of the final split ===")
    print("# training:", len(train_alignment))
    print("# validation:", len(valid_alignment))
    print("# test:", len(test_alignment))
    print("=== end ===\n")

    ut.pair_writer(train_alignment, os.path.join(output_folder, "train"))
    ut.pair_writer(valid_alignment, os.path.join(output_folder, "valid"))
    ut.pair_writer(test_alignment, os.path.join(output_folder, "test"))


def transductive_relation_triple_split(kg_folder, output_folder, train_ratio=0.9, test_ratio=0.05):
    print("\ntransductive relation triple split from", kg_folder)
    relation_triples = ut.triple_reader(os.path.join(kg_folder, "relation_triples"))
    print("# total relation triples:", len(relation_triples))
    triple_dic = dict()
    for h, r, t in relation_triples:
        h_triples = triple_dic.get(h, set())
        h_triples.add((h, r, t))
        triple_dic[h] = h_triples
    init_training_triples = set()
    for h in triple_dic.keys():
        n = 1
        triples = triple_dic.get(h)
        if len(triples) > 1:
            n = math.ceil(len(triples) / 2)
        selected_triples = random.sample(triples, n)
        init_training_triples |= set(selected_triples)
    print("# basic training triples:", len(init_training_triples))
    assert train_ratio + test_ratio < 1.0
    train_num = math.ceil(train_ratio * len(relation_triples))
    test_num = math.ceil(test_ratio * len(relation_triples))
    valid_num = len(relation_triples) - train_num - test_num
    assert train_num > len(init_training_triples)

    valid_test_triple_candidates = set()
    heads, rels, tails = ut.parse_statements(init_training_triples)
    ents = heads | tails
    for h, r, t in relation_triples - init_training_triples:
        if h in ents and r in rels and t in ents:
            valid_test_triple_candidates.add((h, r, t))
    print("# valid and test triple candidates:", len(valid_test_triple_candidates))

    assert len(valid_test_triple_candidates) > test_num + valid_num

    test_data = set(random.sample(valid_test_triple_candidates, test_num))
    valid_data = set(random.sample(valid_test_triple_candidates - test_data, valid_num))
    train_data = relation_triples - valid_data - test_data

    print("\n=== statistics of the final split ===")
    print("# training:", len(train_data))
    print("# validation:", len(valid_data))
    print("# test:", len(test_data))
    print("=== end ===\n")

    assert len(init_training_triples - train_data) == 0
    assert len(train_data) + len(test_data) + len(valid_data) == len(relation_triples)
    assert len(train_data & test_data & valid_data) == 0

    ut.triple_writer(train_data, os.path.join(output_folder, "train"))
    ut.triple_writer(valid_data, os.path.join(output_folder, "valid"))
    ut.triple_writer(test_data, os.path.join(output_folder, "test"))


if __name__ == '__main__':
    relation_triples = ut.relational_ttl_reader("./mappingbased_objects_en.ttl")
    ut.triple_writer(random.sample(relation_triples, len(relation_triples) // 10), "./selected_dbp_triples.txt")
