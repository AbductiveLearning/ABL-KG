from collections import Counter


def triple_writer(triples, output_path, separator="\t", linebreak="\n"):
    file = open(output_path, 'w', encoding='utf8')
    for s, p, o in triples:
        file.write(str(s) + separator + str(p) + separator + str(o) + linebreak)
    file.close()


def pair_writer(pairs, output_path, separator="\t", linebreak="\n"):
    file = open(output_path, 'w', encoding='utf8')
    for p1, p2 in pairs:
        file.write(str(p1) + separator + str(p2) + linebreak)
    file.close()


def item_writer(items, output_path, linebreak="\n"):
    file = open(output_path, 'w', encoding='utf8')
    for i in items:
        file.write(str(i) + linebreak)
    file.close()


def read_relation_triples(file_path):
    # print("read relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[2].strip()
        t = params[1].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    return triples, entities, relations


def get_rt_dict(triples):
    rt_dic = dict()
    ht_dic = dict()
    hr_dic = dict()
    for h, r, t in triples:
        rts = rt_dic.get(h, set())
        rts.add((r, t))
        rt_dic[h] = rts

        rs = ht_dic.get((h, t), set())
        rs.add(r)
        ht_dic[(h, t)] = rs

        hrs = hr_dic.get(t, set())
        hrs.add((h, r))
        hr_dic[t] = hrs

    print("# rts dict:", len(rt_dic))
    return rt_dic, ht_dic, hr_dic


def mine_rules(triples, rts_dic, rs_dic):
    rule_relation_triples = list()
    for h, rts in rts_dic.items():
        for r, t in rts:
            t_rts = rts_dic.get(t, set())
            if len(t_rts) > 0:
                for rr, tt in t_rts:
                    htt_rs = rs_dic.get((h, tt), set())
                    if len(htt_rs) > 0:
                        for rrr in htt_rs:
                            # rule_relation_triples.append((r, rr, rrr))
                            # if rrr != r:
                            if rrr != r and rrr != rr:
                                rule_relation_triples.append((r, rr, rrr))
    return rule_relation_triples


def infer_triples(seen_triples, unseen_triples, rules):
    rts_dic, rs_dic, hr_dic = get_rt_dict(seen_triples)
    selected_triples = set()
    for h, r, t in unseen_triples:
        rts = rts_dic.get(h, set())
        if len(rts) > 0:
            for rr, tt in rts:
                rrrttts = rts_dic.get(tt, set())
                if len(rrrttts) > 0:
                    for rrr, ttt in rrrttts:
                        if ttt == t:
                            if (rr, rrr, r) in rules:
                                selected_triples.add((h, r, t))
                                # print(h, rr, tt, "&", tt, rrr, ttt, "->", h, r, t)
    return selected_triples


def read_rules(rule_file):
    rule_rels = set()
    with open(rule_file) as f:
        for line in f:
            params = line.split(" ")
            assert len(params) == 4
            rule_rels.add((params[0], params[1], params[2]))
    print("rules:", len(rule_rels))
    return rule_rels
