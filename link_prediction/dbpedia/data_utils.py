import time

from rdflib import Graph


def ttl_reader(ttl_path):
    t = time.time()
    g = Graph()
    g.parse(ttl_path, format="turtle")
    # strip() is important!!!
    statements = set([(subj.toPython().strip(), prop.toPython().strip(), str(obj.toPython()).strip())
                      for subj, prop, obj in g.triples((None, None, None))])
    print("reading ttl costs time {:.3f} s".format(time.time() - t))
    return statements


def ill_reader(ill_path, target_namespace):
    # much faster than rdflib
    start_time = time.time()
    target_namespace = '<'+target_namespace
    links = set()
    with open(ill_path, 'r') as file:
        for line in file:
            if line.startswith("#"):  # skip the first and last lines of ttl
                continue
            params = line.strip(' .\n').split(' ')
            if len(params) == 3 and params[2].startswith(target_namespace):
                h = params[0].lstrip('<').rstrip('>').strip()
                t = params[2].lstrip('<').rstrip('>').strip()
                links.add((h, t))
    file.close()
    print("reading ills costs time {:.3f} s".format(time.time() - start_time))
    return links


def relational_ttl_reader(ill_path):
    # much faster than rdflib
    start_time = time.time()
    triples = set()
    with open(ill_path, 'r') as file:
        for line in file:
            if line.startswith("#"):  # skip the first and last lines of ttl
                continue
            params = line.strip(' .\n').split(' ')
            if len(params) == 3:
                h = params[0].strip().lstrip('<').rstrip('>').strip()
                r = params[1].strip().lstrip('<').rstrip('>').strip()
                t = params[2].strip().lstrip('<').rstrip('>').strip()
                triples.add((h, r, t))
    file.close()
    print("reading relational ttl costs time {:.3f} s".format(time.time() - start_time))
    return triples


def parse_statements(statements):
    subjects = set()
    properties = set()
    objects = set()
    for s, p, o in statements:
        subjects.add(s)
        properties.add(p)
        objects.add(o)
    return subjects, properties, objects


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


def item_reader(file_path, to_list=False):
    if to_list:
        items = list()
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file:
                items.append(line.strip('\n'))
    else:
        items = set()
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file:
                items.add(line.strip('\n'))
    return items


def triple_reader(file_path, to_list=False, to_int=True):
    if to_list:
        triples = list()
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file:
                subj, prop, obj = line.strip('\n').split('\t')
                if to_int:
                    triples.append((int(subj), int(prop), int(obj)))
                else:
                    triples.append((subj, prop, obj))
    else:
        triples = set()
        with open(file_path, 'r', encoding='utf8') as file:
            for line in file:
                subj, prop, obj = line.strip('\n').split('\t')
                if to_int:
                    triples.add((int(subj), int(prop), int(obj)))
                else:
                    triples.add((subj, prop, obj))
    return triples
