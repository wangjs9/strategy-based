import json, os
from tqdm import tqdm
import networkx as nx
import spacy
nlp = spacy.load('en_core_web_md', disable=['ner', 'parser', 'textcat'])
from multiprocessing import Pool

import config
from dataprocess.build_graph import load_resources

def load_cpnet():
    global concept2id, relation2id, id2relation, id2concept
    global cpnet, cpnet_simple

    concept2id, relation2id, id2relation, id2concept = load_resources()
    cpnet = nx.read_gpickle(config.conceptnet_graph)
    if os.path.exists(config.data_concept_dict+'simple_concept.graph'):
        cpnet_simple = nx.read_gpickle(config.data_concept_dict+'simple_concept.graph')

    else:
        cpnet_simple = nx.Graph()
        for u, v, data in cpnet.edges(data=True):
            w = data['weight'] if 'weight' in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]['weight'] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        nx.write_gpickle(cpnet_simple, config.data_concept_dict+'simple_concept.graph')

def match_concepts(start, end, emotion):
    res = []
    for idx, start_concepts in tqdm(enumerate(start), total=len(start)):
        start_concepts = [cp for cp in start_concepts if cp in concept2id]
        end_concepts = [cp for cp in end[idx] if cp not in start_concepts and cp in concept2id]
        res.append({'start': start_concepts, 'end': end_concepts, 'emotion': emotion[idx]})
    return res

def find_neighbours_frequency(source_concepts, target_concepts, emotion, T, max_B=50, max_search=50):
    start = [concept2id[s_cpt] for s_cpt in source_concepts if s_cpt in concept2id]  # start nodes for each turn
    Vts = dict([(x, 0) for x in start])  # nodes and their turn

    if len(source_concepts) == 0:
        return {"concepts": [], "labels": [], "distances": [], "triples": []}, -1, 0
    Ets = {} # nodes and their neighbor in last turn
    target_concpet_nlp = nlp(' '.join(target_concepts))
    ts = [concept2id[t_cpt] for t_cpt in target_concepts]
    for t in range(T):
        V = {}
        for s in start:
            if s in cpnet_simple and s not in target_concepts:
                candidates = [c for c in cpnet_simple[s] if c not in Vts]
                candidates_nlp = nlp(' '.join([id2concept[c] for c in candidates]))

                scores = {id: max([c.similarity(t_cpt) for t_cpt in target_concpet_nlp]+[0])+c.similarity(nlp(emotion)[0]) for id, c in zip(candidates, candidates_nlp)}
                candidates = sorted(list(scores.items()), key=lambda x: x[1], reverse=True)[:max_search]
                candidates = [x[0] for x in candidates]
                for c in candidates:
                    if c not in V:
                        V[c] = scores[c]
                    else:
                        V[c] += scores[c]
                    rels = get_edge(s, c)
                    if len(rels) > 0:
                        if c not in Ets:
                            Ets[c] = {s: rels}
                        else:
                            Ets[c].update({s: rels})

        V = list(V.items())
        count_V = sorted(V, key=lambda x: x[1], reverse=True)[:max_B] # the top max_B nodes related to entities in start concepts
        start = [x[0] for x in count_V] # filter the nodes excluded from the dataset

        Vts.update(dict([(x, t + 1) for x in start])) # add new nodes

    concepts = list(Vts.keys())
    distances = list(Vts.values())
    assert (len(concepts) == len(distances))

    triples = []

    for v, N in Ets.items():
        if v in concepts:
            for u, rels in N.items():
                triples.append((u, rels, v))

    labels = []
    found_num = 0
    for c in concepts:
        if c in ts:
            found_num += 1
            labels.append(1)
        else:
            labels.append(0)

    res = [id2concept[x].replace("_", " ") for x in concepts]
    triples = [(id2concept[x].replace("_", " "), y, id2concept[z].replace("_", " ")) for (x, y, z) in triples]
    return {"concepts": res, "labels": labels, "distances": distances, "triples": triples}, found_num, len(res)

def get_edge(src_concept, tgt_concept):
    try:
        rel_list = cpnet[src_concept][tgt_concept]
        return list(set([rel_list[item]['rel'] for item in rel_list]))
    except:
        return []

def DivJobList(JobList, coreNum=1):
    newJobList = []
    length = len(JobList)
    left = length % coreNum
    right = coreNum - left
    step = int(length / coreNum)
    for i in range(right):
        newJobList.append(JobList[:step])
        JobList = JobList[step:]
    step += 1
    for i in range(left):
        newJobList.append(JobList[:step])
        JobList = JobList[step:]
    return newJobList

def FindPath(conceptVer, T=3, max_B=50):
    examples = []
    avg_len, avg_found, total_vaild = 0, 0, 0
    for pair in tqdm(conceptVer, total=len(conceptVer)):
        info, found, avg_nodes = find_neighbours_frequency(pair['start'], pair['end'], pair['emotion'], T, max_B)
        avg_len += avg_nodes
        if found != -1:
            avg_found += found
            total_vaild += 1
        examples.append(info)

    return [examples, avg_len, avg_found, total_vaild]

def sub_process(conceptVer, T=3, max_B=50, coreNum=10):
    print('Finding paths ...')
    pool = Pool(coreNum)
    job_list = DivJobList(conceptVer, coreNum)
    # Tlist = [T for i in range(coreNum)]
    # Blist = [max_B for i in range(coreNum)]
    # job_list = zip(job_list, Tlist, Blist)
    return_dict = pool.map(FindPath, job_list)

    pool.close()
    pool.join()

    examples = []
    avg_len, avg_found, total_vaild = 0, 0, 0
    for idx in range(coreNum):
        exa, al, af, tv = return_dict[idx]
        examples.extend(exa)
        avg_len += al
        avg_found += af
        total_vaild += tv

    return examples, avg_len, avg_found, total_vaild


def process(save_path, start, end, emotion, mode='train', T=3, max_B=50, coreNum=10):
    load_cpnet()
    print('Generating concept version of causality pairs...')
    if os.path.exists(config.data_concept_dict + 'sys_concepts.{}.json'.format(mode)):
        conceptVer = []
        with open(config.data_concept_dict + 'sys_concepts.{}.json'.format(mode), 'r') as f:
            data = f.readlines()
        for line in data:
            conceptVer.append(json.loads(line))

    else:
        conceptVer = match_concepts(start, end, emotion)
        with open(config.data_concept_dict + 'sys_concepts.{}.json'.format(mode), 'w') as f:
            for line in conceptVer:
                json.dump(line, f)
                f.write('\n')
    print('Done')
    # FindPath(conceptVer)
    examples, avg_len, avg_found, total_vaild = sub_process(conceptVer, T, max_B, coreNum)

    print('{} hops avg nodes: {} avg_path: {}'.format(T, avg_len / len(examples), avg_found / total_vaild))
    with open(save_path, 'w') as f:
        for i, line in enumerate(examples):
            json.dump(line, f)
            f.write('\n')
