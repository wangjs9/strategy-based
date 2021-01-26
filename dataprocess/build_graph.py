import pandas as pd
from tqdm import tqdm
from networkx import MultiDiGraph, write_gpickle
import nltk, json, os, pickle
from nltk.stem import WordNetLemmatizer
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["gone", "did", "going", "would", "could", "get", "in", "up", "may", "uk", "us", "take", "make", "object", "person", "people"]

import config

concept_vocab = '../conceptnet/concept.txt'
concept_rel = '../conceptnet/relation.txt'
ConceptnNet = '../conceptnet/concept.en.csv'

def load_resources():

    def jsonKeys2int(x):
        if isinstance(x, dict):
            return {int(k): v for k, v in x.items()}
        return x

    if os.path.exists(config.data_concept_dict+'concept_info.json'):
        with open(config.data_concept_dict+'concept_info.json', 'r') as f:
            data = f.readlines()
            concept2id = json.loads(data[0])
            relation2id = json.loads(data[1])
            id2relation = json.loads(data[2], object_hook=jsonKeys2int)
            id2concept = json.loads(data[3], object_hook=jsonKeys2int)

    else:
        concept2id, id2concept, relation2id, id2relation = dict(), dict(), dict(), dict()
        wnl = WordNetLemmatizer()
        vocab = pickle.load(open(config.data_vocab, 'rb'))
        vocab = list(vocab.word2index.keys())
        lemma = [wnl.lemmatize(w) for w in vocab if w not in vocab]
        vocab.extend(lemma)

        with open(concept_vocab, 'r', encoding='UTF8') as f:
            for w in f.readlines():
                if w.strip() not in vocab:
                    continue
                concept2id[w.strip()] = len(concept2id)
                id2concept[len(id2concept)] = w.strip()
        print('finish loading concept2id')

        with open(concept_rel, 'r', encoding='UTF8') as f:
            for rel in f.readlines():
                relation2id[rel.strip()] = len(relation2id)
                id2relation[len(id2relation)] = rel.strip()
        print('finish loading relation2id')
        with open(config.data_concept_dict+'concept_info.json', 'w') as f:
            json.dump(concept2id, f)
            f.write('\n')
            json.dump(relation2id, f)
            f.write('\n')
            json.dump(id2relation, f)
            f.write('\n')
            json.dump(id2concept, f)
            f.write('\n')
    return concept2id, relation2id, id2relation, id2concept

def save_net():
    concept2id, relation2id, id2relation, id2concept = load_resources()
    concept = concept2id.keys()
    graph = MultiDiGraph()
    conceptnet = pd.read_csv(ConceptnNet, sep='\t', encoding='UTF8', header=None) #['start', 'end', 'rel', 'weight']
    conceptnet = conceptnet[conceptnet[0].isin(concept) & conceptnet[1].isin(concept)]
    conceptnet = conceptnet[~conceptnet[0].isin(nltk_stopwords)]
    conceptnet = conceptnet[~conceptnet[1].isin(nltk_stopwords)]
    conceptnet = conceptnet[conceptnet[0]!=conceptnet[1]]
    conceptnet = conceptnet[conceptnet[2]!='HasContext']
    for idx, line in tqdm(conceptnet.iterrows(), desc="saving to graph"):
        start, end, rel, weight = line
        start = concept2id[str(start)]
        end = concept2id[str(end)]
        rel = relation2id[str(rel)]
        weight = float(weight)

        graph.add_edge(start, end, rel=rel, weight=weight)
        graph.add_edge(end, start, rel=rel+len(relation2id), weight=weight)

    write_gpickle(graph, config.conceptnet_graph)
