import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import Counter
# import nltk
# nltk.download('wordnet')
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

dailydialog_path = '../data/Emotion_stimulus/dailydialog_train.json'
dailydialog_npyFile = '../data/dailydialog/npyFile'
if not os.path.exists(dailydialog_npyFile):
    os.mkdir(dailydialog_npyFile)

import config

word_pairs = {"it's": "it is", "don't": "do not", "doesn't": "does not", "didn't": "did not", "you'd": "you would",
                  "you're": "you are", "you'll": "you will", "i'm": "i am", "they're": "they are", "that's": "that is",
                  "what's": "what is", "couldn't": "could not", "i've": "i have", "we've": "we have", "can't": "cannot",
                  "i'd": "i would", "i'd": "i would", "aren't": "are not", "isn't": "is not", "wasn't": "was not",
                  "weren't": "were not", "won't": "will not", "there's": "there is", "there're": "there are"}

def clean(sentence, word_pairs=word_pairs):
    sentence = sentence.lower()
    for k, v in word_pairs.items():
        sentence = sentence.replace(k, v)
    return sentence

def empathetic(DataType):
    if DataType not in ['train', 'valid', 'test']:
        raise ValueError('`DataType` must be `train`, `valid` or `test`')
    print('Loading data...')
    conversation = pd.read_csv(config.data_dict +'{}.csv'.format(DataType), encoding='unicode_escape', header=0, index_col=False)
    situations = list()
    emotions = list()
    history = list()
    targets = list()



    context = []
    for idx, row in conversation.iterrows():
        conId, sentId, label, situ, _, utterance, _ = row
        sentId = int(sentId)

        if sentId == 1:
            context = []
        if sentId % 2 == 1:
            context.append(clean(utterance, word_pairs))
        else:
            emotions.append(label)
            situations.append(clean(situ, word_pairs))
            history.append(context.copy())
            bot = clean(utterance, word_pairs)
            targets.append(bot)
            context.append(bot)

        assert len(emotions) == len(situations) == len(history) == len(targets)

    ScaleTypes = ['min', 'sys']
    print('Saving data for sys...')
    np.save(config.data_npy_dict+'{}_situation_texts.{}.npy'.format(ScaleTypes[1], DataType), situations)
    np.save(config.data_npy_dict+'{}_dialog_texts.{}.npy'.format(ScaleTypes[1], DataType), history)
    np.save(config.data_npy_dict+'{}_target_texts.{}.npy'.format(ScaleTypes[1], DataType), targets)
    np.save(config.data_npy_dict+'{}_emotion_texts.{}.npy'.format(ScaleTypes[1], DataType), emotions)

    print('Saving data for min...')
    if DataType == 'train':
        num = 200
    else:
        num = 20
    np.save(config.data_npy_dict+'{}_situation_texts.{}.npy'.format(ScaleTypes[0], DataType), situations[:num])
    np.save(config.data_npy_dict+'{}_dialog_texts.{}.npy'.format(ScaleTypes[0], DataType), history[:num])
    np.save(config.data_npy_dict+'{}_target_texts.{}.npy'.format(ScaleTypes[0], DataType), targets[:num])
    np.save(config.data_npy_dict+'{}_emotion_texts.{}.npy'.format(ScaleTypes[0], DataType), emotions[:num])

def dailydialog():
    from dataprocess.find_path import process
    print('Loading data...')
    conversation = json.load(open(dailydialog_path, 'rb'))
    dialog, target, conv_emotion, emotion, cause, src_concept, cas_concept, cas_type, cause_text = [], [], [], [], [], [], [], [], []
    require_pos = {'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'POS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    wnl = WordNetLemmatizer()
    for idx, turns in tqdm(conversation.items(), total=len(conversation)):
        context, majar_emo = [], []
        turns = turns[0]
        for i, turn in enumerate(turns):
            if i > 0:
                dialog.append(context.copy())
                if majar_emo != []:
                    conv_emotion.append(Counter(majar_emo).most_common(1))
                else:
                    conv_emotion.append('neutal')
                emotion.append(emo)
                target.append(clean(turn['utterance']))
                concept = []
                for word, pos in pos_tag(word_tokenize(uttr)):
                    if pos in require_pos:
                        concept.append(word)
                        concept.append(wnl.lemmatize(word))
                src_concept.append(list(set(concept)))
                concept = []
                for word, pos in pos_tag(word_tokenize(' '.join(cause_span))):
                    if pos in require_pos:
                        concept.append(word)
                        concept.append(wnl.lemmatize(word))
                cause_text.append(cause_span)
                cas_concept.append(list(set(concept)))
                cas_type.append(type)
                cause.append([int(id)-1 for id in cause_id])

            uttr = clean(turn['utterance'])
            emo = turn['emotion']
            type = turn.get('type', [])
            cause_id = turn.get('expanded emotion cause evidence', [])
            cause_id = [t if t != 'b' else -1 for t in cause_id]
            cause_span = turn.get('expanded emotion cause span', [])
            cause_span = [clean(c) for c in cause_span]
            if emo != 'neutal':
                majar_emo.append(emo)
            context.append(uttr)

    assert len(dialog) == len(target) == len(conv_emotion) == len(emotion) ==len(
        cause) == len(src_concept) == len(cas_concept) == len(cause_text) == len(cas_type)

    # np.save(dailydialog_npyFile + '/sys_dialog_texts.valid.npy', dialog)
    # np.save(dailydialog_npyFile + '/sys_target_texts.valid.npy', target)
    # np.save(dailydialog_npyFile + '/sys_conv_emotion_texts.valid.npy', conv_emotion)
    # np.save(dailydialog_npyFile + '/sys_emotion_texts.valid.npy', emotion)
    # np.save(dailydialog_npyFile + '/sys_cause_texts.valid.npy', cause_text)
    # np.save(dailydialog_npyFile + '/sys_cause_type.valid.npy', cas_type)
    # np.save(dailydialog_npyFile + '/tmp_cas_concept.valid.npy', cas_concept)
    # np.save(dailydialog_npyFile + '/tmp_src_concept.valid.npy', src_concept)
    # process(dailydialog_npyFile + '/sys_causality_path.valid.npy', cas_concept, src_concept, emotion, mode='valid',
    #         max_B=50)
    # directed_triple(dailydialog_npyFile + '/sys_causality_path.valid.npy',
    #                 dailydialog_npyFile + '/sys_causality_triple.valid.npy')

    np.save(dailydialog_npyFile+'/sys_dialog_texts.train.npy', dialog)
    np.save(dailydialog_npyFile+'/sys_target_texts.train.npy', target)
    np.save(dailydialog_npyFile+'/sys_conv_emotion_texts.train.npy', conv_emotion)
    np.save(dailydialog_npyFile+'/sys_emotion_texts.train.npy', emotion)
    np.save(dailydialog_npyFile + '/sys_cause_texts.train.npy', cause_text)
    np.save(dailydialog_npyFile + '/sys_cause_type.train.npy', cas_type)
    np.save(dailydialog_npyFile+'/tmp_cas_concept.train.npy', cas_concept)
    np.save(dailydialog_npyFile+'/tmp_src_concept.train.npy', src_concept)
    process(dailydialog_npyFile+'/sys_causality_path.train.npy', cas_concept, src_concept, emotion, mode='train', max_B=50)
    directed_triple(dailydialog_npyFile + '/sys_causality_path.train.npy', dailydialog_npyFile + '/sys_causality_triple.train.npy')

def read_json(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def directed_triple(data_path, save_path, max_concepts=400, max_triple=1000):
    # read data from data_path
    data = read_json(data_path)

    _data = []
    max_len = 0
    max_neighbors = 5
    for e in tqdm(data):
        triple_dict = {}
        triples = e['triples']
        concepts = e['concepts']
        labels = e['labels']
        distances = e['distances']
        for t in triples:
            head, tail = t[0], t[-1]
            head_id = concepts.index(head)
            tail_id = concepts.index(tail)
            if distances[head_id] <= distances[tail_id]:
                if t[-1] not in triple_dict:
                    triple_dict[t[-1]] = [t]
                else:
                    if len(triple_dict[t[-1]]) < max_neighbors:
                        triple_dict[t[-1]].append(t)

        results = []
        for l, c in zip(labels, concepts):
            if l == 1:
                results.append(c)
        # print(results)

        causes = []
        for d, c in zip(distances, concepts):
            if d == 0:
                causes.append(c)

        shortest_paths = []
        for result in results:
            shortest_paths.extend(bfs(result, triple_dict, causes))

        ground_truth_concepts = []
        ground_truth_triples = []
        for path in shortest_paths:
            for i, n in enumerate(path[:-1]):
                ground_truth_triples.append((n, path[i + 1]))
                ground_truth_concepts.append(n)
                ground_truth_concepts.append(path[i + 1])
        ground_truth_concepts = list(set(ground_truth_concepts))

        ground_truth_triples_set = set(ground_truth_triples)

        _triples, triple_labels = [], []
        for e1, e2 in ground_truth_triples_set:
            for t in triple_dict[e1]:
                if e2 in t:
                    _triples.append(t)
                    triple_labels.append(1)

        for k, v in triple_dict.items():
            for t in v:
                if t in _triples:
                    continue
                _triples.append(t)
                # if (t[-1], t[0]) in ground_truth_triples_set:
                #     triple_labels.append(1)
                # else:
                triple_labels.append(0)

        if len(concepts) > max_concepts:
            rest_concepts = list(set(concepts) - set(ground_truth_concepts))
            rest_len = max_concepts-len(ground_truth_concepts)
            _concepts = ground_truth_concepts + rest_concepts[:rest_len]
            e['concepts'] = _concepts
            e['distances'] = [distances[concepts.index(c)] for c in _concepts]
            e['labels'] = [distances[labels.index(c)] for c in _concepts]
            concepts = _concepts
        # _triples = _triples[:max_triples]
        # triple_labels = triple_labels[:max_triples]

        heads = []
        tails = []
        relations = []
        for triple in _triples:
            try:
                h = concepts.index(triple[0])
                t = concepts.index(triple[-1])
                heads.append(h)
                tails.append(t)
                relations.append(triple[1])
                if len(heads) == max_triple:
                    break
            except ValueError:
                continue

        max_len = max(max_len, len(_triples))
        e['relations'] = relations
        e['head_ids'] = heads
        e['tail_ids'] = tails
        e['triple_labels'] = triple_labels[:max_triple]
        e.pop('triples')

        _data.append(e)
        # break

    with open(save_path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')

    return _data

def bfs(start, triple_dict, source):
    paths = [[[start]]]
    shortest_paths = []
    count = 0
    while True:
        last_paths = paths[-1]
        new_paths = []
        for path in last_paths:
            if triple_dict.get(path[-1], False):
                triples = triple_dict[path[-1]]
                for triple in triples:
                    new_paths.append(path + [triple[0]])

        for path in new_paths:
            if path[-1] in source:
                shortest_paths.append(path)

        if count == 2:
            break
        paths.append(new_paths)
        count += 1

    return shortest_paths

# def temp():
    # from dataprocess.find_path import process
    # cas_concept = np.load(dailydialog_npyFile+'/tmp_cas_concept.train.npy', allow_pickle=True)
    # src_concept = np.load(dailydialog_npyFile+'/tmp_src_concept.train.npy', allow_pickle=True)
    # emotion = np.load(dailydialog_npyFile+'/sys_emotion_texts.train.npy', allow_pickle=True)
    # process(dailydialog_npyFile + '/sys_causality_path.train.npy', cas_concept, src_concept, emotion, mode='train',
    #         max_B=50)
    # directed_triple(dailydialog_npyFile + '/sys_causality_path.train.npy', dailydialog_npyFile + '/sys_causality_triple.train.npy')

