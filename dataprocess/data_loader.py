import torch
import torch.utils.data as data
import logging
import pprint
pp = pprint.PrettyPrinter(indent=1)
import config
from models.common_layer import write_config
from dataprocess.data_reader import load_dataset

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data
        self.emo_map = {'neutral': 0, 'happy': 1, 'happiness': 1, 'disgust': 2, 'surprise': 3, 'surprised': 3, 'fear': 4, 'anger': 5, 'angry': 5, 'excited': 6, 'sadness': 7, 'sad': 7}
        self.emo_num = 8
        self.strategy = {'no-context': 0, 'inter-person': 0, 'hybrid': 0, 'self-contagion': 0, 'latent': 1}
        self.strategy_num = 3

    def __len__(self):
        return len(self.data["target"])

    def __getitem__(self, index, max_length=1024):
        """Returns one data pair (source and target)."""
        item = {}
        item["dialog_text"] = self.data["dialog"][index][-6:]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]
        item["cause_text"] = self.data["cause"][index]
        item["strategy"] = self.data["strategy"][index]

        item["context"], item["context_mask"] = self.preprocess(item["dialog_text"])
        item["cause_batch"] = self.preprocess(' <SEP> '.join([' '.join(lst) for lst in item["cause_text"]]).split(), clause=True)
        item["context_text"] = [ele for lst in item["dialog_text"] for ele in lst]

        item["target"] = self.preprocess(item["target_text"]+["EOS"], clause=True, max_length=config.max_length)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)
        item["strategy"], item["strategy_label"] = self.preprocess_strategy(item["strategy"])


        item["graph_concept_ids"], item["graph_concept_label"], item["graph_distances"], item["graph_relation"], \
        item["graph_head"], item["graph_tail"], item["graph_triple_label"], item["vocab_map"], item["map_mask"] = self.preprocess_graph(self.data["graphs"][index])

        return item

    def preprocess(self, text, clause=False, max_length=1024):
        """Converts words to ids."""
        if clause:
            sequence = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                        text]
            sequence = sequence[:max_length]
            return torch.LongTensor(sequence)

        else:
            X_dial = []
            X_mask = []
            for i, sentences in enumerate(text):

                for j, sentence in enumerate(sentences):
                    X_dial += [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                               sentence]
                    # >>>>>>>>>> spk: whether this sen is from a user or bot >>>>>>>>>> #
                    spk = self.vocab.word2index["USR"] if i % 2 == len(text) % 2 else self.vocab.word2index["SYS"]
                    X_mask += [spk for _ in range(len(sentence))]

            assert len(X_dial) == len(X_mask)
            if len(X_dial) > max_length-1:
                X_dial = X_dial[1-max_length:]
                X_mask = X_mask[1-max_length:]

            X_dial = [config.CLS_idx] + X_dial
            X_mask = [config.CLS_idx] + X_mask

            return torch.LongTensor(X_dial), torch.LongTensor(X_mask)
            # >>>>>>>>>> context, context mask >>>>>>>>>> #

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * self.emo_num
        program[emo_map[emotion]] = 1
        # >>>>>>>>>> one hot mode and label mode >>>>>>>>>> #
        return program, emo_map[emotion]

    def preprocess_graph(self, graph):
        map_mask = [0 for i in range(self.vocab.n_words)]

        concepts = graph["concepts"]
        vocab_map = []
        for w, idx in self.vocab.word2index.items():
            try:
                pos = concepts.index(w)
                vocab_map.append(pos)
                map_mask[idx] = 1
            except ValueError:
                vocab_map.append(0)
        relation = torch.LongTensor([r[0] for r in graph["relations"]])
        concepts = [self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx for word in
                concepts]
        concept_id = torch.LongTensor(concepts)
        distance = torch.LongTensor(graph["distances"])
        head = torch.LongTensor(graph["head_ids"])
        tail = torch.LongTensor(graph["tail_ids"])
        triple_label = torch.LongTensor(graph["triple_labels"])
        concept_label = torch.LongTensor(graph["labels"])

        vocab_map = torch.LongTensor(vocab_map)

        return concept_id, concept_label, distance, relation, head, tail, triple_label, vocab_map, torch.LongTensor(map_mask)

    def preprocess_strategy(self, strategy):
        program = [0] * self.strategy_num
        label = 0
        if 'latent' in strategy:
            label = 1
        elif strategy == []:
            label = 1

        program[label] = 1
        return program, label

def collate_fn(data):
    def merge(sequences, pad=1):
        """
        padded_seqs: use 1 to pad the rest
        lengths: the lengths of seq in sequences
        """
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.ones(len(sequences), max(lengths)).long()  ## padding index 1
        padded_seqs *= pad
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    data.sort(key=lambda x: len(x["context"]), reverse=True)  ## sort by source seq
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## graph
    concept_ids, concept_num = merge(item_info["graph_concept_ids"])
    distances, _ = merge(item_info["graph_distances"], pad=0)
    relations, triple_num = merge(item_info["graph_relation"])
    heads, _ = merge(item_info["graph_head"])
    tails, _ = merge(item_info["graph_tail"])
    triple_label, _ = merge(item_info["graph_triple_label"], pad=-1)
    concept_label, _ = merge(item_info["graph_concept_label"], pad=-1)
    vocab_map, _ = merge(item_info["vocab_map"], pad=0)
    map_mask, _ = merge(item_info["map_mask"], pad=0)


    ## input
    input_batch, input_lengths = merge(item_info['context'])
    mask_input, mask_input_lengths = merge(item_info['context_mask'])  # use idx for bot or user to mask the seq
    batch_size = input_batch.size(0)
    ## clause
    cause_batch, _ = merge(item_info['cause_batch'])

    ## Target
    target_batch, target_lengths = merge(item_info['target'])

    d = {}
    d["input_batch"] = input_batch.to(config.device)
    d["input_lengths"] = torch.LongTensor(input_lengths)  # mask_input_lengths equals input_lengths
    d["mask_input"] = mask_input.to(config.device)
    ##cause
    d["cause_batch"] = cause_batch.to(config.device)
    ##target
    d["target_batch"] = target_batch.to(config.device)
    d["target_lengths"] = torch.LongTensor(target_lengths)
    ##program
    d["target_program"] = item_info['emotion']  # one hot format
    d["program_label"] = item_info['emotion_label']
    d["strategy"] = torch.LongTensor(item_info['strategy']).to(config.device)
    d["strategy_label"] = torch.LongTensor(item_info['strategy_label']).to(config.device).reshape(batch_size, -1)
    ##graph
    d["concept_ids"] = concept_ids.to(config.device)
    d["concept_num"] = concept_num
    d["distances"] = distances.to(config.device)
    d["relations"] = relations.to(config.device)
    d["triple_num"] = triple_num
    d["heads"] = heads.to(config.device)
    d["tails"] = tails.to(config.device)
    d["concept_label"] = concept_label.to(config.device)
    d["triple_label"] = triple_label.to(config.device)
    d["vocab_map"] = vocab_map
    d["map_mask"] = map_mask
    ##text
    d["input_txt"] = item_info['context_text']
    d["target_txt"] = item_info['target_text']
    d["program_txt"] = item_info['emotion_text']

    return d

def prepare_data_seq(batch_size=32):
    """
    :return:
    vocab: vocabulary including index2word, and word2index
    len(dataset_train.emo_map)
    """
    pairs_tra, pairs_val, vocab = load_dataset()

    logging.info("Vocab  {} ".format(vocab.n_words))

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = torch.utils.data.DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = torch.utils.data.DataLoader(dataset=dataset_valid,
                              batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)

    data_loader_tst = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                  batch_size=1,
                                                  shuffle=True, collate_fn=collate_fn)

    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)