import json, os, pickle
from collections import Counter
from tqdm import tqdm
import pandas as pd
import torch
import torch.utils.data as data
import nltk
import numpy as np
from copy import deepcopy
from models.detector import Emotion, cause, strategy, config
from dataprocess.data_reader import Lang
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report

# {'sadness', 'happiness', 'fear', 'surprise', 'disgust', 'anger', 'neutral'}

class Dataset(data.Dataset):
    def __init__(self, data, vocab):
        self.vocab = vocab
        self.data = data
        self.emo_map = {'sadness': 1, 'happiness': 2, 'fear': 3, 'surprise': 4, 'disgust': 5, 'anger': 6, 'neutral': 0}

    def __len__(self):
        return len(self.data['target'])

    def __getitem__(self, index):
        item = {}
        item["dialog_text"] = self.data["dialog"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotions"][index]
        item["cause_id"] = self.data["cause"][index]
        item["cause_span"] = self.data["cause_span"][index]
        item["type"] = self.data["type"][index]
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)

        return item

    def preprocess_emo(self, emotion, emo_map):
        program = [0] * len(emo_map)
        program[emo_map[emotion]] = 1
        # >>>>>>>>>> one hot mode and label mode >>>>>>>>>> #
        return program, emo_map[emotion]

def save_data(data_path, save_path):
    # dialog, target; emotion, casuse_id, cause_span, type
    # casue_map = {"no-context": 0, "inter-personal": 1, "self-contagion": 2, "hybrid": 3, "latent": 4}
    emotion_labels = []
    data = json.load(open(data_path, 'r'))
    dialog, targets, context_emotion, last_emotions, emotions, cause_id, cause_span, type = [], [], [], [], [], [], [], []
    for key, uttr in data.items():
        uttr = uttr[0]
        context = []
        majar_emotion = []
        last_emo = None
        for idx, sent in enumerate(uttr):
            targets.append(sent["utterance"])
            context.append(sent["utterance"])
            dialog.append(context)
            last_emotions.append(last_emo)
            last_emo = sent["emotion"]
            emotions.append(sent["emotion"])
            if sent["emotion"] != "neutral":
                majar_emotion.append(sent["emotion"])
            emotion_labels.append(sent["emotion"])
            cause_id.append(sent.get("expanded emotion cause evidence", []))
            cause_span.append(sent.get("expanded emotion cause span", []))
            type.append(sent.get("type", []))

        count = Counter(majar_emotion)
        majar = count.most_common(1)
        for i in range(len(uttr)):
            context_emotion.append(majar)

    assert len(dialog) == len(targets) == len(context_emotion) == len(last_emotions) == len(
        emotions) == len(cause_id) == len(cause_span) == len(type)

    print(set(emotion_labels))

    data = {'dialog': dialog, 'target': targets, 'con_emo': context_emotion, 'last_emo': last_emotions,
            'emotions': emotions, 'cause': cause_id, 'span': cause_span, 'type': type}

    with open(save_path, 'w') as f:
        f.write(json.dumps(data))

def read_langs(vocab):
    train_data = json.load(open(config.train_cause, 'r'))
    valid_data = json.load(open(config.valid_cause, 'r'))

    data_train = {'context': [], 'target': [], 'emotion': []}

    for dialog in train_data['dialog']:
        u_lists = []
        for uttr in dialog:
            u = nltk.word_tokenize(uttr.lower())
            u_lists.extend(u)
            vocab.index_words([x for x in u if x.isalph()])
        data_train['context'].append(u_lists)

    for target in train_data['target']:
        u_lists = nltk.word_tokenize(target.lower())
        vocab.index_words([x for x in u_lists if x.isalph()])
        data_train['target'].append(u_lists)

    for emotion in train_data['emotions']:
        data_train['emotion'].append(emotion)

    assert len(data_train['context']) == len(data_train['target']) == len(data_train['emotion'])

    data_valid = {'context': [], 'target': [], 'emotion': []}

    for dialog in valid_data['dialog']:
        u_lists = []
        for uttr in dialog:
            u = nltk.word_tokenize(uttr.lower())
            u_lists.extend(u)
            vocab.index_words([x for x in u if x.isalph()])
        data_valid['context'].append(u_lists)

    for target in valid_data['target']:
        u_lists = nltk.word_tokenize(target.lower())
        vocab.index_words([x for x in u_lists if x.isalph()])
        data_valid['target'].append(u_lists)

    for emotion in valid_data['emotions']:
        data_valid['emotion'].append(emotion)

    assert len(data_valid['context']) == len(data_valid['target']) == len(data_valid['emotion'])

    return data_train, data_valid, vocab

def collate_fn(data):
    d = {}
    d['context'] = data['context'].to(config.device)
    d['target'] = data['target'].to(config.device)
    d['emotion'] = data['emotion'].to(config.device)
    return d

def evaluate(model, data, predict=False):
    pbar = tqdm(enumerate(data), total=len(data))
    loss = []
    acc = []
    predics = []
    for j, batch in pbar:
        loss, acc, predic = model(batch, predict)
        loss.append(loss)
        acc.append(acc)
        predics.extend(predic.cpu())

    loss = np.mean(loss)
    acc = np.mean(acc)

    print("EVAL\tLoss\tACC:")
    print("{}\t{:.4f}\t{:.4f}".format("", loss, acc))

    return

def load_data(batch_size=32):
    if os.path.exists(config.emotion_data_path):
        print("LOADING data")
        with open(config.emotion_data_path, "rb") as f:
            data_train, data_valid, vocab = pickle.load(f)
    else:
        print("BUILDING data...")
        data_train, data_valid, vocab = read_langs(vocab=Lang({}))
        with open(config.emotion_data_path, "wb") as f:
            pickle.dump([data_train, data_valid, vocab], f)
            print("Saved PICKLE")

    dataset_train = Dataset(data_train, vocab)
    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)

    dataset_valid = Dataset(data_valid, vocab)
    data_loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid,
                                                  batch_size=batch_size,
                                                  shuffle=True, collate_fn=collate_fn)

    return data_loader_train, data_loader_valid, vocab

def train(emotion=False, cause=False, strategy=False):

    if not os.path.exists(config.train_cause):
        save_data(config.train_path, config.train_cause)
    if not os.path.exists(config.valid_cause):
        save_data(config.valid_path, config.valid_cause)
    if emotion and cause and strategy:
        raise ValueError('One of `emotion`, `cause` and `strategy` must be True.')
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if emotion:
        train_data, valid_data, vocab = load_data(config.bz)
        print(vocab)
        with open('../data/dailydialog/vocabulary.pkl', 'wb') as f:
            pickle.dump(vocab, f)
        exit()
        model = Emotion(vocab).to(config.device)
        model = model.train()
        best_acc = model.current_acc
        init_iter = model.iter
        check_iter = 500
        patient = 0
        for n_iter in tqdm(range(10000)):
            loss, acc, predic = model(next(train_data))

            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                loss_val, acc_val, _ = evaluate(model, valid_data)

                if acc_val <= best_acc:
                    best_ppl = acc_val
                    patient = 0
                    model.save_model(best_ppl, n_iter + init_iter, acc_val, loss_val)
                    weights_best = deepcopy(model.state_dict())

                else:
                    patient += 1

                if patient > 2:
                    break
        model.load_state_dict({name: weights_best[name] for name in weights_best})
        model.eval()
        _, _, predic = evaluate(model, valid_data)

def load_csv():
    if os.path.exists(config.emotion_data_path):
        print("LOADING data")
        with open(config.emotion_data_path, "rb") as f:
            data_train, data_valid, _ = pickle.load(f)
    else:
        print("BUILDING data...")
        data_train, data_valid, vocab = read_langs(vocab=Lang({}))
        with open(config.emotion_data_path, "wb") as f:
            pickle.dump([data_train, data_valid, vocab], f)
            print("Saved PICKLE")

    d = {'text': [], 'labels': []}
    for idx, target in enumerate(data_train['target']):
        text = '{} <SEP> {}'.format(' '.join(target), ' '.join(data_train['context'][idx]))
        emotion = 1 if config.emo_map[data_train['emotion'][idx]] else 0
        d['text'].append(text)
        d['labels'].append(emotion)

    x_train = pd.DataFrame(data=d)

    d = {'text': [], 'labels': []}
    for idx, target in enumerate(data_valid['target']):
        text = '{} <SEP> {}'.format(' '.join(target), ' '.join(data_valid['context'][idx]))
        emotion = 1 if config.emo_map[data_valid['emotion'][idx]] else 0
        d['text'].append(text)
        d['labels'].append(emotion)


    x_valid = pd.DataFrame(data=d)


    return x_train, x_valid


def train_detector():

    train_args = {
        'fp16': False,
        'overwrite_output_dir': True,
        'max_seq_length': 512,
        'learning_rate': config.lr,
        'sliding_window': False,
        'output_dir': config.emotion_save,
        'best_model_dir': config.emotion_save + 'best_model/',
        'evaluate_during_training': True,
        'evaluate_during_training_steps': int(493/config.bz)+1,
        'save_eval_checkpoints': False,
        'save_model_every_epoch': False,
        'save_steps': 50000,
        'train_batch_size': config.bz,
        'num_train_epochs': config.epochs
    }

    x_train, x_valid = load_csv()
    # x_train = pd.read_csv('../RECCON-main/data/subtask2/fold1/dailydialog_classification_train_with_context.csv')
    cls_model = ClassificationModel('roberta', 'roberta-base', args=train_args, cuda_device=0)
    cls_model.train_model(x_train, eval_df=x_valid)
    cls_model = ClassificationModel('roberta', 'roberta-base', args=train_args, cuda_device=0)
    result, model_outputs, wrong_predictions = cls_model.eval_model(x_valid)
    preds = np.argmax(model_outputs, 1)
    labels = x_valid['labels']

    r = str(classification_report(labels, preds, digits=4))
    print(r)

    rf = open(config.emotion_save + 'results_classification.txt', 'a')
    rf.write(r + '\n' + '-' * 54 + '\n')
    rf.close()

train(True)