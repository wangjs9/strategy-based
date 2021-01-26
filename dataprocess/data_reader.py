import config
import numpy as np
import nltk
import os, pickle
from dataprocess.data2clause import read_json


blacklist = {'goes', 'aside', 'heres', 's2', 'getting', 'par', 'ref', 'cg', 'up', 'after', 'aren', 'promptly', 'ran', 'er', 'shall', 'everybody', 'V', 'did', 'p3', 'recent', 'able', 'tt', 'unlikely', 'moreover', 'amoungst', 'please', 'lf', 'yr', 'elsewhere', 'nn', 'inward', 'is', 'ab', 'cc', 'ar', 'nay', 'x3', 'mg', 'pc', 'et', 'looks', 'op', 'example', 'what', 'rh', 'ac', 'by', 'hereafter', 'io', 'always', 'rt', 'whereby', 'certain', 'vol', 'actually', 'hu', 'below', 'besides', 'dc', 'g', 'quickly', 'ff', 'let', 'xs', 'only', 'ob', 'entirely', 'ni', 'somebody', 'into', 'ten', 'before', 'ju', 'cm', 'bs', 's', 'sorry', 'eu', 'whomever', 'doing', 'ev', 'given', 'ib', 'otherwise', 'ever', 'rm', 'hasn', 'as', 'lets', 'tb', 'whither', 'pp', 'vols', 'whenever', 'sufficiently', 'ones', 'thence', 'added', 'ia', 'ej', 'thoughh', 'well-b', 'p1', 'affected', 'doesn', 'mainly', 'whim', 'gone', 'd', 'indicates', 'non', 'ibid', 'ru', 'http', 'must', 'cz', 'fn', 'through', 'qu', 'sz', 'anybody', 'll', 'across', '0o', 'd2', 'dx', 'available', 'over', 'ix', 'yet', 'give', 'p', 'arent', 'though', 'e3', 'td', 'i8', 'pj', 'made', 'les', 'y2', 'f2', 'o', 'find', 'showed', 'specified', 'ry', 'fix', 'ad', 'nearly', 'when', 'c3', 'downwards', 'hopefully', 'los', 'bi', 'eight', 'at', 'reasonably', 'resulted', 'taking', 'tx', 'used', 'next', 'specifically', 'u', 'di', 'whereas', 'latterly', 'alone', 'le', 'ri', 'likely', 'aj', 'kj', 'pe', 'against', 'end', 'following', 'pu', 'have', 'their', 'ur', 'b1', 'nine', 'regarding', 'normally', 'gets', 'nc', 'hello', 'te', 'indicate', 'cr', 'suggest', 'were', 'substantially', 'oi', 'uk', 'ny', 'insofar', 'described', 'with', 'fifth', 'inner', 'make', 'i6', 'although', 'vs', 'has', 'n2', 'while', 'less', 'section', 'far', 'c2', 'ah', 'was', 'hh', 'hs', 'ti', 'un', 'fl', 'most', 'w', 'stop', 'ap', 'pm', 'rv', 'os', 'own', 'where', 'hundred', 'okay', 'hasnt', 'the', '6b', 'meanwhile', 'ok', 'hid', 'some', 'ow', 'beforehand', 'could_be', 'almost', 'accordance', 'consequently', 'rc', 'no', 'briefly', 'pn', 'seemed', 'who', 'em', 'yours', 'awfully', 'fc', 'pi', 'forth', 'lest', 'ma', 'just', 'unless', 'got', 'wasnt', 'behind', 'sec', 'ml', 'b3', 'having', 'x', 'k', 'mr', 'taken', 'sure', 'oj', 'J', 'pt', 'kg', 'they', 'sc', 'not', 'such', 'whereupon', 'rr', 'b2', 'its', 'last', 'eg', 'bottom', 'b', 'name', 'a1', 'gr', 'obtained', 'sd', 'afterwards', 'page', 'O', 'G', 'predominantly', 'hy', 'now', 'werent', 'ex', 'ca', 'P', 'wheres', 'trying', 'until', 'ms', 'uj', 'refs', 'unto', 'oo', 'dj', 'per', 'r2', 'strongly', 'somewhat', 'isn', 'merely', 'upon', 'proud', 'dp', 'ii', 'tq', 'zi', 'probably', 'right', 'bd', 'neither', 'og', 'saying', 'ds', 'ih', 'sy', 'four', 'done', 'ls', 'towards', 'hereby', 'here', 'why', 'pd', 'rs', 'x2', 'y', 'few', 'amongst', 'beginnings', 'somethan', 'hed', 'follows', 'want_to', 'ga', 'j', 'howbeit', 'r', 'whom', 'xl', 'cl', 'much', 'pq', 'approximately', 't3', 'twice', 'mug', 'keep', 'old', 'usefully', 'are', 'tm', 'beyond', 'mu', 'vj', 'seeming', 'pl', 'eq', 'thereof', 'tell', 'detail', 'm2', 'sixty', 'particular', 'rd', 'whence', 'hardly', 'so', 'said', 'instead', 'mt', 'shan', 'fy', 'comes', 'little', 'three', 'z', 'zz', 'weren', 'specifying', 'py', 'id', 'course', 'won', 'u201d', 'affecting', 'show', 'date', 'i2', 'vt', 'et-al', 'primarily', 'particularly', 'thanx', 'happens', 'com', 'fo', 'how', 'dk', 're', 'whats', 'way', 'cs', 'fire', 'various', 'want', 'forty', 'hence', 'plus', 'whether', 'hadn', 'hither', 'ic', 'es', 'past', 'say', 'between', 'line', 'gives', 'five', 'cq', 'anyhow', 'auth', 'nonetheless', 'whoever', 'took', 'away', 'another', 'la', 'co', 'oz', 'a4', 'nowhere', 'rather', 'yt', 'seem', 'bl', 'om', 'bp', 'sq', 'youre', 'rq', 'but', 'fs', 'toward', 'cit', 'anyone', 'pages', 'ci', 'previously', 'second', 'couldnt', 'xx', 'tl', 'volumtype', 'tv', 'your', 'six', 'still', 'mn', 'shed', 'eo', 'act', 'anyways', 'im', 'outside', 'theres', 'to', 'can', 'dl', 'our', 'ra', 'results', 'both', 'somewhere', 'quite', 'be', 'twenty', 'tp', 'ought', 'wont', 'we', 'resulting', 'often', 'somehow', 'along', 'don', 'i3', 'interest', 'nj', 'out', 'top', 'tip', 'best', 'then', 'itd', 'anyway', 'everyone', 'cv', 'ge', 'ou', 'hes', 'might', 'C', 'sometime', 'ys', 'since', 'unfortunately', 'definitely', 'du', 'lately', 'herein', 'ol', 'xt', 'yj', 'makes', 'ph', 'thickv', 'regards', 'K', 'x1', 'xk', 'qv', 'au', 'pr', 'shows', 'throug', 'concerning', 'nt', 'bj', 'est', 'hj', 'us', 'iz', 'cj', 'qj', 'ay', 'p2', 'that', 'above', 'adj', 'N', 'zero', 'od', 'seen', 'sm', 'usefulness', 'this', 'novel', 'E', 'like', 'considering', 'if', 'km', 'da', 'ng', 'wouldn', 'nor', 'nl', 'm', '3b', 'i', 'thus', '0s', 'a', 'el', 'especially', 'i4', 'these', 'tried', 'haven', 'jr', 've', 'S', 'H', 'vo', 'widely', 'my', 'ao', 'cry', 'jt', 'rn', 'yes', 'ig', 'related', 'thanks', 'thered', 'help', 'az', 'near', 'among', 'cf', 'became', 'xf', 'arise', 'eighty', 'saw', 'asking', 'do', 'nr', 'there', 'possibly', 'ei', 'again', 'ed', 'n', 'allow', 'too', 'largely', 'st', 'themselves', 'whos', 'ain', 'cx', 'accordingly', 'thereupon', 'a2', 'pf', 'gl', 'everything', 'fr', 'third', 'successfully', 'brief', 'ox', 'sup', 'indeed', 'bt', 'obviously', 'each', 'amount', 'iq', 'for', 'ord', 'clearly', 'followed', 'whose', 'sometimes', 'beside', 'gave', 'th', 'ue', 'um', 'ey', 'sometimes_people', 'couldn', 'down', 'gi', 'rf', 'further', 'dt', 'went', 'does', 'nd', 'ip', 'lo', 'ep', 'mill', 'everywhere', 'side', 'someone', 'bk', 'vq', 'says', 'placed', 'fa', 'useful', 'uo', 'ct', 'former', 'gy', 'of', 'tc', 'fj', 'c', 'ot', 'may', 'therere', 'wasn', 'na', 'would', 'found', 'xn', 'en', 'ns', 'liked', 'noone', 'ne', 'F', 'readily', 'shes', 'thereby', 'v', 'ltd', 'due', 'hi', 'pagecount', 'well', 'call', 'meantime', 'wherein', 'D', 'however', 'fill', 'wherever', 'apparently', 'kept', 'a3', 'enough', 'iy', 'Z', 'sr', 'mrs', 'whod', 'ending', 'lt', 'new', 'indicated', 'consider', 'ef', 'df', 'ps', 'thereto', 'namely', 'together', 'ignored', 'try', 'abst', 'really', 'dr', 'respectively', 'full', 'came', 'ke', 'run', 'va', 'ea', 'formerly', 'Y', 'in', 'around', 'ro', 'during', 'fify', 'nobody', 'welcome', 'back', 'sl', 'lb', 'than', 'ask', 'truly', 'pk', 'furthermore', 'fu', 'announce', 'or', 'seems', 'those', 'whereafter', 'thou', 'gs', 'thousand', 'cannot', 'i7', 'from', 'front', 'them', 'whole', 'under', 'theyd', 'didn', 'describe', 'al', 'sub', 'cp', 'bx', 'presumably', 'part', 'many', 'bu', 'therefore', 'cd', 'owing', 'soon', 'anymore', 'ij', 'come', 'cy', 'giving', 'ie', 'oh', '3a', 'omitted', 'perhaps', 'rl', 'you', 't', 'cant', 'me', 'ec', 'ninety', '6o', 'least', 'tn', 'ho', 'ln', 'unlike', 'obtain', 'already', 'had', 'bc', 'ko', 'index', 'l2', '-PRON-', 'nevertheless', 'and', 'sn', 'A', 'maybe', 'recently', 'mine', 'gj', 'which', 'af', 'research-articl', 'therein', 'f', 'q', 'tf', 'go', 'con', 'h', 'going', 'que', 'according', 'any', 'ax', 'eleven', 'lj', 'ut', 'xj', 'noted', 'youd', 'Q', 'wi', 'yl', 'either', 'every', 'M', 'something', 'se', 'U', 'viz', 'it', 'ft', 'despite', 'thank', 'mustn', 'about', 'ups', 'il', 'hereupon', 'exactly', 'poorly', 'vd', 'av', 't1', 'greetings', 'inasmuch', 'til', 'T', 'ae', 'mightn', 't2', 'theirs', 'miss', 'sf', 'h2', 'sj', 'twelve', 'biol', 'fi', 'ir', 'without', 'on', 'sa', 'been', 'ee', 'information', 'tries', 'seven', '3d', 'tj', 'W', 'once', 'ts', 'also', 'nos', 'overall', 'iv', 'dy', 'except', 'sp', 'wo', 'cn', 'get', 'regardless', 'very', 'thru', 'theyre', 'two', 'else', 'L', 'c1', 'ch', 'etc', 'fifteen', 'all', 'mo', 'looking', 'gotten', 'think', 'bn', 'throughout', 'wa', 'oq', 'shown', 'an', 'ce', 'more', 'thats', 'ba', 'apart', 'vu', 'tr', 'br', 'hr', 'si', 'thoroughly', 'thorough', 'ag', 'B', 'xi', 'one', 'showns', 'oc', 'h3', 'de', 'e2', 'immediately', 'keeps', 'xo', 'within', 'bill', 'he', 'pas', 'ui', 'anywhere', 'using', 'wouldnt', 'could', 'take', 'rj', 'X', 'wonder', 'latter', 'aw', 'jj', 'am', 'mostly', 'ss', 'several', 'tends', 'wed', 'secondly', 'usually', 'www', 'sincere', 'appreciate', 'dd', 'oa', 'thereafter', 'even', 'million', 'R', 'inc', 'lc', 'js', 'lr', 'specify', 'certainly', 'relatively', 'edu', 'allows', 'e', 'l', 'necessarily', 'thin', 'none', 'put', 'slightly', 'whatever', 'po', 'via', 'cu', 'onto', 'sent', 'provides', 'later', 'move', 'off', 'xv', 'look'}

class Lang:
    """
    create a new word dictionary, including 3 dictionaries:
    1) word to index;
    2) word and its count;
    3) index to word;
    and one counter indicating the number of words.
    """

    def __init__(self, init_index2word):
        """
        :param init_index2word: a dictionary containing (id: token) pairs
        """
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def read_langs(vocab):

    # >>>>>>>>>> historical utterances >>>>>>>>>> #
    train_dialog = np.load(config.data_npy_dict+'sys_dialog_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> next expected utterance from the bot >>>>>>>>>> #
    train_target = np.load(config.data_npy_dict+'sys_target_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> emotions of the conversation >>>>>>>>>> #
    train_emotion = np.load(config.data_npy_dict+'sys_emotion_texts.train.npy', allow_pickle=True)
    # >>>>>>>>>> action of the conversation >>>>>>>>>> #
    train_strategy = np.load(config.data_npy_dict+'sys_cause_type.train.npy', allow_pickle=True)
    # >>>>>>>>>> cause of the conversation >>>>>>>>>> #
    train_cause = np.load(config.data_npy_dict+'sys_cause_texts.train.npy', allow_pickle=True)
    train_causality = read_json(config.data_npy_dict+'sys_causality_triple.train.npy')

    dev_dialog = np.load(config.data_npy_dict+'sys_dialog_texts.valid.npy', allow_pickle=True)
    dev_target = np.load(config.data_npy_dict+'sys_target_texts.valid.npy', allow_pickle=True)
    dev_emotion = np.load(config.data_npy_dict+'sys_emotion_texts.valid.npy', allow_pickle=True)
    dev_strategy = np.load(config.data_npy_dict + 'sys_cause_type.valid.npy', allow_pickle=True)
    dev_cause = np.load(config.data_npy_dict + 'sys_cause_texts.valid.npy', allow_pickle=True)
    dev_causality = read_json(config.data_npy_dict+'sys_causality_triple.valid.npy')

    data_train = {'dialog': [], 'target': [], 'emotion': [], 'cause': [], 'strategy': [], 'graphs': []}
    data_dev = {'dialog': [], 'target': [], 'emotion': [], 'cause': [], 'strategy': [], 'graphs': []}


    drop_index = np.random.choice(np.argwhere(train_emotion[np.where(train_emotion=="neutral")]).flatten(), 2000)
    # all_index = np.argwhere(train_emotion).flatten()
    # train_index = np.delete(all_index, drop_index)

    for idx, emotion in enumerate(train_emotion):
        if idx in drop_index:
            continue
        data_train['emotion'].append(emotion)

    for idx, dialog in enumerate(train_dialog):
        u_lists = []
        for utts in dialog:
            u_list = nltk.word_tokenize(utts)
            vocab.index_words(u_list)
            u_lists.append(u_list)
        if idx not in drop_index:
            data_train['dialog'].append(u_lists)

    for idx, target in enumerate(train_target):
        u_list = nltk.word_tokenize(target)
        vocab.index_words(u_list)
        if idx not in drop_index:
            data_train['target'].append(u_list)

    for idx, strategy in enumerate(train_strategy):
        if idx in drop_index:
            continue
        data_train['strategy'].append(strategy)

    for idx, cause in enumerate(train_cause):
        if idx in drop_index:
            continue
        u_lists = []
        for utts in cause:
            u_list = nltk.word_tokenize(utts)
            u_lists.append(u_list)
        data_train['cause'].append(u_lists)

    for idx, graph in enumerate(train_causality):
        if idx in drop_index:
            continue
        data_train['graphs'].append(graph)

    assert len(data_train['dialog']) == len(data_train['target']) == len(data_train['emotion']) == len(
        data_train['cause']) == len(data_train['graphs']) == len(data_train['strategy'])

    # drop_index = np.random.choice(np.argwhere(train_emotion[np.where(train_emotion == "neutral")]).flatten(), 200)
    # all_index = np.argwhere(train_emotion).flatten()
    # train_index = np.delete(all_index, drop_index)

    for emotion in dev_emotion:
        data_dev['emotion'].append(emotion)

    for dialog in dev_dialog:
        u_lists = []
        for utts in dialog:
            u_list = nltk.word_tokenize(utts)
            vocab.index_words(u_list)
            u_lists.append(u_list)
        data_dev['dialog'].append(u_lists)

    for target in dev_target:
        u_list = nltk.word_tokenize(target)
        vocab.index_words(u_list)
        data_dev['target'].append(u_list)

    for strategy in dev_strategy:
        data_dev['strategy'].append(strategy)

    for cause in dev_cause:
        u_lists = []
        for utts in cause:
            u_list = nltk.word_tokenize(utts)
            u_lists.append(u_list)
        data_dev['cause'].append(u_lists)

    data_dev['graphs'] = dev_causality

    assert len(data_dev['dialog']) == len(data_dev['target']) == len(data_dev['emotion']) == len(
        data_dev['cause']) == len(data_dev['graphs']) == len(data_dev['strategy'])

    return data_train, data_dev, vocab

def load_dataset():
    if os.path.exists(config.data_path):
        print("LOADING {}".format(config.dataset))
        with open(config.data_path, "rb") as f:
            [data_tra, data_val, vocab] = pickle.load(f)
            # >>>>>>>>>> dictionaries >>>>>>>>>> #
    else:
        print("Building dataset...")
        data_tra, data_val, vocab = read_langs(vocab=Lang(
            {config.UNK_idx: "UNK", config.PAD_idx: "PAD", config.EOS_idx: "EOS", config.SOS_idx: "SOS",
             config.USR_idx: "USR", config.SYS_idx: "SYS", config.CLS_idx: "CLS", config.SEP_idx: "SEP"}))
        with open(config.data_path, "wb") as f:
            pickle.dump([data_tra, data_val, vocab], f)
            print("Saved PICKLE")
    for i in range(3):
        print('[emotion]:', data_tra['emotion'][i])
        print('[dialog]:', [' '.join(u) for u in data_tra['dialog'][i]])
        print('[target]:', ' '.join([ele for ele in data_tra['target'][i]]))
        print(" ")
    return data_tra, data_val, vocab
