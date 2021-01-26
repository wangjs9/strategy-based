import torch, os
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from models.common_layer import share_embedding, EncoderLayer, LayerNorm, _gen_bias_mask, NoamOpt, _gen_timing_signal


class Config():
    def __init__(self):
        super(Config, self).__init__()
        data_dict = '../data/Emotion_stimulus/'
        self.train_path = data_dict + 'dailydialog_train.json'
        self.valid_path = data_dict + 'dailydialog_valid.json'
        self.train_cause = '../data/dailydialog/cause.train.json'
        self.valid_cause = '../data/dailydialog/cause.valid.json'
        self.blacklist = {'goes', 'aside', 'heres', 's2', 'getting', 'par', 'ref', 'cg', 'up', 'after', 'aren', 'promptly', 'ran', 'er', 'shall', 'everybody', 'V', 'did', 'p3', 'recent', 'able', 'tt', 'unlikely', 'moreover', 'amoungst', 'please', 'lf', 'yr', 'elsewhere', 'nn', 'inward', 'is', 'ab', 'cc', 'ar', 'nay', 'x3', 'mg', 'pc', 'et', 'looks', 'op', 'example', 'what', 'rh', 'ac', 'by', 'hereafter', 'io', 'always', 'rt', 'whereby', 'certain', 'vol', 'actually', 'hu', 'below', 'besides', 'dc', 'g', 'quickly', 'ff', 'let', 'xs', 'only', 'ob', 'entirely', 'ni', 'somebody', 'into', 'ten', 'before', 'ju', 'cm', 'bs', 's', 'sorry', 'eu', 'whomever', 'doing', 'ev', 'given', 'ib', 'otherwise', 'ever', 'rm', 'hasn', 'as', 'lets', 'tb', 'whither', 'pp', 'vols', 'whenever', 'sufficiently', 'ones', 'thence', 'added', 'ia', 'ej', 'thoughh', 'well-b', 'p1', 'affected', 'doesn', 'mainly', 'whim', 'gone', 'd', 'indicates', 'non', 'ibid', 'ru', 'http', 'must', 'cz', 'fn', 'through', 'qu', 'sz', 'anybody', 'll', 'across', '0o', 'd2', 'dx', 'available', 'over', 'ix', 'yet', 'give', 'p', 'arent', 'though', 'e3', 'td', 'i8', 'pj', 'made', 'les', 'y2', 'f2', 'o', 'find', 'showed', 'specified', 'ry', 'fix', 'ad', 'nearly', 'when', 'c3', 'downwards', 'hopefully', 'los', 'bi', 'eight', 'at', 'reasonably', 'resulted', 'taking', 'tx', 'used', 'next', 'specifically', 'u', 'di', 'whereas', 'latterly', 'alone', 'le', 'ri', 'likely', 'aj', 'kj', 'pe', 'against', 'end', 'following', 'pu', 'have', 'their', 'ur', 'b1', 'nine', 'regarding', 'normally', 'gets', 'nc', 'hello', 'te', 'indicate', 'cr', 'suggest', 'were', 'substantially', 'oi', 'uk', 'ny', 'insofar', 'described', 'with', 'fifth', 'inner', 'make', 'i6', 'although', 'vs', 'has', 'n2', 'while', 'less', 'section', 'far', 'c2', 'ah', 'was', 'hh', 'hs', 'ti', 'un', 'fl', 'most', 'w', 'stop', 'ap', 'pm', 'rv', 'os', 'own', 'where', 'hundred', 'okay', 'hasnt', 'the', '6b', 'meanwhile', 'ok', 'hid', 'some', 'ow', 'beforehand', 'could_be', 'almost', 'accordance', 'consequently', 'rc', 'no', 'briefly', 'pn', 'seemed', 'who', 'em', 'yours', 'awfully', 'fc', 'pi', 'forth', 'lest', 'ma', 'just', 'unless', 'got', 'wasnt', 'behind', 'sec', 'ml', 'b3', 'having', 'x', 'k', 'mr', 'taken', 'sure', 'oj', 'J', 'pt', 'kg', 'they', 'sc', 'not', 'such', 'whereupon', 'rr', 'b2', 'its', 'last', 'eg', 'bottom', 'b', 'name', 'a1', 'gr', 'obtained', 'sd', 'afterwards', 'page', 'O', 'G', 'predominantly', 'hy', 'now', 'werent', 'ex', 'ca', 'P', 'wheres', 'trying', 'until', 'ms', 'uj', 'refs', 'unto', 'oo', 'dj', 'per', 'r2', 'strongly', 'somewhat', 'isn', 'merely', 'upon', 'proud', 'dp', 'ii', 'tq', 'zi', 'probably', 'right', 'bd', 'neither', 'og', 'saying', 'ds', 'ih', 'sy', 'four', 'done', 'ls', 'towards', 'hereby', 'here', 'why', 'pd', 'rs', 'x2', 'y', 'few', 'amongst', 'beginnings', 'somethan', 'hed', 'follows', 'want_to', 'ga', 'j', 'howbeit', 'r', 'whom', 'xl', 'cl', 'much', 'pq', 'approximately', 't3', 'twice', 'mug', 'keep', 'old', 'usefully', 'are', 'tm', 'beyond', 'mu', 'vj', 'seeming', 'pl', 'eq', 'thereof', 'tell', 'detail', 'm2', 'sixty', 'particular', 'rd', 'whence', 'hardly', 'so', 'said', 'instead', 'mt', 'shan', 'fy', 'comes', 'little', 'three', 'z', 'zz', 'weren', 'specifying', 'py', 'id', 'course', 'won', 'u201d', 'affecting', 'show', 'date', 'i2', 'vt', 'et-al', 'primarily', 'particularly', 'thanx', 'happens', 'com', 'fo', 'how', 'dk', 're', 'whats', 'way', 'cs', 'fire', 'various', 'want', 'forty', 'hence', 'plus', 'whether', 'hadn', 'hither', 'ic', 'es', 'past', 'say', 'between', 'line', 'gives', 'five', 'cq', 'anyhow', 'auth', 'nonetheless', 'whoever', 'took', 'away', 'another', 'la', 'co', 'oz', 'a4', 'nowhere', 'rather', 'yt', 'seem', 'bl', 'om', 'bp', 'sq', 'youre', 'rq', 'but', 'fs', 'toward', 'cit', 'anyone', 'pages', 'ci', 'previously', 'second', 'couldnt', 'xx', 'tl', 'volumtype', 'tv', 'your', 'six', 'still', 'mn', 'shed', 'eo', 'act', 'anyways', 'im', 'outside', 'theres', 'to', 'can', 'dl', 'our', 'ra', 'results', 'both', 'somewhere', 'quite', 'be', 'twenty', 'tp', 'ought', 'wont', 'we', 'resulting', 'often', 'somehow', 'along', 'don', 'i3', 'interest', 'nj', 'out', 'top', 'tip', 'best', 'then', 'itd', 'anyway', 'everyone', 'cv', 'ge', 'ou', 'hes', 'might', 'C', 'sometime', 'ys', 'since', 'unfortunately', 'definitely', 'du', 'lately', 'herein', 'ol', 'xt', 'yj', 'makes', 'ph', 'thickv', 'regards', 'K', 'x1', 'xk', 'qv', 'au', 'pr', 'shows', 'throug', 'concerning', 'nt', 'bj', 'est', 'hj', 'us', 'iz', 'cj', 'qj', 'ay', 'p2', 'that', 'above', 'adj', 'N', 'zero', 'od', 'seen', 'sm', 'usefulness', 'this', 'novel', 'E', 'like', 'considering', 'if', 'km', 'da', 'ng', 'wouldn', 'nor', 'nl', 'm', '3b', 'i', 'thus', '0s', 'a', 'el', 'especially', 'i4', 'these', 'tried', 'haven', 'jr', 've', 'S', 'H', 'vo', 'widely', 'my', 'ao', 'cry', 'jt', 'rn', 'yes', 'ig', 'related', 'thanks', 'thered', 'help', 'az', 'near', 'among', 'cf', 'became', 'xf', 'arise', 'eighty', 'saw', 'asking', 'do', 'nr', 'there', 'possibly', 'ei', 'again', 'ed', 'n', 'allow', 'too', 'largely', 'st', 'themselves', 'whos', 'ain', 'cx', 'accordingly', 'thereupon', 'a2', 'pf', 'gl', 'everything', 'fr', 'third', 'successfully', 'brief', 'ox', 'sup', 'indeed', 'bt', 'obviously', 'each', 'amount', 'iq', 'for', 'ord', 'clearly', 'followed', 'whose', 'sometimes', 'beside', 'gave', 'th', 'ue', 'um', 'ey', 'sometimes_people', 'couldn', 'down', 'gi', 'rf', 'further', 'dt', 'went', 'does', 'nd', 'ip', 'lo', 'ep', 'mill', 'everywhere', 'side', 'someone', 'bk', 'vq', 'says', 'placed', 'fa', 'useful', 'uo', 'ct', 'former', 'gy', 'of', 'tc', 'fj', 'c', 'ot', 'may', 'therere', 'wasn', 'na', 'would', 'found', 'xn', 'en', 'ns', 'liked', 'noone', 'ne', 'F', 'readily', 'shes', 'thereby', 'v', 'ltd', 'due', 'hi', 'pagecount', 'well', 'call', 'meantime', 'wherein', 'D', 'however', 'fill', 'wherever', 'apparently', 'kept', 'a3', 'enough', 'iy', 'Z', 'sr', 'mrs', 'whod', 'ending', 'lt', 'new', 'indicated', 'consider', 'ef', 'df', 'ps', 'thereto', 'namely', 'together', 'ignored', 'try', 'abst', 'really', 'dr', 'respectively', 'full', 'came', 'ke', 'run', 'va', 'ea', 'formerly', 'Y', 'in', 'around', 'ro', 'during', 'fify', 'nobody', 'welcome', 'back', 'sl', 'lb', 'than', 'ask', 'truly', 'pk', 'furthermore', 'fu', 'announce', 'or', 'seems', 'those', 'whereafter', 'thou', 'gs', 'thousand', 'cannot', 'i7', 'from', 'front', 'them', 'whole', 'under', 'theyd', 'didn', 'describe', 'al', 'sub', 'cp', 'bx', 'presumably', 'part', 'many', 'bu', 'therefore', 'cd', 'owing', 'soon', 'anymore', 'ij', 'come', 'cy', 'giving', 'ie', 'oh', '3a', 'omitted', 'perhaps', 'rl', 'you', 't', 'cant', 'me', 'ec', 'ninety', '6o', 'least', 'tn', 'ho', 'ln', 'unlike', 'obtain', 'already', 'had', 'bc', 'ko', 'index', 'l2', '-PRON-', 'nevertheless', 'and', 'sn', 'A', 'maybe', 'recently', 'mine', 'gj', 'which', 'af', 'research-articl', 'therein', 'f', 'q', 'tf', 'go', 'con', 'h', 'going', 'que', 'according', 'any', 'ax', 'eleven', 'lj', 'ut', 'xj', 'noted', 'youd', 'Q', 'wi', 'yl', 'either', 'every', 'M', 'something', 'se', 'U', 'viz', 'it', 'ft', 'despite', 'thank', 'mustn', 'about', 'ups', 'il', 'hereupon', 'exactly', 'poorly', 'vd', 'av', 't1', 'greetings', 'inasmuch', 'til', 'T', 'ae', 'mightn', 't2', 'theirs', 'miss', 'sf', 'h2', 'sj', 'twelve', 'biol', 'fi', 'ir', 'without', 'on', 'sa', 'been', 'ee', 'information', 'tries', 'seven', '3d', 'tj', 'W', 'once', 'ts', 'also', 'nos', 'overall', 'iv', 'dy', 'except', 'sp', 'wo', 'cn', 'get', 'regardless', 'very', 'thru', 'theyre', 'two', 'else', 'L', 'c1', 'ch', 'etc', 'fifteen', 'all', 'mo', 'looking', 'gotten', 'think', 'bn', 'throughout', 'wa', 'oq', 'shown', 'an', 'ce', 'more', 'thats', 'ba', 'apart', 'vu', 'tr', 'br', 'hr', 'si', 'thoroughly', 'thorough', 'ag', 'B', 'xi', 'one', 'showns', 'oc', 'h3', 'de', 'e2', 'immediately', 'keeps', 'xo', 'within', 'bill', 'he', 'pas', 'ui', 'anywhere', 'using', 'wouldnt', 'could', 'take', 'rj', 'X', 'wonder', 'latter', 'aw', 'jj', 'am', 'mostly', 'ss', 'several', 'tends', 'wed', 'secondly', 'usually', 'www', 'sincere', 'appreciate', 'dd', 'oa', 'thereafter', 'even', 'million', 'R', 'inc', 'lc', 'js', 'lr', 'specify', 'certainly', 'relatively', 'edu', 'allows', 'e', 'l', 'necessarily', 'thin', 'none', 'put', 'slightly', 'whatever', 'po', 'via', 'cu', 'onto', 'sent', 'provides', 'later', 'move', 'off', 'xv', 'look'}
        if not os.path.exists('./detectors'):
            os.mkdir('./detectors')
        if not os.path.exists('./detectors/data'):
            os.mkdir('./detectors/data')
        self.emotion_save = './detectors/emotion/'
        if not os.path.exists(self.emotion_save):
            os.mkdir(self.emotion_save)
        self.emotion_data_path = './detectors/data/emotion.p'

        self.emb_dim = 300
        self.emb_file = '../glove.6B/glove.6B.{}d.txt'.format(self.emb_dim)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.emo_map = {'neutral': 0, 'happy': 1, 'happiness': 1, 'disgust': 2, 'surprise': 3, 'surprised': 3,
                        'fear': 4, 'anger': 5, 'angry': 5, 'excited': 6, 'sadness': 7, 'sad': 7}

        self.hidden_dim = 300
        self.hop = 6
        self.heads = 8
        self.depth = 40
        self.filter = 50
        self.latent_size = 57
        self.input_layer_size = [300, 256]
        self.output_layer_size = [256, 300]
        self.bz = 8
        self.lr = 1e-5
        self.epochs = 12



config = Config()

class ACT_basic(nn.Module):
    """

    """
    def __init__(self,hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size,1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False):
        # init_hdd
        ## [B, S]
        halting_probability = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        remainders = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0],inputs.shape[1]).cuda()
        ## [B, S, HDD]
        previous_state = torch.zeros_like(inputs).cuda()

        step = 0
        # for l in range(self.num_layers):
        while( ((halting_probability<self.threshold) & (n_updates < max_hop)).byte().any()):
            # as long as there is a True value, the loop continues
            # Add timing signal
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)

            p = self.sigma(self.p(state)).squeeze(-1) # (1, 1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if(decoding):
                state, _, attention_weight = fn((state,encoder_output,[]))
            else:
                # apply transformation on the state
                state = fn(state)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            if(decoding):
                if(step==0):  previous_att_weight = torch.zeros_like(attention_weight).cuda()      ## [B, S, src_size]
                previous_att_weight = ((attention_weight * update_weights.unsqueeze(-1)) + (previous_att_weight * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop
            ## to save a line I assigned to previous_state so in the next
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1

        if(decoding):
            return previous_state, previous_att_weight, (remainders,n_updates)
        else:
            return previous_state, (remainders,n_updates)

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length) if use_mask else None,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)

        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)


        self.act_fn = ACT_basic(hidden_size)
        self.remainders = None
        self.n_updates = None

    def forward(self, inputs, mask):
        # Add input dropout
        x = self.input_dropout(inputs) # (batch_size, seq_len, embed_dim)
        # Project to hidden size
        x = self.embedding_proj(x) # (batch_size, seq_len, hidden_size)

        if self.universal:
            x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal,
                                                               self.position_signal, self.num_layers)
            y = self.layer_norm(x)

        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)

        return y

class CVAE(nn.Module):
    def __init__(self, input_layer_size, latent_size, output_layer_size):
        super(CVAE, self).__init__()
        self.enc_MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(input_layer_size[:-1], input_layer_size[1:])):
            self.enc_MLP.add_module(
                name="L{:d}".formate(i), module=nn.Linear(in_size, out_size)
            )
            self.enc_MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.means = nn.Linear(input_layer_size[-1], latent_size)
        self.log_var = nn.Linear(input_layer_size[-1], latent_size)

        self.dec_MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([latent_size]+output_layer_size[:-1], output_layer_size)):
            self.dec_MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            if i < len(output_layer_size)-1:
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, x, c):
        x = self.enc_MLP(torch.cat(x, c), dim=-1)
        means = self.means(x)
        log_var = self.log_var(x)
        z = self.sample(means, log_var)
        recon_x = self.dec_MLP(z, c)
        KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())
        return recon_x, KLD

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps

class Emotion(nn.Module):
    def __init__(self, vocab, model_file_path=None, load_optim=False):
        super(Emotion, self).__init__()
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding = share_embedding(self.vocab, True)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=True)

        self.context_ecoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=True)

        self.linear = nn.Linear(config.emb_dim, 2)
        # self.linear = nn.Linear(config.emb_dim, len(config.emo_map))

        optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=config.weight_decay, betas=(0.9, 0.98),
                                     eps=1e-9)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.schedule * i for i in range(4)],
                                                         gamma=0.1)
        self.scheduler = NoamOpt(config.hidden_dim, 1, 8000, optimizer, scheduler)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.iter = state['iter']
            self.current_acc = state['current_acc']
            self.embedding.load_state_dict(state['embedding_dict'])
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.context_ecoder.load_state_dict(state['context_encoder_state_dict'])
            if load_optim:
                try:
                    self.scheduler.load_state_dict(state['optimizer'])
                except AttributeError:
                    pass
        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, iter, acc, loss):
        self.iter = iter
        state = {
            'iter': self.iter,
            'embedding_dict': self.embedding.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'context_encoder_state_dict': self.context_encoder.state_dict(),
            'optimizer': self.scheduler.state_dict(),
            'current_acc': acc
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}'.format(
            iter, acc))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def forward(self, batch, predict=False):
        context, target, emotion = batch
        emotion = (emotion > 0).to(int)
        self.scheduler.optimizer.zero_grad()
        context = self.context_ecoder(context)
        context = torch.sum(context, dim=-2, keepdim=True)
        target = torch.cat(context, self.encoder(target), dim=-2)
        target = torch.sum(target, dim=-2)
        pre_logit = torch.sigmoid(self.linear(target))
        logit = torch.softmax(pre_logit, dim=-1)

        predic = torch.max(logit.data, 1)[1]
        loss = -1
        if not predict:
            loss = F.cross_entropy(logit, emotion)
            loss.backward()
            self.scheduler.step()
            train_acc = metrics.accuracy_score(emotion.cpu(), predic.cpu())

        return loss, train_acc, predic


class cause(nn.Module):
    def __init__(self):
        super(cause, self).__init__()

    def forward(self):
        pass

class strategy(nn.Module):
    def __init__(self):
        super(strategy, self).__init__()

    def forward(self):
        pass
