UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6
SEP_idx = 7

import argparse, os, torch

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dailydialog', help='`empathetic_dialogues`, `cornell_movie` or `dailydialog`')
parser.add_argument("--model", type=str, default="multihop", help='`trs`, `strategy` or `multihop`')

parser.add_argument('--data_dict', type=str, default='../data/', help='name of data dictionary')
parser.add_argument('--glove_path', type=str, default='../glove.6B/glove.6B.300d.txt', help='name of vocab embedding file')
parser.add_argument("--emb_path", type=str, default="utils/embedding.txt")
parser.add_argument("--emb_file", type=str, default="../glove.6B/glove.6B.{}d.txt")
parser.add_argument('--save_path', type=str, default='', help='name of save file')
parser.add_argument('--triple_dict', type=str, default='', help='name of conceptnet graph file')

parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--cause_hidden_dim", type=int, default=128)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument('--bz', type=int, default=16, help='the size of batch')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--schedule', type=int, default=500, help='schedule step')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay rate')
parser.add_argument('--gs', type=int, default=10000, help='total number of global steps')
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--strategy_num", type=int, default=3)
parser.add_argument("--cvae_latent", type=int, default=56)
parser.add_argument("--cvae_size", type=int, default=512)
parser.add_argument("--max_length", type=int, default=56)

parser.add_argument("--weight_sharing", type=bool, default=True)
parser.add_argument("--label_smoothing", type=bool, default=True)
parser.add_argument("--noam", type=bool, default=True)
parser.add_argument("--universal", type=bool, default=True)
parser.add_argument("--emo_multitask", type=bool, default=True)
parser.add_argument("--act", type=bool, default=True)
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--pretrain_emb", type=bool, default=True)
parser.add_argument("--test", type=bool, default=False)


## transformer
parser.add_argument("--hop", type=int, default=6)
parser.add_argument("--heads", type=int, default=8)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

## graph
parser.add_argument("--hop_num", type=int, default=2)
parser.add_argument("--max_mem_size", type=int, default=400)
parser.add_argument("--max_triple_size", type=int, default=1000)

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


arg = parser.parse_args()
print_opts(arg)
model = arg.model

# >>>>>>>>>> hyperparameters >>>>>>>>>> #
emb_dim = arg.emb_dim
hidden_dim = arg.hidden_dim
cause_hidden_dim = arg.cause_hidden_dim
strategy_num = arg.strategy_num
### transformer
hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter
hop_num = arg.hop_num
pretrain_emb = arg.pretrain_emb
label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight
emo_multitask = arg.emo_multitask
cvae_latent = arg.cvae_latent
cvae_size = arg.cvae_size
max_length = arg.max_length

# >>>>>>>>>> data path >>>>>>>>>> #
dataset = arg.dataset
data_dict = arg.data_dict + dataset
data_npy_dict = data_dict + '/npyFile/'
data_concept_dict = data_dict + '/concept/'
if not os.path.exists(data_npy_dict):
    os.mkdir(data_npy_dict)
data_path = data_dict + '/dataset_preproc.p'
data_vocab = data_dict + '/vocabulary.pkl'
embed_path = arg.glove_path
emb_path = arg.emb_path
emb_file = arg.emb_file.format(str(emb_dim))
save_path = arg.save_path if arg.save_path else './save/{}/'.format(model)
posembedding_path = arg.data_dict + '/embedding_pos.txt'
conceptnet_graph = data_concept_dict + 'concept.graph'

# >>>>>>>>>> training parameters >>>>>>>>>> #
bz = arg.bz
lr = arg.lr
schedule = arg.schedule
weight_decay = arg.weight_decay
test = arg.test
beam_size = arg.beam_size
device = 'cuda' if torch.cuda.is_available() else 'cpu'



