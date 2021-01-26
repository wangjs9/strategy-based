### TAKEN FROM https://github.com/kolloldas/torchnlp
import torch
import torch.nn as nn

import numpy as np
import math
from models.common_layer import share_embedding, LabelSmoothing, NoamOpt, get_input_from_batch, get_output_from_batch
from systems.common import Encoder, Decoder, Generator
import config
import pprint

pp = pprint.PrettyPrinter(indent=1)
import os
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class CVAE(nn.Module):
    def __init__(self, input_layer_size, latent_size, output_layer_size, immediate_size=56):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.context = nn.Linear(config.hidden_dim, immediate_size, bias=False)

        self.enc_MLP = nn.Sequential()
        input_layer_size[0] += config.strategy_num+immediate_size
        for i, (in_size, out_size) in enumerate(zip(input_layer_size[:-1], input_layer_size[1:])):
            self.enc_MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            self.enc_MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.means = nn.Linear(input_layer_size[-1], latent_size)
        self.log_var = nn.Linear(input_layer_size[-1], latent_size)

        self.dec_MLP = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([latent_size+config.strategy_num+immediate_size]+output_layer_size[:-1], output_layer_size)):
            self.dec_MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            if i < len(output_layer_size)-1:
                self.dec_MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.dec_MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, x, c, context):
        context = self.context(context)
        x = self.enc_MLP(torch.cat((x, c, context), axis=-1))
        means = self.means(x)
        log_var = self.log_var(x)
        z = self.sample(means, log_var)
        z = torch.cat((z, c, context), axis=-1)
        recon_x = self.dec_MLP(z)
        KLD = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp())
        return recon_x, KLD

    def sample(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + std * eps

    def inference(self, c, context):
        context = self.context(context)
        z = torch.randn([c.size(0), 1, self.latent_size]).repeat(1, c.size(1), 1).to(config.device)
        z = torch.cat((z, c, context), axis=-1)
        recon_x = self.dec_MLP(z)
        return recon_x


class Strategy(nn.Module):

    def __init__(self, vocab, decoder_number, model_file_path=None, load_optim=False):
        """
        vocab: a Lang type data, which is defined in data_reader.py
        decoder_number: the number of classes
        """
        super(Strategy, self).__init__()
        self.iter = 0
        self.current_loss = 1000
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)

        ## decoders

        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads, total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, max_length=config.max_length)
        self.cvae = CVAE([config.hidden_dim, config.cvae_size], config.cvae_latent, [config.cvae_size, config.hidden_dim])
        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if (config.noam):
            optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=config.weight_decay, betas=(0.9, 0.98),
                                         eps=1e-9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[config.schedule * i for i in range(4)],
                                                             gamma=0.1)
            self.scheduler = NoamOpt(config.hidden_dim, 1, 8000, optimizer, scheduler)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                  milestones=[config.schedule * i for i in range(4)],
                                                                  gamma=0.1)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.iter = state['iter']
            self.current_loss = state['current_loss']
            self.embedding.load_state_dict(state['embedding_dict'])
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.cvae.load_state_dict(state['cvae_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict'])
            if load_optim:
                try:
                    self.scheduler.load_state_dict(state['optimizer'])
                except AttributeError:
                    pass

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter):
        self.iter = iter
        state = {
            'iter': iter,
            'embedding_dict': self.embedding.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'cvae_dict': self.cvae.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'optimizer': self.scheduler.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir,
            'model_{}_{:.4f}'.format(iter, running_avg_ppl))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, train=True):
        enc_batch, cause_batch = get_input_from_batch(batch)
        dec_batch = get_output_from_batch(batch)

        if (config.noam):
            self.scheduler.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src) # (batch_size, seq_len, hidden_size)

        # Decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(config.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1) # make the first token of sentence be SOS

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), encoder_outputs, (mask_src, mask_trg))

        pre_logit, KLD_loss = self.cvae(self.embedding(dec_batch), batch['strategy'].unsqueeze(1).repeat(1, pre_logit.size(1), 1), pre_logit)

        # shape: pre_logit --> (batch_size, seq_len, hidden_size)
        ## compute output dist
        logit = self.generator(pre_logit)

        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)) + KLD_loss

        loss_bce_program, program_acc = 0, 0
        # multi-task
        if config.emo_multitask:
            # add the loss function of label prediction
            q_h = encoder_outputs[:, 0] # the first token of the sentence CLS, shape: (batch_size, 1, hidden_size)
            logit_prob = self.decoder_key(q_h).to('cuda') # (batch_size, 1, decoder_num)
            loss += nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).cuda())
            loss_bce_program = nn.CrossEntropyLoss()(logit_prob, torch.LongTensor(batch['program_label']).cuda()).item()
            pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
            program_acc = accuracy_score(batch["program_label"], pred_program)

        if (config.label_smoothing):
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1)).item()

        if (train):
            loss.backward()
            self.scheduler.step()

        if (config.label_smoothing):
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_bce_program, program_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), loss_bce_program, program_acc

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch, cause_batch = get_input_from_batch(batch)

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(config.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):

            out, attn_dist = self.decoder(self.embedding(ys), encoder_outputs, (mask_src, mask_trg))

            out = self.cvae.inference(batch['strategy'].unsqueeze(1).repeat(1, out.size(1), 1), out)
            prob = self.generator(out)
            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).to(config.device)], dim=1).to(config.device)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>':
                    break
                else:
                    st += e + ' '
            sent.append(st)
        return sent

