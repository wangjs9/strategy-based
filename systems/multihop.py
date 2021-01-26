import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from models.common_layer import share_embedding, LabelSmoothing, NoamOpt, \
    get_input_from_batch, get_graph_from_batch, get_output_from_batch
from systems.common import Encoder, Decoder, Generator
import config
import pprint
pp = pprint.PrettyPrinter(indent=1)
import os
from torch_scatter import scatter_max, scatter_mean, scatter_add
from sklearn.metrics import accuracy_score

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class G_Cause(nn.Module):
    def __init__(self, embedding):
        super(G_Cause, self).__init__()
        self.embedding = embedding
        self.device = config.device
        self.hop_num = config.hop_num
        self.relation_embd = nn.Embedding(47*2, config.emb_dim)
        self.W_s = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim, bias=False) for _ in range(config.hop_num)])
        self.W_n = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim, bias=False) for _ in range(config.hop_num)])
        self.W_r = nn.ModuleList(
            [nn.Linear(config.emb_dim, config.emb_dim, bias=False) for _ in range(config.hop_num)])
        self.triple_linear = nn.Linear(config.emb_dim * 3, config.emb_dim, bias=False)
        self.gate_linear = nn.Linear(config.emb_dim, 1)

    def multi_layer_gcn(self, concept_hidden, relation_hidden, head, tail, triple_label,
                             layer_number=2):
        for i in range(layer_number):
            concept_hidden, relation_hidden = self.gcn(concept_hidden, relation_hidden, head, tail,
                                                            triple_label, i)
        return concept_hidden, relation_hidden

    def gcn(self, concept_hidden, relation_hidden, head, tail, triple_label, layer_idx):
        # shape:
        # concept_hidden: (batch_size, max_mem_size, hidden_size)
        # head: (batch_size, 5, max_trp_size)
        batch_size = head.size(0)
        max_trp_size = head.size(1)
        max_mem_size = concept_hidden.size(1)
        hidden_size = concept_hidden.size(2)

        update_node = torch.zeros_like(concept_hidden).to(self.device).float() # (batch_size, max_mem_size, hidden_size)
        count = torch.ones_like(head).to(self.device).masked_fill_(triple_label == -1, 0).float() # (batch_size, max_trp_size)
        count_out = torch.zeros(batch_size, max_mem_size).to(head.device).float() # (batch_size, max_mem_size)

        o = concept_hidden.gather(1, head.unsqueeze(2).expand(batch_size, max_trp_size, hidden_size)) # (batch_size, max_trp_size, hidden_size)
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, tail, dim=1, out=update_node)
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), tail, dim=1, out=update_node)
        scatter_add(count, tail, dim=1, out=count_out)

        o = concept_hidden.gather(1, tail.unsqueeze(2).expand(batch_size, max_trp_size, hidden_size))
        o = o.masked_fill(triple_label.unsqueeze(2) == -1, 0)
        scatter_add(o, head, dim=1, out=update_node)
        scatter_add(- relation_hidden.masked_fill(triple_label.unsqueeze(2) == -1, 0), head, dim=1, out=update_node)
        scatter_add(count, head, dim=1, out=count_out)

        update_node = self.W_s[layer_idx](concept_hidden) + self.W_n[layer_idx](update_node) / count_out.clamp(
            min=1).unsqueeze(2)
        update_node = nn.ReLU()(update_node)

        return update_node, self.W_r[layer_idx](relation_hidden)

    def multi_hop(self, triple_prob, distance, head, tail, concept_label, triple_label, gamma=0.8, iteration=3,
                       method="avg"):
        '''
        triple_prob: bsz x L x mem_t
        distance: bsz x mem
        head, tail: bsz x mem_t
        concept_label: bsz x mem
        triple_label: bsz x mem_t

        Init binary vector with source concept == 1 and others 0
        expand to size: bsz x L x mem
        '''
        concept_probs = []

        cpt_size = (triple_prob.size(0), triple_prob.size(1), distance.size(1))
        init_mask = torch.zeros_like(distance).unsqueeze(1).expand(*cpt_size).to(distance.device).float()
        init_mask.masked_fill_((distance == 0).unsqueeze(1), 1)
        final_mask = init_mask.clone()

        init_mask.masked_fill_((concept_label == -1).unsqueeze(1), 0)
        concept_probs.append(init_mask)

        head = head.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
        tail = tail.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)

        for step in range(iteration):
            '''
            Calculate triple head score
            '''
            node_score = concept_probs[-1]
            triple_head_score = node_score.gather(2, head)
            triple_head_score.masked_fill_((triple_label == -1).unsqueeze(1), 0)
            '''
            Method: 
                - avg:
                    s(v) = Avg_{u \in N(v)} gamma * s(u) + R(u->v) 
                - max: 
                    s(v) = max_{u \in N(v)} gamma * s(u) + R(u->v)
            '''
            update_value = triple_head_score * gamma + triple_prob
            out = torch.zeros_like(node_score).to(node_score.device).float()
            if method == "max":
                scatter_max(update_value, tail, dim=-1, out=out)
            elif method == "avg":
                scatter_mean(update_value, tail, dim=-1, out=out)
            out.masked_fill_((concept_label == -1).unsqueeze(1), 0)

            concept_probs.append(out)

        '''
        Natural decay of concept that is multi-hop away from source
        '''
        total_concept_prob = final_mask * -1e5
        for prob in concept_probs[1:]:
            total_concept_prob += prob
        # bsz x L x mem

        return total_concept_prob

    def comp_cause(self, concept_ids, relation, head, tail, triple_label):
        self.batch_size = concept_ids.size(0)

        ## calculate graph
        memory = self.embedding(concept_ids)
        rel_repr = self.embedding(relation)
        node_repr, rel_repr = self.multi_layer_gcn(memory, rel_repr, head, tail, triple_label,
                                                        layer_number=2)
        head_repr = torch.gather(node_repr, 1,
                                 head.unsqueeze(-1).expand(node_repr.size(0), head.size(1), node_repr.size(-1)))
        tail_repr = torch.gather(node_repr, 1,
                                 tail.unsqueeze(-1).expand(node_repr.size(0), tail.size(1), node_repr.size(-1)))
        triple_repr = torch.cat((head_repr, rel_repr, tail_repr), dim=-1)
        triple_repr = self.triple_linear(triple_repr)

        encoded_cause = torch.sum(triple_repr, dim=1)

        assert (not torch.isnan(triple_repr).any().item())

        return triple_repr, encoded_cause

    def comp_pointer(self, hidden_state, concept_label, distance, head, tail, triple_repr, triple_label, vocab_map, map_mask):
        triple_logits = torch.matmul(hidden_state, triple_repr.transpose(1, 2))
        triple_prob = nn.Sigmoid()(triple_logits)
        triple_prob = triple_prob.masked_fill((triple_label == -1).unsqueeze(1), 0)

        cpt_probs = self.multi_hop(triple_prob, distance, head, tail, concept_label, triple_label, config.hop_num)
        cpt_probs = F.log_softmax(cpt_probs, dim=-1)
        cpt_probs_vocab = cpt_probs.gather(-1, vocab_map.unsqueeze(1).expand(cpt_probs.size(0),
                                                                             cpt_probs.size(1), -1))

        # cpt_probs_vocab = torch.sum(cpt_probs_vocab, dim=1)
        cpt_probs_vocab.masked_fill_((map_mask == 0).unsqueeze(1), 0)
        # bsz x graph_num x L x vocab

        gate = F.log_softmax(self.gate_linear(hidden_state), dim=-1)
        # bsz x L x 1

        return gate, cpt_probs_vocab


class MultiHopCause(nn.Module):

    def __init__(self, vocab, decoder_number, model_file_path=None, load_optim=False):
        """
        vocab: a Lang type data, which is defined in data_reader.py
        decoder_number: the number of classes
        """
        super(MultiHopCause, self).__init__()
        self.iter = 0
        self.current_loss = 1000
        self.device = config.device
        self.vocab = vocab
        self.vocab_size = vocab.n_words

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.gcause = G_Cause(self.embedding)

        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)

        self.decoder = Decoder(config.emb_dim, hidden_size=config.hidden_dim, num_layers=config.hop,
                               num_heads=config.heads, total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, max_length=config.max_length)

        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(config.hidden_dim, self.vocab_size)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if (config.noam):
            optimizer = torch.optim.Adam(self.parameters(), lr=0, weight_decay=config.weight_decay, betas=(0.9, 0.98),
                                         eps=1e-9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[config.schedule*i for i in range(4)], gamma=0.1)
            self.scheduler = NoamOpt(config.hidden_dim, 1, 8000, optimizer, scheduler)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[config.schedule*i for i in range(4)], gamma=0.1)

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.iter = state['iter']
            self.current_loss = state['current_loss']
            self.embedding.load_state_dict(state['embedding_dict'])
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.gcause.load_state_dict(state['cause_encoder_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
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
            'iter': self.iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'cause_encoder_dict': self.gcause.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.scheduler.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}'.format(
            iter, running_avg_ppl))
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, train=True):
        enc_batch, cause_batch = get_input_from_batch(batch)
        graphs, use_graph = get_graph_from_batch(batch)
        dec_batch = get_output_from_batch(batch)

        if (config.noam):
            self.scheduler.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        ## encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mak = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mak, mask_src)

        ## graph processing
        if use_graph:
            concept_ids, concept_label, distance, relation, head, tail, triple_label, vocab_map, map_mask = graphs
            triple_repr, cause_repr = self.gcause.comp_cause(concept_ids, relation, head, tail, triple_label)

        ## decode
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1).to(self.device)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        dec_input = self.embedding(dec_batch_shift)

        if use_graph:
            dec_input[:, 0] = dec_input[:, 0] + cause_repr

        ## logit
        pre_logit, attn_dist = self.decoder(dec_input, encoder_outputs, (mask_src, mask_trg))
        logit = self.generator(pre_logit)
        if use_graph:
            gate, cpt_probs_vocab = self.gcause.comp_pointer(pre_logit, concept_label, distance, head, tail, triple_repr,
                                triple_label, vocab_map, map_mask)
            logit = logit * (1 - gate) + gate * cpt_probs_vocab

        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        loss_bce_program, loss_bce_caz, program_acc = 0, 0, 0

        # multi-task
        if config.emo_multitask:
            # add the loss function of label prediction
            # q_h = torch.mean(encoder_outputs,dim=1)
            q_h = encoder_outputs[:, 0]  # the first token of the sentence CLS, shape: (batch_size, 1, hidden_size)
            logit_prob = self.decoder_key(q_h).to(self.device)  # (batch_size, 1, decoder_num)
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
        graphs, use_graph = get_graph_from_batch(batch)

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        if use_graph:
            concept_ids, concept_label, distance, relation, head, tail, triple_label, vocab_map, map_mask = graphs
            ## graph_encoder
            triple_repr, cause_repr = self.gcause.comp_cause(concept_ids, relation, head, tail, triple_label)

        ys = torch.ones(1, 1).fill_(config.SOS_idx).long().to(self.device)
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            dec_input = self.embedding(ys)
            if use_graph:
                dec_input[:, 0] = dec_input[:, 0] + cause_repr

            out, attn_dist = self.decoder(dec_input, encoder_outputs, (mask_src, mask_trg))

            if use_graph:
                gate, cpt_probs_vocab = self.gcause.comp_pointer(out, concept_label, distance, head, tail, triple_repr,
                                                            triple_label, vocab_map, map_mask)
                prob = self.generator(out) * (1 - gate) + gate * cpt_probs_vocab

            else:
                prob = self.generator(out)

            _, next_word = torch.max(prob[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in
                                  next_word.view(-1)])
            next_word = next_word.data[0]

            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).to(self.device)], dim=1).to(self.device)

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


