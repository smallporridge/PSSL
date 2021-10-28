''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Constants
import pickle
from Layers import EncoderLayer, DecoderLayer
from torch.autograd import Variable

cudaid=1
vocabulary = pickle.load(open('vocab.dict', 'rb'))

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class Encoder_high(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_emb, src_pos, return_attns=False, needpos=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        if needpos:
            enc_output = src_emb + self.position_enc(src_pos)
        else:
            enc_output = src_emb

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

class contrastive(nn.Module):

    def load_embedding(self): # load the pretrained embedding
        weight = torch.zeros(len(vocabulary), self.d_word_vec)
        weight[-1] = torch.rand(self.d_word_vec)
        weight[-2] = torch.rand(self.d_word_vec)
        with open('word2vec.txt', 'r') as fr:
            for line in fr:
                line = line.strip().split()
                wordid = vocabulary[line[0]]
                weight[wordid, :] = torch.FloatTensor([float(t) for t in line[1:]]) 
        print("Successfully load the word vectors...")
        return weight

    def __init__(
            self, max_querylen=20, max_hislen=50, d_word_vec=100, d_model=100, d_inner=512,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, temperature=0.1):

        super().__init__()

        self.d_word_vec = d_word_vec
        self.max_hislen = max_hislen
        self.max_querylen = max_querylen
        self.temperature = temperature
        self.encoder_query = Encoder_high(
            len_max_seq=max_querylen,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.encoder_sequence = Encoder_high(
            len_max_seq=max_hislen+1,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)
        #self.embedding = nn.Embedding.from_pretrained(self.load_embedding()) #static word vectors
        self.embedding = nn.Embedding(len(vocabulary), self.d_word_vec) 
        self.embedding.weight.data.copy_(self.load_embedding()) #finetuned word vectors
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def pairwise_loss(self, score1, score2):
        return (1/(1+torch.exp(score2-score1)))


    def forward(self, his_pos_1, his_pos_2, his_contrastive_1, his_contrastive_2, qd):
        batch_size = his_pos_1.size(0)

        if qd:
            his_contrastive_1_encoding = self.embedding(his_contrastive_1)
            his_contrastive_2_encoding = self.embedding(his_contrastive_2)

            his_contrastive_1_encoding = self.encoder_query(his_contrastive_1_encoding, his_contrastive_1)
            his_contrastive_2_encoding = self.encoder_query(his_contrastive_2_encoding, his_contrastive_2)

            his_contrastive_1_encoding = torch.sum(his_contrastive_1_encoding, 1)
            his_contrastive_2_encoding = torch.sum(his_contrastive_2_encoding, 1)
        else:
            his_contrastive_1_embedding = self.embedding(his_contrastive_1)
            his_contrastive_2_embedding = self.embedding(his_contrastive_2)

            his_contrastive_1_encoding = his_contrastive_1_embedding.view(-1, self.max_querylen, self.d_word_vec)
            his_contrastive_2_encoding = his_contrastive_2_embedding.view(-1, self.max_querylen, self.d_word_vec)  

            his_contrastive_1_encoding = torch.sum(his_contrastive_1_encoding, 1)
            his_contrastive_2_encoding = torch.sum(his_contrastive_2_encoding, 1)

            his_contrastive_1_encoding = his_contrastive_1_encoding.view(-1, self.max_hislen+1, self.d_word_vec)
            his_contrastive_2_encoding = his_contrastive_2_encoding.view(-1, self.max_hislen+1, self.d_word_vec)   

            his_contrastive_1_encoding_no =  torch.mean(his_contrastive_1_encoding, 1)
            his_contrastive_2_encoding_no =  torch.mean(his_contrastive_2_encoding, 1)

            his_contrastive_1_encoding = self.encoder_sequence(his_contrastive_1_encoding, his_pos_1, needpos=True)[:,-1,:]
            his_contrastive_2_encoding = self.encoder_sequence(his_contrastive_2_encoding, his_pos_2, needpos=True)[:,-1,:] 

        sent_norm2 = his_contrastive_1_encoding.norm(dim=-1, keepdim=True)  # [batch]
        sent_norm3 = his_contrastive_2_encoding.norm(dim=-1, keepdim=True)  # [batch]
        batch_self_11 = torch.einsum("ad,bd->ab", his_contrastive_1_encoding, his_contrastive_1_encoding) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm2) + 1e-6)  # [batch, batch]
        batch_cross_12 = torch.einsum("ad,bd->ab", his_contrastive_1_encoding, his_contrastive_2_encoding) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm3) + 1e-6)  # [batch, batch]
        batch_self_11 = batch_self_11 / self.temperature
        batch_cross_12 = batch_cross_12 / self.temperature
        batch_first = torch.cat([batch_self_11, batch_cross_12], dim=-1)  # [batch, batch * 2]
        batch_arange = torch.arange(batch_size).cuda(cudaid)
        mask = F.one_hot(batch_arange, num_classes=batch_size * 2) * -1e10
        batch_first += mask
        batch_label1 = batch_arange + batch_size  # [batch]

        batch_self_22 = torch.einsum("ad,bd->ab", his_contrastive_2_encoding, his_contrastive_2_encoding) / (torch.einsum("ad,bd->ab", sent_norm3, sent_norm3) + 1e-6)  # [batch, batch]
        batch_cross_21 = torch.einsum("ad,bd->ab", his_contrastive_2_encoding, his_contrastive_1_encoding) / (torch.einsum("ad,bd->ab", sent_norm3, sent_norm2) + 1e-6)  # [batch, batch]
        batch_self_22 = batch_self_22 / self.temperature
        batch_cross_21 = batch_cross_21 / self.temperature
        batch_second = torch.cat([batch_self_22, batch_cross_21], dim=-1)  # [batch, batch * 2]
        batch_second += mask
        batch_label2 = batch_arange + batch_size  # [batch]

        batch_predict = torch.cat([batch_first, batch_second], dim=0)
        batch_label = torch.cat([batch_label1, batch_label2], dim=0)  # [batch * 2]
        contras_loss = self.ce_loss(batch_predict, batch_label)

        batch_logit = batch_predict.argmax(dim=-1)
        acc = torch.sum(batch_logit == batch_label).float() / (batch_size * 2)

        return contras_loss, acc, his_contrastive_1_encoding