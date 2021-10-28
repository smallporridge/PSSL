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

def kernel_mus(n_kernels):
        l_mu = [1]
        if n_kernels == 1:
            return l_mu
        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

def kernel_sigmas(n_kernels):
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma
    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

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

class knrm(nn.Module):
    def __init__(self, k):
        super(knrm, self).__init__()
        tensor_mu = torch.FloatTensor(kernel_mus(k)).cuda(cudaid)
        tensor_sigma = torch.FloatTensor(kernel_sigmas(k)).cuda(cudaid)
        self.mu = Variable(tensor_mu, requires_grad = False).view(1, 1, 1, k)
        self.sigma = Variable(tensor_sigma, requires_grad = False).view(1, 1, 1, k)
        self.dense = nn.Linear(k, 1, 1)

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1) # n*m*d*1
        pooling_value = torch.exp((- ((sim - self.mu) ** 2) / (self.sigma ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * 0.01 * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1)#soft-TF
        return log_pooling_sum

    def forward(self, inputs_q, inputs_d, mask_q, mask_d):
        q_embed_norm = F.normalize(inputs_q, 2, 2)
        d_embed_norm = F.normalize(inputs_d, 2, 2)
        mask_d = mask_d.view(mask_d.size()[0], 1, mask_d.size()[1], 1)
        mask_q = mask_q.view(mask_q.size()[0], mask_q.size()[1], 1)
        log_pooling_sum = self.get_intersect_matrix(q_embed_norm, d_embed_norm, mask_q, mask_d)
        output = F.tanh(self.dense(log_pooling_sum))
        return output

class Contextual(nn.Module):

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
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1):

        super().__init__()

        self.d_word_vec = d_word_vec
        self.max_hislen = max_hislen
        self.max_querylen = max_querylen
        self.knrm = knrm(11)
        self.feature_layer=nn.Sequential(nn.Linear(110, 1),nn.Tanh())

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

        self.score_layer=nn.Linear(3, 1)

    def pairwise_loss(self, score1, score2):
        return (1/(1+torch.exp(score2-score1)))


    def forward(self, query, docs1, docs2, his_sequence, his_pos, features1, features2):
        qenc_output = self.embedding(query)
        d1enc_output = self.embedding(docs1)
        d2enc_output = self.embedding(docs2)

        query_encoding_1 = torch.sum(qenc_output, 1)
        doc1_encoding_1 = torch.sum(d1enc_output, 1)
        doc2_encoding_1 = torch.sum(d2enc_output, 1)

        his_sequence_embedding = self.embedding(his_sequence)
        his_sequence_encoding = his_sequence_embedding.view(-1, self.max_querylen, self.d_word_vec)

        qenc_output_2 = self.encoder_query(qenc_output, query)
        d1enc_output_2 = self.encoder_query(d1enc_output, docs1)
        d2enc_output_2 = self.encoder_query(d2enc_output, docs2)
        query_encoding_2 = torch.sum(qenc_output_2, 1)
        doc1_encoding_2 = torch.sum(d1enc_output_2, 1)
        doc2_encoding_2 = torch.sum(d2enc_output_2, 1)
        his_sequence_encoding = torch.sum(his_sequence_encoding, 1)

        his_sequence_encoding = his_sequence_encoding.view(-1, self.max_hislen+1, self.d_word_vec)
        his_sequence_encoding = self.encoder_sequence(his_sequence_encoding, his_pos)[:,-1,:]

        score_1_1 = torch.cosine_similarity(his_sequence_encoding, doc1_encoding_1, dim=1).unsqueeze(1)
        score_2_1 = torch.cosine_similarity(his_sequence_encoding, doc2_encoding_1, dim=1).unsqueeze(1)

        # score_1_2 = self.feature_layer(features1)
        # score_2_2 = self.feature_layer(features2)

        score_1_2 = torch.tanh(features1).unsqueeze(1)
        score_2_2 = torch.tanh(features2).unsqueeze(1)

        q_mask = get_non_pad_mask(query)
        d1_mask = get_non_pad_mask(docs1)
        d2_mask = get_non_pad_mask(docs2)
        # score_1_3 = self.knrm(qenc_output, d1enc_output, q_mask, d1_mask)
        # score_2_3 = self.knrm(qenc_output, d2enc_output, q_mask, d2_mask)

        # score_1_4 = self.knrm(qenc_output_2, d1enc_output_2, q_mask, d1_mask)
        # score_2_4 = self.knrm(qenc_output_2, d2enc_output_2, q_mask, d2_mask)

        score_1_5 = torch.cosine_similarity(query_encoding_2, doc1_encoding_2, dim=1).unsqueeze(1)
        score_2_5 = torch.cosine_similarity(query_encoding_2, doc2_encoding_2, dim=1).unsqueeze(1)

        score1_all = torch.cat([score_1_1, score_1_2, score_1_5], 1)
        score2_all = torch.cat([score_2_1, score_2_2, score_2_5], 1)
        score_1 = self.score_layer(score1_all)
        score_2 = self.score_layer(score2_all)

        score = torch.cat([score_1, score_2], 1)

        p_score = torch.cat([self.pairwise_loss(score_1, score_2),
                    self.pairwise_loss(score_2, score_1)], 1)

        pre = F.softmax(score, 1)

        return score, pre, p_score


