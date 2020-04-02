#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-20 下午2:41
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from torch import nn
from utils import constants

from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


class CNNchar(nn.Module):
    """
    Class for CH conditioning
    """

    def __init__(self, n_ch_tokens, ch_maxlen, ch_emb_size,
                 conv_filter_nums, conv_filter_sizes, device, ch_drop=0.25):
        super(CNNchar, self).__init__()
        assert len(conv_filter_nums) == len(conv_filter_sizes)

        self.n_ch_tokens = n_ch_tokens
        self.ch_maxlen = ch_maxlen
        self.ch_emb_size = ch_emb_size
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_filter_nums = conv_filter_nums
        self.ch_drop = nn.Dropout(ch_drop)
        self.device = device
        self.char_embedding = nn.Embedding(
            self.n_ch_tokens,
            self.ch_emb_size,
            padding_idx=constants.PAD_IDX
        )
        self.char_encoders = []
        for i, filter_size in enumerate(self.conv_filter_sizes):
            f = nn.Conv3d(
                in_channels=1,
                out_channels=self.conv_filter_nums[i],
                kernel_size=(
                    1, filter_size, self.ch_emb_size
                )
            )
            self.char_encoders.append(f.to(device))
        self.cnns = nn.ModuleList(self.char_encoders)
        self.feature_dim = sum(self.conv_filter_nums)
        self.batch_norm = nn.BatchNorm1d(self.feature_dim, affine=False)
        self.highway1 = Highway(self.feature_dim)
        self.highway2 = Highway(self.feature_dim)

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        batch_size, max_len, max_len_char = inputs.size(0), inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # input_embed = self.dropout_embed(input_embed)
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.ch_emb_size)
        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = F.tanh(char_encoder(input_embed))
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0], -1)
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        char_conv_outputs = char_conv_outputs.permute(0, 2, 1)
        char_conv_outputs = self.ch_drop(char_conv_outputs)
        highway_input = char_conv_outputs.contiguous().view(-1, self.feature_dim)
        highway_input = self.batch_norm(highway_input)
        highway_output = self.highway1(highway_input)
        highway_output = self.highway2(highway_output).contiguous().view(batch_size, max_len, -1)

        return highway_output

    def init_ch(self):
        initrange = 0.5 / self.ch_emb_size
        nn.init.uniform_(self.char_embedding.weight, -initrange, initrange)
        for name, p in self.cnns.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0)
            elif "weight" in name:
                nn.init.xavier_uniform_(p)


class CharCNN(nn.Module):
    """
    Class for CH conditioning
    """

    def __init__(self, n_ch_tokens, ch_maxlen, ch_emb_size,
                 ch_feature_maps, ch_kernel_sizes, ch_drop=0.25):
        super(CharCNN, self).__init__()
        assert len(ch_feature_maps) == len(ch_kernel_sizes)

        self.n_ch_tokens = n_ch_tokens
        self.ch_maxlen = ch_maxlen
        self.ch_emb_size = ch_emb_size
        self.ch_feature_maps = ch_feature_maps
        self.ch_kernel_sizes = ch_kernel_sizes
        self.ch_drop = nn.Dropout(ch_drop)
        self.feature_dim = sum(self.ch_feature_maps)
        self.batch_norm = nn.BatchNorm1d(self.feature_dim, affine=False)
        self.highway1 = Highway(self.feature_dim)
        self.highway2 = Highway(self.feature_dim)

        self.feature_mappers = nn.ModuleList()
        for i in range(len(self.ch_feature_maps)):
            reduced_length = self.ch_maxlen - self.ch_kernel_sizes[i] + 1
            self.feature_mappers.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1,
                        out_channels=self.ch_feature_maps[i],
                        kernel_size=(
                            self.ch_kernel_sizes[i],
                            self.ch_emb_size
                        )
                    ),
                    nn.Tanh(),
                    nn.MaxPool2d(kernel_size=(reduced_length, 1))
                )
            )

        self.embs = nn.Embedding(
            self.n_ch_tokens,
            self.ch_emb_size,
            padding_idx=0
        )

    def forward(self, x):
        # x - [batch_size x maxlen]
        bsize, length = x.size()
        assert length == self.ch_maxlen
        x_embs = self.embs(x).view(bsize, 1, self.ch_maxlen, self.ch_emb_size)

        cnn_features = []
        for i in range(len(self.ch_feature_maps)):
            cnn_features.append(
                self.feature_mappers[i](x_embs).view(bsize, -1)
            )
        char_output = self.ch_drop(torch.cat(cnn_features, dim=1))
        char_output = self.batch_norm(char_output)
        highway_output = self.highway1(char_output)
        highway_output = self.highway2(highway_output)
        return highway_output

    def init_ch(self):
        initrange = 0.5 / self.ch_emb_size
        nn.init.uniform_(self.embs.weight, -initrange, initrange)
        for name, p in self.feature_mappers.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0)
            elif "weight" in name:
                nn.init.xavier_uniform_(p)


class Hidden(nn.Module):
    """
    Class for Hidden conditioning
    """

    def __init__(self, in_size, out_size):
        super(Hidden, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.out_size
        )
        self.h_bn = nn.BatchNorm1d(self.out_size)

    def forward(self, rnn_type, v, hidden):
        if rnn_type == 'LSTM':
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0), hidden], dim=-1)
            hidden = F.tanh(self.linear(inp_h))
        else:
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0), hidden], dim=-1)
            hidden = F.tanh(self.linear(inp_h))
        return hidden

    def init_hidden(self):
        nn.init.constant_(self.linear.bias, 0.0)
        nn.init.xavier_normal_(self.linear.weight)


class Gated(nn.Module):
    """
    Class for Gated conditioning
    """

    def __init__(self, cond_size, hidden_size):
        super(Gated, self).__init__()
        self.cond_size = cond_size
        self.hidden_size = hidden_size
        self.in_size = self.cond_size + self.hidden_size
        self.zt_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )
        self.zt_bn = nn.BatchNorm1d(self.hidden_size)
        self.rt_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.cond_size
        )
        self.rt_bn = nn.BatchNorm1d(self.cond_size)
        self.ht_linear = nn.Linear(
            in_features=self.in_size,
            out_features=self.hidden_size
        )
        self.ht_bn = nn.BatchNorm1d(self.hidden_size)

    def forward(self, rnn_type, v, hidden):
        if rnn_type == 'LSTM':
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0), hidden], dim=-1)
            z_t = F.sigmoid(self.zt_linear(inp_h))
            r_t = F.sigmoid(self.rt_linear(inp_h))
            mul = torch.mul(r_t, v)
            hidden_ = torch.cat([mul, hidden], dim=-1)
            hidden_ = F.tanh(self.ht_linear(hidden_))
            hidden = torch.mul((1 - z_t), hidden) + torch.mul(z_t, hidden_)
        else:
            inp_h = torch.cat(
                [torch.unsqueeze(v, 0), hidden], dim=-1)
            z_t = F.sigmoid(self.zt_linear(inp_h))
            r_t = F.sigmoid(self.rt_linear(inp_h))
            mul = torch.mul(r_t, v)
            hidden_ = torch.cat([mul, hidden], dim=-1)
            hidden_ = F.tanh(self.ht_linear(hidden_))
            hidden = torch.mul((1 - z_t), hidden) + torch.mul(z_t, hidden_)
        return hidden

    def init_gated(self):
        nn.init.constant_(self.zt_linear.bias, 0.0)
        nn.init.xavier_normal_(self.zt_linear.weight)
        nn.init.constant_(self.rt_linear.bias, 0.0)
        nn.init.xavier_normal_(self.rt_linear.weight)
        nn.init.constant_(self.ht_linear.bias, 0.0)
        nn.init.xavier_normal_(self.ht_linear.weight)


class InputAttention(nn.Module):
    """
    Class for Input Attention conditioning
    """

    def __init__(self, n_attn_tokens, n_attn_embsize,
                 n_attn_hid, attn_dropout, sparse=False):
        super(InputAttention, self).__init__()
        self.n_attn_tokens = n_attn_tokens
        self.n_attn_embsize = n_attn_embsize
        self.n_attn_hid = n_attn_hid
        self.attn_dropout = attn_dropout
        self.sparse = sparse

        self.embs = nn.Embedding(
            num_embeddings=self.n_attn_tokens,
            embedding_dim=self.n_attn_embsize,
            padding_idx=constants.PAD_IDX,
            sparse=self.sparse
        )

        self.ann = nn.Sequential(
            nn.Dropout(p=self.attn_dropout),
            nn.Linear(
                in_features=self.n_attn_embsize,
                out_features=self.n_attn_hid
            ),
            nn.Tanh()
        )  # maybe use ReLU or other?

        self.a_linear = nn.Linear(
            in_features=self.n_attn_hid,
            out_features=self.n_attn_embsize
        )

    def forward(self, word, context):
        x_embs = self.embs(word)
        mask = self.get_mask(context)
        att = mask * x_embs
        return mask, att

    def get_mask(self, context):
        context_embs = self.embs(context)
        lengths = (context != constants.PAD_IDX)
        for_sum_mask = lengths.unsqueeze(2).float()
        lengths = lengths.sum(1).float().view(-1, 1)
        logits = self.a_linear(
            (self.ann(context_embs) * for_sum_mask).sum(1) / lengths
        )
        return F.sigmoid(logits)

    def init_attn(self, freeze):
        initrange = 0.5 / self.n_attn_embsize
        with torch.no_grad():
            nn.init.uniform_(self.embs.weight, -initrange, initrange)
            nn.init.xavier_uniform_(self.a_linear.weight)
            nn.init.constant_(self.a_linear.bias, 0)
            nn.init.xavier_uniform_(self.ann[1].weight)
            nn.init.constant_(self.ann[1].bias, 0)
        self.embs.weight.requires_grad = not freeze

    def init_attn_from_pretrained(self, weights, freeze):
        self.load_state_dict(weights)
        self.embs.weight.requires_grad = not freeze


class ScaledDotAttention(nn.Module):
    def __init__(self, n_attn_tokens, n_attn_embsize, n_attn_hid, attn_dropout,
                 freeze, device, pretrain_w2v=None):
        super(ScaledDotAttention, self).__init__()
        self.context_tokens = n_attn_tokens
        self.attn_embsize = n_attn_embsize
        self.attn_hid = n_attn_hid
        self.dropout = nn.Dropout(attn_dropout)
        self.device = device
        self.embedding = nn.Embedding(
            self.context_tokens,
            self.attn_embsize,
            padding_idx=constants.PAD_IDX
        )
        self.rnn = nn.GRU(
            input_size=self.attn_embsize,
            hidden_size=self.attn_hid,
            num_layers=1,
            dropout=0,
            bidirectional=True
        )
        self.softmax = nn.Softmax(dim=2)
        self.linear_q = nn.Linear(self.attn_embsize, self.attn_embsize)
        self.linear_k = nn.Linear(self.attn_hid * 2, self.attn_embsize)

        self.init_weights(freeze=freeze, pretrain_w2v=pretrain_w2v)

    def forward(self, word, context, pool_type="max"):
        word_emb = self.embedding(word)
        context_emb = self.embedding(context)
        lengths = (context != constants.PAD_IDX).sum(dim=0).detach().cpu()

        # Sort by length (keep idx)
        context_len_sorted, idx_sort = np.sort(lengths.numpy())[::-1], np.argsort(-lengths.numpy())
        context_len_sorted = torch.from_numpy(context_len_sorted.copy())
        idx_unsort = np.argsort(idx_sort)
        context_emb = context_emb.index_select(1, torch.from_numpy(idx_sort).to(self.device))
        context_emb = pack(context_emb, context_len_sorted, batch_first=False)
        context_vec, _ = self.rnn(context_emb, None)
        context_vec = unpack(context_vec, batch_first=False)[0]

        # Un-sort by length
        context_vec = context_vec.index_select(1, torch.from_numpy(idx_unsort).to(self.device))
        # Pooling
        if pool_type == "mean":
            lengths = torch.FloatTensor(lengths.numpy().copy()).unsqueeze(1)
            emb = torch.sum(context_vec, 0).squeeze(0)
            if emb.ndimension() == 1:
                emb = emb.unsqueeze(0)
            emb = emb / lengths.expand_as(emb).to(self.device)
        elif pool_type == "max":
            emb = torch.max(context_vec, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        V = F.relu(self.linear_q(word_emb.unsqueeze(1)))
        C = F.relu(self.linear_k(torch.transpose(context_vec, 0, 1)))
        T = torch.transpose(context_vec, 0, 1)

        scale = (C.size(-1)) ** -0.5
        att = torch.bmm(V, C.transpose(1, 2)) * scale
        att = self.softmax(att)
        att = self.dropout(att)
        c = torch.bmm(att, T)
        c = c.squeeze(1)
        return c, emb, att

    def init_weights(self, freeze, pretrain_w2v=None):
        if pretrain_w2v is not None:
            with open(pretrain_w2v, 'rb') as infile:
                emb = pickle.load(infile)
                infile.close()
            self.embedding.weight.data.copy_(
                torch.from_numpy(emb)
            )
        else:
            self.embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(self.context_tokens, self.attn_embsize)
                )
            )
        self.embedding.weight.requires_grad = not freeze
        nn.init.constant_(self.linear_q.bias, 0.0)
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.constant_(self.linear_k.bias, 0.0)
        nn.init.xavier_normal_(self.linear_k.weight)
        self.init_rnn()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_rnn(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


class CNNchar(nn.Module):
    """
    Class for CH conditioning
    """

    def __init__(self, n_ch_tokens, ch_maxlen, ch_emb_size,
                 conv_filter_nums, conv_filter_sizes, device, ch_drop=0.25):
        super(CNNchar, self).__init__()
        assert len(conv_filter_nums) == len(conv_filter_sizes)

        self.n_ch_tokens = n_ch_tokens
        self.ch_maxlen = ch_maxlen
        self.ch_emb_size = ch_emb_size
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_filter_nums = conv_filter_nums
        self.ch_drop = nn.Dropout(ch_drop)
        self.device = device
        self.char_embedding = nn.Embedding(
            self.n_ch_tokens,
            self.ch_emb_size,
            padding_idx=constants.PAD_IDX
        )
        self.char_encoders = []
        for i, filter_size in enumerate(self.conv_filter_sizes):
            f = nn.Conv3d(
                in_channels=1,
                out_channels=self.conv_filter_nums[i],
                kernel_size=(
                    1, filter_size, self.ch_emb_size
                )
            )
            self.char_encoders.append(f.to(device))
        self.cnns = nn.ModuleList(self.char_encoders)
        self.feature_dim = sum(self.conv_filter_nums)
        self.batch_norm = nn.BatchNorm1d(self.feature_dim, affine=False)
        self.highway1 = Highway(self.feature_dim)
        self.highway2 = Highway(self.feature_dim)

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]

        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        batch_size, max_len, max_len_char = inputs.size(0), inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.char_embedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # input_embed = self.dropout_embed(input_embed)
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.ch_emb_size)
        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = F.tanh(char_encoder(input_embed))
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0], -1)
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        char_conv_outputs = char_conv_outputs.permute(0, 2, 1)
        char_conv_outputs = self.ch_drop(char_conv_outputs)
        highway_input = char_conv_outputs.contiguous().view(-1, self.feature_dim)
        highway_input = self.batch_norm(highway_input)
        highway_output = self.highway1(highway_input)
        highway_output = self.highway2(highway_output).contiguous().view(batch_size, max_len, -1)

        return highway_output

    def init_ch(self):
        initrange = 0.5 / self.ch_emb_size
        nn.init.uniform_(self.char_embedding.weight, -initrange, initrange)
        for name, p in self.cnns.named_parameters():
            if "bias" in name:
                nn.init.constant_(p, 0)
            elif "weight" in name:
                nn.init.xavier_uniform_(p)


class Highway(nn.Module):
    """Highway network"""

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)
        self.init_weights()

    def forward(self, x):
        t = F.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1 - t, x)

    def init_weights(self):
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_normal_(self.fc2.weight)


class CAGAttention(nn.Module):
    def __init__(self, n_attn_tokens, n_attn_embsize, n_attn_hid, attn_dropout,
                 freeze, device, pretrain_w2v=None):
        super(CAGAttention, self).__init__()
        self.context_tokens = n_attn_tokens
        self.attn_embsize = n_attn_embsize
        self.attn_hid = n_attn_hid
        self.dropout = nn.Dropout(attn_dropout)
        self.device = device
        self.embedding = nn.Embedding(
            self.context_tokens,
            self.attn_embsize,
            padding_idx=constants.PAD_IDX
        )
        self.rnn = nn.GRU(
            input_size=self.attn_embsize,
            hidden_size=self.attn_hid,
            num_layers=1,
            dropout=0,
            bidirectional=True
        )
        self.softmax = nn.Softmax(dim=2)
        self.linear_q = nn.Linear(300, self.attn_embsize)
        self.z1 = nn.Linear(self.attn_hid * 8, self.attn_embsize)
        self.z2 = nn.Linear(self.attn_embsize, self.attn_embsize)
        self.f1 = nn.Linear(self.attn_hid * 8, self.attn_embsize)
        self.f2 = nn.Linear(self.attn_embsize, self.attn_embsize)

        self.init_weights(freeze=freeze, pretrain_w2v=pretrain_w2v)

    def forward(self, word, context, pool_type="max", gate_attention=True):
        word_emb = self.embedding(word)
        np.savetxt('emb.txt',word_emb)
        context_emb = self.embedding(context)
        lengths = (context != constants.PAD_IDX).sum(dim=0).detach().cpu()

        # Sort by length (keep idx)
        context_len_sorted, idx_sort = np.sort(lengths.numpy())[::-1], np.argsort(-lengths.numpy())
        context_len_sorted = torch.from_numpy(context_len_sorted.copy())
        idx_unsort = np.argsort(idx_sort)
        context_emb = context_emb.index_select(1, torch.from_numpy(idx_sort).to(self.device))
        context_emb = pack(context_emb, context_len_sorted, batch_first=False)
        context_vec, _ = self.rnn(context_emb, None)
        context_vec = unpack(context_vec, batch_first=False)[0]

        # Un-sort by length
        context_vec = context_vec.index_select(1, torch.from_numpy(idx_unsort).to(self.device))
        # Pooling
        if pool_type == "mean":
            lengths = torch.FloatTensor(lengths.numpy().copy()).unsqueeze(1)
            emb = torch.sum(context_vec, 0).squeeze(0)
            if emb.ndimension() == 1:
                emb = emb.unsqueeze(0)
            emb = emb / lengths.expand_as(emb).to(self.device)
        elif pool_type == "max":
            emb = torch.max(context_vec, 0)[0]
            if emb.ndimension() == 3:
                emb = emb.squeeze(0)
                assert emb.ndimension() == 2
        a=context_vec.squeeze(1)
        np.savetxt('context.txt', context_vec.squeeze(1))
        elewise = word_emb * emb
        elediff = torch.abs(word_emb - emb)
        Hw = torch.cat(
            [word_emb, emb, elewise, elediff], dim=-1
        )
        V_ = self.linear_q(word_emb)
        Z_ = F.tanh(self.z1(Hw) + self.z2(V_))
        F_ = F.sigmoid(self.f1(Hw) + self.f2(V_))
        U_ = (1 - F_) * V_ + F_ * Z_
        np.savetxt('U_1.txt', U_)
        if gate_attention:
            U_ = self.gated_attention(U_, context_vec)
            np.savetxt('U_2.txt', U_)
        # V = F.relu(self.linear_q(word_emb.unsqueeze(1)))
        # C = F.relu(self.linear_k(torch.transpose(context_vec, 0, 1)))
        # T = torch.transpose(context_vec, 0, 1)
        #
        # scale = (C.size(-1)) ** -0.5
        # att = torch.bmm(V, C.transpose(1, 2)) * scale
        # att = self.softmax(att)
        # att = self.dropout(att)
        # c = torch.bmm(att, T)
        # c = c.squeeze(1)
        return U_, emb

    def init_weights(self, freeze, pretrain_w2v=None):
        if pretrain_w2v is not None:
            with open(pretrain_w2v, 'rb') as infile:
                emb = pickle.load(infile)
                infile.close()
            self.embedding.weight.data.copy_(
                torch.from_numpy(emb)
            )
        else:
            self.embedding.weight.data.copy_(
                torch.from_numpy(
                    self.random_embedding(self.context_tokens, self.attn_embsize)
                )
            )
        self.embedding.weight.requires_grad = not freeze
        nn.init.constant_(self.linear_q.bias, 0.0)
        nn.init.xavier_normal_(self.linear_q.weight)
        nn.init.constant_(self.z1.bias, 0.0)
        nn.init.xavier_normal_(self.z1.weight)
        nn.init.constant_(self.z2.bias, 0.0)
        nn.init.xavier_normal_(self.z2.weight)
        nn.init.constant_(self.f1.bias, 0.0)
        nn.init.xavier_normal_(self.f1.weight)
        nn.init.constant_(self.f2.bias, 0.0)
        nn.init.xavier_normal_(self.f2.weight)
        self.init_rnn()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_rnn(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def tmul(self, t1, t2):
        return torch.mul(t1, t2)

    def tcat(self, t1, t2):
        return torch.cat([t1, t2], axis=2)

    def tsum(self, t1, t2):
        return t1 + t2

    def pairwise_interaction(self, doc, qry):
        # doc: B x N x D
        # qry: B x Q x D
        shuffled = qry.permute(0, 2, 1)  # B x D x Q
        return torch.bmm(doc, shuffled)  # B x N x Q

    def gated_attention(self, doc, qry):

        word = doc.unsqueeze(1)
        context = qry.permute(1, 2, 0)
        inter = torch.bmm(word, context)
        # alphas_r = F.softmax(inter.view(-1, inter.size(-1))).view_as(inter)
        alphas_r = F.softmax(inter, dim=2)
        q_rep = torch.bmm(alphas_r, context.permute(0, 2, 1))
        np.savetxt("attn_weights.txt",alphas_r.squeeze(1))
        att = self.tmul(word, q_rep)

        return att.squeeze(1)


class CHLSTM(nn.Module):
    def __init__(self, n_ch_tokens, char_lstm_dim, ch_emb_size, device, ch_drop=0.1):
        super(CHLSTM, self).__init__()
        self.n_ch_tokens = n_ch_tokens
        self.ch_emb_size = ch_emb_size
        self.char_lstm_dim = char_lstm_dim
        self.ch_drop = nn.Dropout(ch_drop)
        self.device = device
        self.char_embedding = nn.Embedding(
            self.n_ch_tokens,
            self.ch_emb_size,
            padding_idx=constants.PAD_IDX
        )

        self.char_gru = nn.GRU(input_size=self.ch_emb_size,
                               hidden_size=self.char_lstm_dim,
                               dropout=0,
                               batch_first=True,
                               bidirectional=True)
        # (batch, seq_len, embed_dim)

        self.trans = nn.Linear(
            self.char_lstm_dim * 2, 300 // 2)

    def forward(self, x):
        x = x.permute(1, 0)
        x_embs = self.char_embedding(x)
        lengths = (x != constants.PAD_IDX).sum(dim=0).detach().cpu()
        char_len_sorted, idx_sort = np.sort(lengths.numpy())[::-1], np.argsort(-lengths.numpy())
        char_len_sorted = torch.from_numpy(char_len_sorted.copy())
        idx_unsort = np.argsort(idx_sort)
        x_embs = x_embs.index_select(1, torch.from_numpy(idx_sort).to(self.device))
        x_embs = pack(x_embs, char_len_sorted, batch_first=False)
        _, hn = self.char_gru(x_embs, None)
        emb = torch.cat((hn[0], hn[1]), 1)  # batch x 2*nhid
        emb = emb.index_select(0, torch.from_numpy(idx_unsort).to(self.device))
        emb = self.ch_drop(self.trans(emb))

        # _, hn = self.char_gru(x_embs)
        # emb = torch.cat((hn[0], hn[1]), 1)
        return emb

    def init_ch(self):
        initrange = 0.5 / self.ch_emb_size
        nn.init.uniform_(self.char_embedding.weight, -initrange, initrange)
        for name, param in self.char_gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.constant_(self.trans.bias, 0.0)
        nn.init.xavier_normal_(self.trans.weight)

