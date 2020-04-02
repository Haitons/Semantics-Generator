#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-11 下午11:30
import torch
from torch import nn
import torch.nn.functional as F
from model.layers import Hidden, Gated
from regularize.locked_dropout import LockedDropout
from regularize.weight_dropout import WeightDrop


class GruDecoder(nn.Module):
    def __init__(self, params, embedding_dim, cond_size=None):
        super(GruDecoder, self).__init__()
        self.params = params
        self.embedding_dim = embedding_dim
        self.cond_size = cond_size

        self.hdrop = nn.Dropout(self.params["dropouth"])
        self.dropouth = self.params["dropouth"]
        self.idrop = nn.Dropout(self.params["dropouti"])
        self.dropouti = self.params["dropouti"]
        self.lockdrop = LockedDropout()

        self.use_input = self.params["use_input"]
        self.use_hidden = self.params["use_hidden"]
        self.use_gated = self.params["use_gated"]
        self.use_ch = self.params["use_ch"]
        self.use_elmo = self.params["use_elmo"]
        self.is_ada = self.params["use_input_adaptive"]
        self.is_attn = self.params["use_input_attention"]
        self.gate = self.params["att_gate"]

        if self.params["rnn_type"] in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.params["rnn_type"])(self.embedding_dim, self.params["hidim"],
                                                            self.params["nlayers"], dropout=0)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.params["rnn_type"]]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                               options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.embedding_dim, self.params["hidim"], self.params["nlayers"],
                              nonlinearity=nonlinearity, dropout=0)
        if self.params["wdrop"] != 0:
            self.rnn = WeightDrop(self.rnn, ['weight_hh_l0'], dropout=self.params["wdrop"])
        if self.use_hidden:
            self.hidden = Hidden(
                in_size=self.cond_size + self.params["hidim"],
                out_size=self.params["hidim"]
            )
        if self.use_gated:
            self.gated = Gated(
                cond_size=self.cond_size,
                hidden_size=self.params["hidim"]
            )
        if self.is_attn:
            if self.gate:
                self.att_gate = nn.Linear(self.embedding_dim, self.embedding_dim)

        self.init_weights()

    def forward(self, data, init_hidden, return_h=False):
        hidden = init_hidden
        seq_emb = data["seq_emb"]
        batch_size = seq_emb.size(1)
        seq_emb = self.lockdrop(seq_emb, self.dropouti)
        raw_outputs = []
        lock_outputs = []
        outputs = []
        for time_step in range(seq_emb.size(0)):
            if time_step != 0:
                raw_outputs = []
                lock_outputs = []
            inp_seq = seq_emb[time_step, :, :].view(1, batch_size, -1)
            if self.use_input:
                inp_seq = torch.cat([torch.unsqueeze(data["word_emb"], 0), inp_seq], dim=-1)
            elif self.is_ada:
                inp_seq = torch.cat([torch.unsqueeze(data["input_adaptive"], 0), inp_seq], dim=-1)
            elif self.is_attn:
                inp_seq = torch.cat([torch.unsqueeze(data["att"], 0), inp_seq], dim=-1)
                if self.use_ch:
                    inp_seq=torch.cat([torch.unsqueeze(data["chars"], 0),inp_seq],dim=-1)
                if self.use_elmo:
                    inp_seq = torch.cat([torch.unsqueeze(data["elmo"], 0), inp_seq], dim=-1)
                if self.gate:
                    g = F.sigmoid(self.att_gate(inp_seq))
                    inp_seq = g * inp_seq
            outs, hidden = self.rnn(inp_seq, hidden)
            if self.use_hidden:
                outs = self.hidden(self.params["rnn_type"], data["word_emb"], outs)
            if self.use_gated:
                outs = self.gated(self.params["rnn_type"], data["word_emb"], outs)
            raw_outputs.append(outs)
            outs = self.lockdrop(outs, self.dropouth)
            lock_outputs.append(outs)
            if time_step == 0:
                rnn_hs = raw_outputs
                dropped_rnn_hs = lock_outputs
            else:
                for i in range(len(rnn_hs)):
                    rnn_hs[i] = torch.cat((rnn_hs[i], raw_outputs[i]), 0)
                    dropped_rnn_hs[i] = torch.cat((dropped_rnn_hs[i], lock_outputs[i]), 0)
            outputs.append(outs)
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.view(outputs.size(0) * outputs.size(1), outputs.size(2))
        if return_h:
            return outputs, hidden, rnn_hs, dropped_rnn_hs
        return outputs, hidden

    def init_weights(self):
        if self.params["pretrain"]:
            self.init_rnn()
        else:
            if self.params["lm_ckpt"] is not None:
                lm_ckpt_weights = torch.load(self.params["lm_ckpt"])
                self.init_rnn_from_pretrained(lm_ckpt_weights)
            else:
                self.init_rnn()
                if self.params["att_gate"]:
                    nn.init.constant_(self.att_gate.bias, 0.0)
                    nn.init.xavier_normal_(self.att_gate.weight)
            if self.use_hidden:
                self.hidden.init_hidden()
            if self.use_gated:
                self.gated.init_gated()

    def init_rnn(self):
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def init_rnn_from_pretrained(self, weights):
        # k[4:] because we need to remove prefix "rnn." because
        # self.rnn.state_dict() is without "rnn." prefix
        correct_state_dict = {
            k[8:]: v for k, v in weights.items() if k[:8] == "gru.rnn."
        }
        # also we need to correctly initialize weight_ih_l0
        # with pretrained weights because it has different size with
        # self.rnn.state_dict(), other weights has correct shapes if
        # hidden sizes have same shape as in the LM pretraining
        if self.use_input or self.is_ada or self.is_attn:
            w = torch.empty(3 * self.params["hidim"], self.embedding_dim)
            nn.init.xavier_uniform_(w)
            w[:, self.cond_size:] = correct_state_dict["weight_ih_l0"]
            correct_state_dict["weight_ih_l0"] = w
        self.rnn.load_state_dict(correct_state_dict)
