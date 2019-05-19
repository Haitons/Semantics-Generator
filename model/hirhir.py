#!/usr/bin/env python3
#-*- coding:utf-8 -*-
# author:haiton
# datetime:19-5-3 下午12:04


#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:19-3-29 下午4:28
import torch
import torch.nn as nn
import pickle
import numpy as np
from model.module import GruDecoder
from allennlp.modules.elmo import Elmo, batch_to_ids
from model.layers import CharCNN, InputAttention, ScaledDotAttention
from regularize.embed_dropout import embedded_dropout


class SenseGenerator_Hierarchical(nn.Module):
    def __init__(self, params):
        super(SenseGenerator_Hierarchical, self).__init__()

        self.params = params
        self.embedding = nn.Embedding(self.params["vocab_size"], self.params["emdim"])
        self.embedding_dim = self.params["emdim"]

        self.drop = nn.Dropout(self.params["dropout"])
        self.edrop = nn.Dropout(self.params["dropoute"])
        self.dropoute = self.params["dropoute"]

        self.use_ch = self.params["use_ch"]
        self.use_elmo = self.params["use_elmo"]
        self.is_attn = self.params["use_input_attention"]

        self.device = torch.device('cuda' if self.params["cuda"] else 'cpu')
        self.ws = self.params["intial_hidden"]
        self.cond_size = self.params["emdim"]
        # ch
        if self.use_ch:
            self.ch = CharCNN(
                n_ch_tokens=self.params["n_ch_tokens"],
                ch_maxlen=self.params["ch_maxlen"],
                ch_emb_size=self.params["ch_emb_size"],
                ch_feature_maps=self.params["ch_feature_maps"],
                ch_kernel_sizes=self.params["ch_kernel_sizes"]
            )
            self.cond_size += sum(self.params["ch_feature_maps"])
        # elmo
        if self.use_elmo:
            options_file = self.params["options_file"]
            weight_file = self.params["weight_file"]
            self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
            self.cond_size += 1024
        self.hs_size = self.cond_size

        if self.is_attn:
            if self.params["att_type"] == "ScaledDot":
                self.scaleddot_attention = ScaledDotAttention(
                    n_attn_tokens=self.params["n_attn_tokens"],
                    n_attn_embsize=self.params["n_attn_embsize"],
                    n_attn_hid=self.params["n_attn_hid"],
                    attn_dropout=self.params["attn_dropout"],
                    freeze=self.params["fix_attn_embeddings"],
                    device=torch.device('cuda' if self.params["cuda"] else 'cpu'),
                    pretrain_w2v=self.params["att_w2v"],
                )
                self.embedding_dim += self.params["n_attn_hid"] * 2
                self.cond_size = self.params["n_attn_hid"] * 2
            elif self.params["att_type"] == "Input":
                self.input_attention = InputAttention(
                    n_attn_tokens=self.params["n_attn_tokens"],
                    n_attn_embsize=self.params["n_attn_embsize"],
                    n_attn_hid=self.params["n_attn_hid"],
                    attn_dropout=self.params["attn_dropout"],
                    sparse=self.params["attn_sparse"]
                )
                self.embedding_dim += self.params["n_attn_embsize"]
                self.cond_size = self.params["n_attn_embsize"]
        if self.ws == "word":
            self.hs = nn.Linear(self.hs_size, self.params["hidim"])
        elif self.is_attn:
            if self.ws == "word+contexts" or self.ws == "word+attention":
                self.hs = nn.Linear(
                    self.hs_size + self.params["n_attn_hid"] * 2, self.params["hidim"]
                )
            elif self.ws == "context" or self.ws == "attention":
                self.hs = nn.Linear(self.params["n_attn_hid"] * 2, self.params["hidim"])
        self.decoder = nn.Linear(self.params["hidim"], self.params["vocab_size"])
        if self.params["tied"]:
            if self.params["hidim"] != self.embedding_dim:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.embedding.weight
        self.task1 = GruDecoder(params, self.embedding_dim, self.cond_size)
        self.task2 = GruDecoder(params, self.embedding_dim + self.params["hidim"],
                                self.cond_size + self.params["hidim"])
        self.init_weights()

    def forward(self, inputs, init_hidden1, init_hidden2, return_h=False):
        seq = inputs["seq"]
        seq_emb = embedded_dropout(self.embedding, seq, dropout=self.dropoute if self.training else 0)
        data = {
            'seq_emb': seq_emb
        }
        word = inputs['word']
        word_emb = self.embedding(word)
        batch_size = seq.size(1)
        if self.is_attn:
            if self.params["att_type"] == "ScaledDot":
                context_attention = self.scaleddot_attention(
                    inputs["context_word"], inputs["context"]
                )
                att = context_attention[0]
                context = context_attention[1]
            else:
                context, att = self.input_attention(
                    inputs["context_word"], torch.transpose(inputs["context"], 0, 1)
                )
            data["att"] = att
        if self.use_ch:
            char_embeddings = self.ch(inputs['chars'])
            word_emb = torch.cat(
                [word_emb, char_embeddings], dim=-1)
        if self.use_elmo:
            character_ids = batch_to_ids(inputs["context_text"]).to(self.device)
            embeddings = self.elmo(character_ids)
            elmo_embedding = embeddings['elmo_representations'][0]
            word_id = torch.LongTensor(inputs["word_id"]).view(-1, 1)
            one_hot = torch.zeros(elmo_embedding.size(0), elmo_embedding.size(1))
            one_hot = one_hot.scatter_(1, word_id, 1).unsqueeze(1)
            elmo_word_embedding = torch.bmm(one_hot.to(self.device), elmo_embedding).squeeze(1)
            word_emb = torch.cat(
                [word_emb, elmo_word_embedding],
                dim=-1
            )
        data["word_emb"] = word_emb
        if init_hidden1 is not None:
            hidden1 = init_hidden1
        else:
            hidden1 = None
            if self.ws == "word":
                hidden1 = self.init_hidden(
                    word_emb, batch_size, self.params["nlayers"], self.params["hidim"]
                )
            elif self.ws == "word+contexts":
                hidden1 = self.init_hidden(
                    word_emb, batch_size, self.params["nlayers"], self.params["hidim"], context
                )
            elif self.ws == "word+attention":
                hidden1 = self.init_hidden(
                    word_emb, batch_size, self.params["nlayers"], self.params["hidim"], att
                )
            elif self.ws == "context":
                hidden1 = self.init_hidden(
                    context, batch_size, self.params["nlayers"], self.params["hidim"]
                )
            elif self.ws == "attention":
                hidden1 = self.init_hidden(
                    att, batch_size, self.params["nlayers"], self.params["hidim"]
                )
        if init_hidden2 is not None:
            hidden2 = init_hidden2
        else:
            hidden2 = None
            if self.ws == "word":
                hidden2 = self.init_hidden(
                    word_emb, batch_size, self.params["nlayers"], self.params["hidim"]
                )
            elif self.ws == "word+contexts":
                hidden2 = self.init_hidden(
                    word_emb, batch_size, self.params["nlayers"], self.params["hidim"], context
                )
            elif self.ws == "word+attention":
                hidden2 = self.init_hidden(
                    word_emb, batch_size, self.params["nlayers"], self.params["hidim"], att
                )
            elif self.ws == "context":
                hidden2 = self.init_hidden(
                    context, batch_size, self.params["nlayers"], self.params["hidim"]
                )
            elif self.ws == "attention":
                hidden2 = self.init_hidden(
                    att, batch_size, self.params["nlayers"], self.params["hidim"]
                )
        task1_output = self.task1(data, hidden1, return_h)
        outputs1 = task1_output[0]
        hidden1 = task1_output[1]
        decoded1 = self.decoder(self.drop(outputs1))
        # Task 2
        # seq = inputs["usage"]
        # seq_emb = embedded_dropout(self.embedding, seq, dropout=self.dropoute if self.training else 0)
        # outputs = outputs1.view(data['seq_emb'].size(0), data['seq_emb'].size(1), -1)
        # data["seq_emb"] = torch.cat([seq_emb, outputs], -1)
        # task2_output = self.task2(data, hidden2, return_h)
        # outputs2 = task2_output[0]
        # hidden2 = task2_output[1]
        # decoded2 = self.decoder(self.drop(outputs2))

        seq = inputs["usage"]
        seq_emb = embedded_dropout(self.embedding, seq, dropout=self.dropoute if self.training else 0)
        data["seq_emb"] = seq_emb
        task1_output = self.task1(data, None, return_h)
        outputs1 = task1_output[0]
        outputs = outputs1.view(data['seq_emb'].size(0), data['seq_emb'].size(1), -1)
        data["seq_emb"] = torch.cat([seq_emb, outputs], -1)
        task2_output = self.task2(data, hidden2, return_h)
        outputs2 = task2_output[0]
        hidden2 = task2_output[1]
        decoded2 = self.decoder(self.drop(outputs2))
        if return_h:
            rnn_hs1 = task1_output[2]
            dropped_rnn_hs1 = task1_output[3]
            rnn_hs2 = task2_output[2]
            dropped_rnn_hs2 = task2_output[3]
            return decoded1, hidden1, decoded2, hidden2, rnn_hs1, dropped_rnn_hs1, rnn_hs2, dropped_rnn_hs2
        return decoded1, hidden1, decoded2, hidden2

    def init_hidden(self, v, batch_size, num_layers, hidden_dim, feature=None):
        if feature is not None:
            v = torch.cat([v, feature], dim=-1)
        hidden = self.hs(v).view(-1, batch_size, hidden_dim)
        hidden = hidden.expand(num_layers, batch_size, hidden_dim).contiguous()
        if self.params["rnn_type"] == 'LSTM':
            h_c = hidden
            h_h = torch.zeros_like(h_c)
            hidden = (h_h, h_c)
        return hidden

    def init_weights(self):
        if self.params["lm_ckpt"] is not None:
            lm_ckpt_weights = torch.load(self.params["lm_ckpt"])
            self.init_embeddings_from_pretrained(
                lm_ckpt_weights["embedding.weight"],
                self.params["fix_embeddings"]
            )
            self.init_linear_from_pretrained(lm_ckpt_weights)
        else:
            if self.params["w2v_weights"]:
                with open(self.params["w2v_weights"], 'rb') as infile:
                    pretrain_emb = pickle.load(infile)
                    infile.close()
                self.embedding.weight.data.copy_(
                    torch.from_numpy(pretrain_emb)
                )
            else:
                self.embedding.weight.data.copy_(
                    torch.from_numpy(
                        self.random_embedding(self.params["vocab_size"], self.embedding_dim)
                    )
                )
            self.embedding.weight.requires_grad = not self.params["fix_embeddings"]
            self.init_linear()
            if self.ws != "None":
                nn.init.constant_(self.hs.bias, 0.0)
                nn.init.xavier_normal_(self.hs.weight)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for i in range(vocab_size):
            pretrain_emb[i, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def init_embeddings_from_pretrained(self, weights, freeze):
        self.embedding = self.embedding.from_pretrained(weights, freeze)

    def init_linear(self):
        nn.init.constant_(self.decoder.bias, 0.0)
        nn.init.xavier_normal_(self.decoder.weight)

    def init_linear_from_pretrained(self, weights):
        # k[8: ] because we need to remove prefix "decoder." because
        # self.decoder.state_dict() is without "decoder." prefix
        self.decoder.load_state_dict(
            {k[8:]: v for k, v in weights.items() if k[:8] == "decoder."}
        )
