#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 上午10:16
import json
import pickle
import math
import numpy as np
from . import constants
from torch.utils.data import Dataset
from utils.util import read_data


def pad(seq, size, value):
    if len(seq) < size:
        seq.extend([value] * (size - len(seq)))
    return seq

def create_mask(context_list):
    eg_mask=[]
    for i in range(len(context_list)):
        if context_list[i]!=constants.PAD_IDX:
            eg_mask.append(0)
        else:
            eg_mask.append(-float('inf'))
    return eg_mask


def pad_char_seq(seq, size, value, char_max_len):
    if len(seq) < size:
        seq.extend([[value] * char_max_len] * (size - len(seq)))
    return seq


def edit(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    return matrix[len(str1)][len(str2)]


def wordidx(context, word):
    dis = 999
    idx = len(context)
    for i in range(len(context)):
        if context[i] == word:
            return i
    for i in range(len(context)):
        if edit(context[i], word) < dis:
            dis = edit(context[i], word)
            idx = i
    return idx


class Vocabulary:
    """Word/char vocabulary"""

    def __init__(self):
        self.token2id = {
            constants.PAD: constants.PAD_IDX,
            constants.UNK: constants.UNK_IDX,
            constants.BOS: constants.BOS_IDX,
            constants.EOS: constants.EOS_IDX,
        }
        self.id2token = {
            constants.PAD_IDX: constants.PAD,
            constants.UNK_IDX: constants.UNK,
            constants.BOS_IDX: constants.BOS,
            constants.EOS_IDX: constants.EOS,
        }
        self.token_maxlen = -float("inf")

    def encode(self, tok):
        if tok in self.token2id:
            return self.token2id[tok]
        else:
            return constants.UNK_IDX

    def decode(self, idx):
        if idx in self.id2token:
            return self.id2token[idx]
        else:
            raise ValueError("No such idx: {0}".format(idx))

    def encode_seq(self, seq):
        e_seq = []
        for s in seq:
            e_seq.append(self.encode(s))
        return e_seq

    def decode_seq(self, seq):
        d_seq = []
        for i in seq:
            d_seq.append(self.decode(i))
        return d_seq

    def decode_char_seq(self, seq):
        c_seq = []
        for i in seq:
            c_seq.append(
                [constants.BOS_IDX] + \
                self.encode_seq(list(i)) + \
                [constants.EOS_IDX]
            )
        return c_seq

    def add_token(self, tok):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.token2id)
            self.id2token[len(self.id2token)] = tok

    def save(self, path):
        with open(path, "w") as outfile:
            json.dump([self.id2token, self.token_maxlen], outfile, indent=4)
        outfile.close()

    def load(self, path):
        with open(path, 'r') as infile:
            self.id2token, self.token_maxlen = json.load(infile)
            self.id2token = {int(k): v for k, v in self.id2token.items()}
            self.token2id = {}
            for i in self.id2token.keys():
                self.token2id[self.id2token[i]] = i

    def __len__(self):
        return len(self.token2id)


class DefinitionModelingDataset(Dataset):
    def __init__(self, file, vocab_path, label, input_vectors_path=None,
                 input_adaptive_vectors_path=None, context_vocab_path=None, ch_vocab_path=None,
                 use_seed=True, mode='train'):
        self.data = read_data(file)
        self.voc = Vocabulary()
        self.voc.load(vocab_path)
        if context_vocab_path is not None:
            self.context_voc = Vocabulary()
            self.context_voc.load(context_vocab_path)
        self.use_seed = use_seed
        self.label = label
        self.mode = mode
        assert self.label == "Definition-Generation" or self.label == "Usage-Generation" or \
               self.label == "Joint", \
            "task label only in Definition-Generation,Usage-Generation or Joint"
        assert self.mode == "train" or self.mode == "gen", "mode only in train or gen"
        if input_vectors_path is not None:
            with open(input_vectors_path, 'rb') as infile:
                self.input_vectors = pickle.load(infile)
        if input_adaptive_vectors_path is not None:
            self.input_adaptive_vectors = np.load(
                input_adaptive_vectors_path
            ).astype(np.float32)
        if ch_vocab_path is not None:
            self.ch_voc = Vocabulary()
            self.ch_voc.load(ch_vocab_path)

    def __getitem__(self, idx):
        if self.label == "Definition-Generation":
            inp = {
                "seq": self.voc.encode_seq([constants.BOS] + self.data[idx][1] + [constants.EOS]),
                "target": self.voc.encode_seq(self.data[idx][1] + [constants.EOS]),
                "word": self.voc.encode(self.data[idx][0]),
            }
        elif self.label == "Usage-Generation":
            inp = {
                "seq": self.voc.encode_seq([constants.BOS] + self.data[idx][3] + [constants.EOS]),
                "target": self.voc.encode_seq(self.data[idx][3] + [constants.EOS]),
                "word": self.voc.encode(self.data[idx][0]),
            }
        else:
            inp = {
                "seq": self.voc.encode_seq([constants.BOS] + self.data[idx][1] + [constants.EOS]),
                "target": self.voc.encode_seq(self.data[idx][1] + [constants.EOS]),
                "word": self.voc.encode(self.data[idx][0]),
                "usage": self.voc.encode_seq([constants.BOS] + self.data[idx][3] + [constants.EOS]),
                "usage_target": self.voc.encode_seq(self.data[idx][3] + [constants.EOS]),
            }
        if self.use_seed:
            inp['target'] = inp['seq'][1:]
            inp['seq'] = [self.voc.encode(self.data[idx][0])] + inp['seq'][1:]
            if self.label == "Joint":
                inp["usage_target"] = inp["usage_target"][1:]
                inp['usage'] = [self.voc.encode(self.data[idx][0])] + inp['usage'][1:]
        if self.mode == "gen":
            inp["seq"] = [inp["seq"][0]]
            if self.label == "Joint":
                inp["usage"] = [inp["usage"][0]]
        if hasattr(self, "context_voc"):
            inp["context_word"] = self.context_voc.encode(self.data[idx][0])
            inp["context"] = self.context_voc.encode_seq(self.data[idx][2])
            inp["context_text"] = self.data[idx][2]
            inp["word_id"] = wordidx(self.data[idx][2], self.data[idx][0])
        if hasattr(self, "ch_voc"):
            inp['chars'] = [constants.BOS_IDX] + \
                           self.ch_voc.encode_seq(list(self.data[idx][0])) + \
                           [constants.EOS_IDX]
            # CH_maxlen: +2 because EOS + BOS
            inp["CH_maxlen"] = self.ch_voc.token_maxlen + 2
        if hasattr(self, "input_vectors"):
            inp['input'] = self.input_vectors[idx]
        if hasattr(self, "input_adaptive_vectors"):
            inp["input_adaptive"] = self.input_adaptive_vectors[idx]
        inp["text"] = self.voc.decode_seq(inp["seq"])
        return inp
    # def __getitem__(self, idx):
    #     if self.label == "Definition-Generation":
    #         inp = {
    #             "seq": self.voc.encode_seq(self.data[idx][1]),
    #             "target": self.voc.encode_seq(self.data[idx][1] + [constants.EOS])[1:],
    #             "word": self.voc.encode(self.data[idx][0]),
    #         }
    #     elif self.label == "Usage-Generation":
    #         inp = {
    #             "seq": self.voc.encode_seq(self.data[idx][3]),
    #             "target": self.voc.encode_seq(self.data[idx][3] + [constants.EOS])[1:],
    #             "word": self.voc.encode(self.data[idx][0]),
    #         }
    #     else:
    #         inp = {
    #             "seq": self.voc.encode_seq(self.data[idx][1]),
    #             "target": self.voc.encode_seq(self.data[idx][1] + [constants.EOS])[1:],
    #             "word": self.voc.encode(self.data[idx][0]),
    #             "usage": self.voc.encode_seq(self.data[idx][3]),
    #             "usage_target": self.voc.encode_seq(self.data[idx][3] + [constants.EOS])[1:],
    #         }
    #     if self.use_seed:
    #         inp['target'] = inp['seq'] + self.voc.encode_seq([constants.EOS])
    #         inp['seq'] = [self.voc.encode(self.data[idx][0])] + inp['seq']
    #         if self.label == "Joint":
    #             inp["usage_target"] = inp["usage"]+ self.voc.encode_seq([constants.EOS])
    #             inp['usage'] = [self.voc.encode(self.data[idx][0])] + inp['usage']
    #     if self.mode == "gen":
    #         inp["seq"] = [inp["seq"][0]]
    #         if self.label == "Joint":
    #             inp["usage"] = [inp["usage"][0]]
    #     if hasattr(self, "context_voc"):
    #         inp["context_word"] = self.context_voc.encode(self.data[idx][0])
    #         inp["context"] = self.context_voc.encode_seq(self.data[idx][2])
    #         inp["context_text"] = self.data[idx][2]
    #         inp["word_id"] = wordidx(self.data[idx][2], self.data[idx][0])
    #     if hasattr(self, "ch_voc"):
    #         inp['chars'] = [constants.BOS_IDX] + \
    #                        self.ch_voc.encode_seq(list(self.data[idx][0])) + \
    #                        [constants.EOS_IDX]
    #         # CH_maxlen: +2 because EOS + BOS
    #         inp["CH_maxlen"] = self.ch_voc.token_maxlen + 2
    #     if hasattr(self, "input_vectors"):
    #         inp['input'] = self.input_vectors[idx]
    #     if hasattr(self, "input_adaptive_vectors"):
    #         inp["input_adaptive"] = self.input_adaptive_vectors[idx]
    #     inp["text"] = self.voc.decode_seq(inp["seq"])
    #     return inp

    def __len__(self):
        return len(self.data)


def DefinitionModelingCollate(batch):
    batch_word = []
    batch_x = []
    batch_y = []
    batch_text = []

    is_joint = "usage" in batch[0]
    is_ch = "chars" in batch[0] and "CH_maxlen" in batch[0]
    is_input = "input" in batch[0]
    is_ada = "input_adaptive" in batch[0]
    is_attn = "context_word" in batch[0] and "context" in batch[0]
    if is_joint:
        batch_usage = []
        batch_usage_target = []
    if is_ch:
        batch_ch = []
        CH_maxlen = batch[0]["CH_maxlen"]
    if is_input:
        batch_input = []
    if is_ada:
        batch_input_adaptive = []
    if is_attn:
        batch_context_word = []
        batch_context = []
        batch_context_text = []
        batch_word_id = []
        batch_context_mask=[]
        context_maxlen = -float("inf")

    seq_lengths = []
    if is_joint:
        usage_lengths = []
    for i in range(len(batch)):
        batch_x.append(batch[i]["seq"])
        batch_y.append(batch[i]["target"])
        batch_word.append(batch[i]["word"])
        batch_text.append(batch[i]["text"])
        if is_joint:
            batch_usage.append(batch[i]["usage"])
            batch_usage_target.append(batch[i]["usage_target"])
            usage_lengths.append(len(batch_usage[-1]))
        if is_ch:
            batch_ch.append(batch[i]["chars"])
        if is_input:
            batch_input.append(batch[i]["input"])
        if is_ada:
            batch_input_adaptive.append(batch[i]["input_adaptive"])
        if is_attn:
            batch_context_word.append(batch[i]["context_word"])
            batch_context.append(batch[i]["context"])
            batch_context_text.append(batch[i]["context_text"])
            batch_word_id.append(batch[i]["word_id"])
            context_maxlen = max(context_maxlen, len(batch_context[-1]))
        seq_lengths.append(len(batch_x[-1]))
    seq_maxlen = max(seq_lengths)
    if is_joint:
        usage_maxlen = max(usage_lengths)
        seq_maxlen = max(seq_maxlen, usage_maxlen)

    for i in range(len(batch)):
        batch_x[i] = pad(batch_x[i], seq_maxlen, constants.PAD_IDX)
        batch_y[i] = pad(batch_y[i], seq_maxlen, constants.PAD_IDX)
        batch_text[i] = pad(batch_text[i], seq_maxlen, constants.PAD)
        if is_joint:
            batch_usage[i] = pad(batch_usage[i], seq_maxlen, constants.PAD_IDX)
            batch_usage_target[i] = pad(batch_usage_target[i], seq_maxlen, constants.PAD_IDX)
        if is_attn:
            batch_context[i] = pad(
                batch_context[i], context_maxlen, constants.PAD_IDX
            )
            batch_context_mask.append(create_mask(batch_context[i]))
        if is_ch:
            batch_ch[i] = pad(batch_ch[i], CH_maxlen, constants.PAD_IDX)
    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)
    batch_word = np.array(batch_word)
    ret_batch = {
        "word": batch_word,
        "seq": batch_x,
        "target": batch_y,
        "text": batch_text,
    }
    if is_joint:
        batch_usage = np.array(batch_usage)
        ret_batch["usage"] = batch_usage
        batch_usage_target = np.array(batch_usage_target)
        ret_batch["usage_target"] = batch_usage_target
    if is_ch:
        batch_ch = np.array(batch_ch)
        ret_batch["chars"] = batch_ch
    if is_input:
        batch_input = np.array(batch_input)
        ret_batch["input"] = batch_input
    if is_ada:
        batch_input_adaptive = np.array(
            batch_input_adaptive,
            dtype=np.float32
        )
        ret_batch["input_adaptive"] = batch_input_adaptive
    if is_attn:
        batch_context_word = np.array(batch_context_word)
        batch_context = np.array(batch_context)
        ret_batch["context_word"] = batch_context_word
        ret_batch["context"] = batch_context
        ret_batch["context_text"] = batch_context_text
        ret_batch["word_id"] = batch_word_id
        ret_batch["eg_mask"]=np.array(batch_context_mask)
    return ret_batch


class LanguageModelingDataset(Dataset):
    """LanguageModeling dataset."""

    def __init__(self, file, vocab_path, bptt):
        """
        Args:
            file (string): Path to the file
            vocab_path (string): path to word vocab to use
            bptt (int): length of one sentence
        """
        with open(file, "r") as infile:
            self.data = infile.read().lower().split()
        self.voc = Vocabulary()
        self.voc.load(vocab_path)
        self.bptt = bptt

    def __len__(self):
        return math.ceil(len(self.data) / (self.bptt + 1))

    def __getitem__(self, idx):
        i = idx + self.bptt * idx
        sample = {
            "seq": self.voc.encode_seq(self.data[i: i + self.bptt]),
            "target": self.voc.encode_seq(self.data[i + 1: i + self.bptt + 1]),
        }
        return sample


def LanguageModelingCollate(batch):
    batch_x = []
    batch_y = []
    maxlen = -float("inf")
    for i in range(len(batch)):
        batch_x.append(batch[i]["seq"])
        batch_y.append(batch[i]["target"])
        maxlen = max(maxlen, len(batch[i]["seq"]), len(batch[i]["target"]))

    for i in range(len(batch)):
        batch_x[i] = pad(batch_x[i], maxlen, constants.PAD_IDX)
        batch_y[i] = pad(batch_y[i], maxlen, constants.PAD_IDX)

    ret_batch = {
        "seq": np.array(batch_x),
        "target": np.array(batch_y),
    }
    return ret_batch
