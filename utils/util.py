#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 下午2:19
import logging
import sys
import time
import codecs
import numpy as np


from collections import defaultdict
from utils import constants
from datetime import timedelta


def get_time_dif(start_time):
    """Compute time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def read_data(file_path):
    """Read definitions file"""
    content = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip().split('|||')
            defs = []
            definition = line[3].strip().split(" ")
            for d in definition:
                defs.append(d)
            contexts = []
            context = line[-1].strip().split(" ")
            for c in context:
                contexts.append(c)
            usages = []
            usage = line[4].strip().split(" ")
            for u in usage:
                usages.append(u)
            content.append([line[0], defs, contexts, usages])
    return content

def read_data(file_path):
    """Read definitions file"""
    content = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            defs = []
            definition = line[1].strip().split(" ")
            for d in definition:
                defs.append(d)
            contexts = []
            context = line[-1].strip().split(" ")
            for c in context:
                contexts.append(c)
            content.append([line[0], defs, contexts])
    return content

def to_char(seq, char_vocab, max_len):
    char_seq_list = []
    for word_list in seq:
        char_list = []
        for word in word_list:
            if word == constants.BOS:
                chars = [constants.BOS_IDX]
            elif word == constants.EOS:
                chars = [constants.EOS_IDX]
            elif word == constants.UNK:
                chars = [constants.UNK_IDX]
            elif word == constants.PAD:
                chars = [constants.PAD_IDX]
            else:
                chars = [char_vocab.encode(c) for c in list(word)]
                chars.insert(0, constants.BOS_IDX)
                chars.append(constants.EOS_IDX)
            for k in range(0, max_len - len(chars)):
                chars.append(constants.PAD_IDX)
            char_list.append(chars)
        char_seq_list.append(char_list)
    return np.array(char_seq_list)


def read_hypernyms(file_path):
    """Read hypernyms file"""
    hyp_token = []
    hnym_data = defaultdict(list)
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split('\t')
            word = line[0]
            hyp_token.append(word)
            line = line[1:]
            assert len(line) % 2 == 0
            for i in range(int(len(line) / 2)):
                hnym = line[2 * i]
                hyp_token.append(hnym)
                weight = line[2 * i + 1]
                hnym_data[word].append([hnym, weight])
    return hnym_data, hyp_token


class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()



def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

if __name__ == "__main__":
    a = read_hypernyms("../data/Wn_Gcide/bag_of_hypernyms.txt")
    print()
