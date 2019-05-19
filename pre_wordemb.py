#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-14 下午7:14
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from gensim.models.keyedvectors import KeyedVectors
from utils.datasets import Vocabulary
from utils.util import get_time_dif

parser = argparse.ArgumentParser(description="Prepare word embeddings for model")
parser.add_argument(
    '--vocab', type=str, required=True,
    help="preprocessed vocabulary json file."
)
parser.add_argument(
    '--w2v', type=str, required=True,
    help="location of w2v binary file"
)
parser.add_argument(
    '--save', type=str, default="../data/processed/word_embedding.pkl",
    help="where to save word embeddings"
)
parser.add_argument(
    '--emb_dim', type=int, default=300,
    help="word embedding dims")
args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    print('Start prepare word embeddings at {}'.format(time.asctime(time.localtime(start_time))))
    vocab = Vocabulary()
    vocab.load(args.vocab)
    word2vec = KeyedVectors.load_word2vec_format(args.w2v, binary=True)
    init_embedding = np.random.uniform(-1.0, 1.0, (len(vocab), args.emb_dim))
    for word in tqdm(vocab.token2id.keys()):
        if word in word2vec:
            init_embedding[vocab.encode(word)] = word2vec[word]
    init_embedding[vocab.encode('<pad>')] = np.zeros([args.emb_dim])
    with open(args.save, 'wb') as f:
        pickle.dump(init_embedding, f)
        f.close()
    time_dif = get_time_dif(start_time)
    print("Finished!Prepare word embeddings time usage:", time_dif)
