#!/usr/bin/env bash
python ../pre_vocab.py \
--defs ../data/oxford/train_us.txt ../data/oxford/valid_us.txt ../data/oxford/test_us.txt \
--save ../data/oxford/processed/vocab.json \
--save_chars ../data/oxford/processed/char_vocab.json \
--save_context ../data/oxford/processed/context_vocab.json \

python ../pre_wordemb.py \
--vocab ../data/oxford/processed/vocab.json \
--w2v ../data/word2vec/GoogleNews-vectors-negative300.bin \
--save ../data/oxford/processed/def_embedding \

python ../pre_wordemb.py \
--vocab ../data/oxford/processed/context_vocab.json \
--w2v ../data/word2vec/GoogleNews-vectors-negative300.bin \
--save ../data/oxford/processed/context_embedding \

echo "Preparing vocab and embedding finished! "