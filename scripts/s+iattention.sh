#!/usr/bin/env bash
--voc
./data/oxford/processed/vocab.json
--train_defs
./data/oxford/1.txt
--eval_defs
./data/oxford/2.txt
--test_defs
./data/oxford/3.txt
--context_voc
./data/oxford/processed/context_vocab.json
--emdim
300
--nlayers
2
--hidim
300
--lr
0.001
--decay_factor
0.1
--decay_patience
0
--epochs
16
--batch_size
11
--clip
5
--random_seed
42
--exp_dir
./checkpoints/
--cuda
--use_seed
--w2v_weights
./data/oxford/processed/def_embedding
--use_input_attention
--att_type
Input
--n_attn_embsize
300
--n_attn_hid
300
--attn_dropout
0.1