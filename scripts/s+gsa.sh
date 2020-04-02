#!/usr/bin/env bash
python ../train.py \
--voc ../data/oxford/processed/vocab.json \
--train_defs ../data/oxford/train_us.txt \
--eval_defs ../data/oxford/valid_us.txt \
--test_defs ../data/oxford/test.txt \
--context_voc ../data/oxford/processed/context_vocab.json \
--emdim 300 \
--nlayers 2 \
--hidim 300 \
--lr 0.001 \
--decay_factor 0.1 \
--decay_patience 0 \
--epochs 16 \
--batch_size 11 \
--clip 5 \
--random_seed 42 \
--exp_dir ../checkpoints/ \
--cuda \
--w2v_weights ../data/oxford/processed/def_embedding \
--use_input_attention \
--att_type ScaledDot \
--att_gate \
--n_attn_embsize 300 \
--n_attn_hid 300 \
--attn_dropou 0.1 \
--use_elmo \
--options_file ../data/elmo/2x4096_512_2048cnn_2xhighway_options.json \
--weight_file ../data/elmo/2x4096_512_2048cnn_2xhighway_weights.hdf5 \
--use_ch \
--ch_voc ../data/oxford/processed/char_vocab.json \
--ch_emb_size 64 \
--ch_feature_maps 10 30 40 40 40 \
--ch_kernel_sizes 2 3 4 5 6 \
--intial_hidden word+contexts \
--att_w2v ../data/oxford/processed/context_embedding