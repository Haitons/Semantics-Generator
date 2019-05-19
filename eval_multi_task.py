#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 下午2:40
import argparse
import torch
import json
from model.sense_generator_share_embedding import SenseGenerator
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils.pipeline_multi_taslk import test
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Script to evaluate model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
parser.add_argument(
    "--datasplit", type=str, required=True,
    help="train, val or test set to evaluate on"
)

args = parser.parse_args()
assert args.datasplit in ["train", "val", "test"], ("--datasplit must be "
                                                    "train, val or test")

with open(args.params, "r") as infile:
    model_params = json.load(infile)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SenseGenerator(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt))
dataset=[]
dataloader=[]
for i in range(len(model_params["label_type"])):
    if args.datasplit == "train":
        dataset.append(DefinitionModelingDataset(
            file=model_params["train_defs"],
            vocab_path=model_params["voc"],
            label=model_params["label_type"][i],
            input_vectors_path=model_params["input_train"],
            input_adaptive_vectors_path=model_params["input_adaptive_train"],
            context_vocab_path=model_params["context_voc"],
            ch_vocab_path=model_params["ch_voc"],
            use_seed=model_params["use_seed"],
            hypm_path=model_params["hypm_train"]
        ))
    elif args.datasplit == "val":
        dataset.append(DefinitionModelingDataset(
            file=model_params["eval_defs"],
            vocab_path=model_params["voc"],
            label=model_params["label_type"][i],
            input_vectors_path=model_params["input_eval"],
            input_adaptive_vectors_path=model_params["input_adaptive_eval"],
            context_vocab_path=model_params["context_voc"],
            ch_vocab_path=model_params["ch_voc"],
            use_seed=model_params["use_seed"],
            hypm_path=model_params["hypm_eval"]
        ))
    elif args.datasplit == "test":
        dataset.append(DefinitionModelingDataset(
            file=model_params["test_defs"],
            vocab_path=model_params["voc"],
            label=model_params["label_type"][i],
            input_vectors_path=model_params["input_test"],
            input_adaptive_vectors_path=model_params["input_adaptive_test"],
            context_vocab_path=model_params["context_voc"],
            ch_vocab_path=model_params["ch_voc"],
            use_seed=model_params["use_seed"],
            hypm_path=model_params["hypm_test"],
        ))
    dataloader.append(DataLoader(
        dataset[i],
        model_params["batch_size"],
        collate_fn=DefinitionModelingCollate,
        num_workers=2,
        shuffle=True
    ))

if __name__ == '__main__':
    print('=========model architecture==========')
    print(model)
    print('=============== end =================')
    loss1, ppl1, loss2, ppl2 = test(model, model_params["label_type"],dataloader, device)
    task1_msg = 'Task1: {0:>6},Test Loss: {1:>6.6}, Test Ppl: {2:>6.6}'
    task2_msg = 'Task2: {0:>6},Test Loss: {1:>6.6}, Test Ppl: {2:>6.6}'
    print(task1_msg.format(model_params["label_type"][0], loss1, ppl1) + "\n")
    print(task2_msg.format(model_params["label_type"][1], loss2, ppl2) + "\n")
