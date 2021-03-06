#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-21 上午10:18
import torch
import json
import argparse
from model.sense_generator import SenseGenerator
from model.sense_generator_share_embedding import SenseGenerator_ShareEmbedding
from model.sense_generator_hierarchical import SenseGenerator_Hierarchical
from torch.utils.data import DataLoader
from utils.datasets import DefinitionModelingDataset, DefinitionModelingCollate
from utils.pipeline import generate, beam_generate,generate_definitions,generate_usages,generate_batch
from utils.datasets import Vocabulary

parser = argparse.ArgumentParser(description='Script to generate using model')
parser.add_argument(
    "--params", type=str, required=True,
    help="path to saved model params"
)
parser.add_argument(
    "--ckpt", type=str, required=True,
    help="path to saved model weights"
)
# parser.add_argument(
#     "--generate_list", type=str, required=True,
#     help="path to word list to generate"
# )
parser.add_argument(
    "--strategy", type=str, default="Greedy",
    help="generate strategy(Greedy,Multinomial or Beam)."
)
parser.add_argument(
    "--tau", type=float, required=False,
    help="temperature to use in sampling"
)
parser.add_argument(
    "--beam_size", type=int, required=False,
    help="number of samples to generate if strategy is beam"
)
parser.add_argument(
    "--length", type=int, required=True,
    help="maximum length of generated samples"
)
parser.add_argument(
    "--gen_dir", type=str, default="gen/",
    help="where to save generate file"
)
parser.add_argument(
    "--gen_name", type=str, default="gen.txt",
    help="generate file name"
)
parser.add_argument(
    "--type", type=str, required=True,
    help="train, val or test set to evaluate on"
)
parser.add_argument(
    '--generate_usage', action='store_true',
    help='use CUDA'
)
args = parser.parse_args()
assert args.strategy in ["Greedy", "Multinomial", "Beam"], ("--type must be Greedy,Multinomial or Beam")
if args.strategy == "Multinomial":
    assert args.tau is not None, ("--strategy is Multinomial,"
                                  " --tau is required")
if args.strategy == "Beam":
    assert args.beam_size is not None, ("--strategy is Beam,"
                                        " --beam_size is required")
with open(args.params, "r") as infile:
    model_params = json.load(infile)
dataset = DefinitionModelingDataset(
    file=model_params["test_defs"],
    vocab_path=model_params["voc"],
    label="Definition-Generation" if args.type == "single" else "Joint",
    input_adaptive_vectors_path=model_params["input_adaptive_test"] if args.type == "single" else None,
    context_vocab_path=model_params["context_voc"],
    ch_vocab_path=model_params["ch_voc"],
    use_seed=model_params["use_seed"],
    mode="gen"
)
dataloader = DataLoader(
    dataset,
    batch_size=3,
    collate_fn=DefinitionModelingCollate,
    num_workers=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.type == "multi":
    if model_params["type"] == "same_level":
        model = SenseGenerator_ShareEmbedding(model_params).to(device)
    elif model_params["type"] == "hir_level":
        model = SenseGenerator_Hierarchical(model_params).to(device)
else:
    model = SenseGenerator(model_params).to(device)
model.load_state_dict(torch.load(args.ckpt))
voc = Vocabulary()
voc.load(model_params["voc"])
context_voc = Vocabulary()
context_voc.load(model_params["context_voc"])

if __name__ == "__main__":
    print('=========model architecture==========')
    print(model)
    print('=============== end =================')
    if args.type=="multi":
        generate_definitions(model, dataloader, voc, context_voc, tau=args.tau, length=args.length, device=device, save=args.gen_dir,
            strategy=args.strategy)
        if args.generate_usage:
            generate_usages(model, dataloader, voc, context_voc, tau=args.tau, length=args.length, device=device, save=args.gen_dir,
            strategy=args.strategy)
    else:
        generate_batch(
            model, dataloader, voc, context_voc, tau=args.tau, length=args.length, device=device, save=args.gen_dir,
            strategy=args.strategy
        )
    print("Finished!")
