#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-9-26 下午2:27
import argparse
from rouge import Rouge
from itertools import islice

parser = argparse.ArgumentParser(description='Script to compute BLEU')
parser.add_argument(
    "--ref", type=str, required=True,
    help="path to file with references"
)
parser.add_argument(
    "--hyp", type=str, required=True,
    help="path to file with hypotheses"
)
args = parser.parse_args()


def next_n_lines(file_opened, N):
    return [x.strip() for x in islice(file_opened, N)]


def read_def_file(file, n):
    defs = []
    while True:
        lines = next_n_lines(file, n + 2)
        if len(lines) == 0:
            break
        assert len(lines) == n + 2, "Something bad in hyps file"
        definition = lines[2].split("Definition:")[1].strip()
        defs.append(definition)
    return defs


def read_ref_file(file):
    defs = []
    while True:
        lines = next_n_lines(file, 3)
        if len(lines) == 0:
            break
        assert len(lines) == 3, "Something bad in refs file"
        definition = lines[2].split("Definition:")[1].strip()

        defs.append(definition)
    return defs


with open(args.ref) as ifp:
    refs = read_ref_file(ifp)
with open(args.hyp) as ifp:
    hyps = read_def_file(ifp, 1)

assert len(refs) == len(hyps), "Number of words being defined mismatched!"

rouge = Rouge()
score = 0
count = 0
total_refs = 0
total_hyps = 0
for hypothesis, reference in zip(hyps, refs):
    total_hyps += 1
    total_refs += 1
    scores = rouge.get_scores(hypothesis, reference)
    score += scores[0]["rouge-l"]["f"]
    count += 1
print("ROUGE-L: ", score / count)
print("NUM HYPS USED: ", total_hyps)
print("NUM REFS USED: ", total_refs)
