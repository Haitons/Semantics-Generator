#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:haiton
# datetime:18-11-30 上午10:22
import json
import torch
import numpy as np
import os
import torch.nn.functional as F
from utils import constants
from scipy.stats import bernoulli
from utils.beamsearch import BeamSearch
from utils.util import to_char
from tqdm import tqdm
from torch import nn


def train_epoch(epoch, dataloader, model, loss_fn, optimizer, device, clip_to, ac_re=False, alpha=0, beta=0):
    """
    Function for training the model one epoch
        epoch - training epoch
        dataloader - DefinitionModeling dataloader
        model - DefinitionModelingModel
        loss_fn - loss function to use
        optimizer - optimizer to use (usually Adam)
        device - cuda/cpu
        clip_to - value to clip gradients
    """
    # switch model to training mode
    model.train()
    # train
    loss_epoch = []
    task1_loss_epoch = []
    task2_loss_epoch = []
    inputs = []
    num_batches = int(len(dataloader[-1].dataset) / dataloader[-1].batch_size) + 1
    for i in range(len(dataloader)):
        inputs.append(iter(dataloader[i]))
    turn_list = bernoulli.rvs(0.5, size=num_batches)
    for batch, turn in enumerate(tqdm(turn_list, desc='Epoch: %03d' % (epoch + 1), leave=False)):
        if turn == 1:
            main_task = True
        else:
            main_task = False
        inp = next(inputs[turn])
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
        }
        data["word"] = torch.from_numpy(
            inp['word']
        ).to(device)
        if model.is_attn:
            data["context_word"] = torch.from_numpy(
                inp['context_word']
            ).to(device)
            data["context"] = torch.t(torch.from_numpy(
                inp["context"]
            )).to(device)
        if model.use_ch:
            data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
        if model.use_elmo:
            data["context_text"] = inp["context_text"]
            data["word_id"] = inp["word_id"]
        targets = torch.t(torch.from_numpy(inp['target'])).to(device)
        if ac_re:
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, main_task,return_h=ac_re)
        else:
            output, hidden = model(data, None, main_task,return_h=ac_re)
        loss = loss_fn(output, targets.contiguous().view(-1))
        optimizer.zero_grad()
        if ac_re:
            # Activiation Regularization
            loss = loss + sum(
                alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()
        # `clip_grad_norm`
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_to)
        optimizer.step()
        loss_epoch.append(loss.item())
        if main_task:
            task2_loss_epoch.append(loss.item())
        else:
            task1_loss_epoch.append(loss.item())
    train_loss = np.mean(loss_epoch)
    train_ppl = np.exp(train_loss)
    task1_loss = np.mean(task1_loss_epoch)
    task1_ppl = np.exp(task1_loss)
    task2_loss = np.mean(task2_loss_epoch)
    task2_ppl = np.exp(task2_loss)
    return train_loss, train_ppl, task1_loss, task1_ppl, task2_loss, task2_ppl


def train_shunxu(epoch, dataloader, model, loss_fn, optimizer, device, clip_to, ac_re=False,alpha=0, beta=0):
    """
    Function for training the model one epoch
        epoch - training epoch
        dataloader - DefinitionModeling dataloader
        model - DefinitionModelingModel
        loss_fn - loss function to use
        optimizer - optimizer to use (usually Adam)
        device - cuda/cpu
        clip_to - value to clip gradients
    """
    # switch model to training mode
    model.train()
    # train
    loss_epoch = []
    task1_loss_epoch = []
    task2_loss_epoch = []
    for i in range(len(dataloader)):
        if i == (len(dataloader) - 1):
            main_task = True
        else:
            main_task = False
        for inp in tqdm(dataloader[i], desc='Epoch: %03d' % (epoch + 1), leave=False):
            data = {
                'seq': torch.t(torch.from_numpy(
                    inp['seq'])
                ).long().to(device),
            }
            data["word"] = torch.from_numpy(
                inp['word']
            ).to(device)
            if model.is_attn:
                data["context_word"] = torch.from_numpy(
                    inp['context_word']
                ).to(device)
                data["context"] = torch.t(torch.from_numpy(
                    inp["context"]
                )).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            if model.use_elmo:
                data["context_text"] = inp["context_text"]
                data["word_id"] = inp["word_id"]
            targets = torch.t(torch.from_numpy(inp['target'])).to(device)
            if ac_re:
                output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, main_task, return_h=ac_re)
            else:
                output, hidden = model(data, None, main_task, return_h=ac_re)
            loss = loss_fn(output, targets.contiguous().view(-1))
            optimizer.zero_grad()
            if ac_re:
                # Activiation Regularization
                loss = loss + sum(
                    alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss.backward()
            # `clip_grad_norm`
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_to)
            optimizer.step()
            loss_epoch.append(loss.item())
            if main_task:
                task2_loss_epoch.append(loss.item())
            else:
                task1_loss_epoch.append(loss.item())
    train_loss = np.mean(loss_epoch)
    train_ppl = np.exp(train_loss)
    task1_loss = np.mean(task1_loss_epoch)
    task1_ppl = np.exp(task1_loss)
    task2_loss = np.mean(task2_loss_epoch)
    task2_ppl = np.exp(task2_loss)
    return train_loss, train_ppl, task1_loss, task1_ppl, task2_loss, task2_ppl


def test(model, label_type, dataloader, device):
    """
    Function for testing the model on dataloader
        dataloader - DefinitionModeling dataloader
        model - DefinitionModelingModel
        device - cuda/cpu
    """
    # switch model to evaluation mode
    model.eval()
    # eval
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        loss1_epoch = []
        loss2_epoch = []
        for i in range(len(label_type)):
            if i == (len(label_type) - 1):
                main_task = True
            else:
                main_task = False
            for inp in tqdm(dataloader[i], desc='Evaluate model in definitions.', leave=False):
                data = {
                    'seq': torch.t(torch.from_numpy(
                        inp['seq'])
                    ).long().to(device),
                }
                data["word"] = torch.from_numpy(
                    inp['word']
                ).to(device)
                if model.is_attn:
                    data["context_word"] = torch.from_numpy(
                        inp['context_word']
                    ).to(device)
                    data["context"] = torch.t(torch.from_numpy(
                        inp["context"]
                    )).to(device)
                if model.use_ch:
                    data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
                if model.use_elmo:
                    data["context_text"] = inp["context_text"]
                    data["word_id"] = inp["word_id"]
                targets = torch.t(torch.from_numpy(inp['target'])).to(device)
                output, hidden = model(data, None, main_task)
                loss = loss_fn(output, targets.contiguous().view(-1))
                if main_task:
                    loss2_epoch.append(loss.item())
                else:
                    loss1_epoch.append(loss.item())
    return np.mean(loss1_epoch), np.exp(np.mean(loss1_epoch)), np.mean(loss2_epoch), np.exp(np.mean(loss2_epoch))


def generate(model, label_type, dataloader, voc, context_voc, tau, length, device, save, strategy="Greedy"):
    """
    model - DefinitionModelingModel
    voc - model Vocabulary
    tau - temperature to generate with
    n - number of samples
    length - length of the sample
    device - cuda/cpu
    prefix - prefix to read until generation
    input - vectors for Input/InputAdaptive conditioning
    word - word for InputAttention conditioning
    context - context for InputAttention conditioning
    context_voc - Vocabulary for InputAttention conditioning
    """
    model.eval()
    if not os.path.exists(save):
        os.makedirs(save)

    for i in range(len(label_type)):
        defsave = open(
            save + label_type[i]+".txt",
            "w"
        )
        if i == (len(label_type) - 1):
            main_task = True
        else:
            main_task = False
        for inp in tqdm(dataloader[i], desc='Generate definitions.', leave=False):
            data = {
                'seq': torch.t(torch.from_numpy(
                    inp['seq'])
                ).long().to(device),
            }
            data["word"] = torch.from_numpy(
                inp['word']
            ).to(device)
            if model.is_attn:
                data["context_word"] = torch.from_numpy(
                    inp['context_word']
                ).to(device)
                data["context"] = torch.t(torch.from_numpy(
                    inp["context"]
                )).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            if model.use_elmo:
                data["context_text"] = inp["context_text"]
                data["word_id"] = inp["word_id"]
            def_word = voc.id2token[inp['word'][0]]
            context = context_voc.decode_seq(inp["context"][0])
            defsave.write("Word:" + def_word + "\n")
            defsave.write("Context:")
            defsave.write(" ".join(context) + "\n")
            defsave.write("Definition:")
            hidden = None
            ret = []
            with torch.no_grad():
                for j in range(length):
                    output, hidden = model(data, hidden, main_task)
                    if strategy == "Greedy":
                        word_weights = output.squeeze().div(1).exp().cpu()
                        # word_idx = torch.argmax(word_weights)
                        top_word = torch.topk(word_weights, 2)
                        word_idx = top_word[1][0] if top_word[1][0] != inp["word"][0].item() else top_word[1][1]
                    elif strategy == "Multinomial":
                        word_weights = F.softmax(
                            output / tau, dim=1
                        ).multinomial(num_samples=1)
                        word_idx = word_weights[0][0]
                        # word_weights = F.softmax(
                        #     output / tau, dim=1
                        # ).multinomial(num_samples=2)
                        # word_idx=word_weights[0][0] if word_weights[0][0]!=inp["word"][0].item() else word_weights[0][1]
                    if word_idx == constants.EOS_IDX:
                        break
                    else:
                        data['seq'].fill_(word_idx)
                        word = word_idx.item()
                        ret.append(voc.decode(word))
                output = " ".join(map(str, ret))
                defsave.write(output + "\n")


def beam_generate(model, dataloader, voc, beamsize, device, output_save_path):
    model.eval()
    if not os.path.exists(output_save_path):
        os.makedirs(output_save_path)
    beam = BeamSearch(
        40,
        constants.UNK_IDX,
        constants.BOS_IDX,
        constants.EOS_IDX,
        beam_size=beamsize)
    output_lines = {}
    output_scores = {}
    with torch.no_grad():
        for inp in tqdm(dataloader, desc='Generate Definitions', leave=False):
            data = {
                'word': torch.from_numpy(inp['word']).to(device),
                'seq': torch.t(torch.from_numpy(inp["seq"])).to(device)
            }
            if model.use_input:
                data["input_vectors"] = torch.from_numpy(inp['input']).to(device)
            if model.use_ch:
                data["chars"] = torch.from_numpy(inp['chars']).long().to(device)
            if model.use_he:
                data["hypm"] = torch.from_numpy(inp['hypm']).long().to(device)
                data["hypm_weights"] = torch.from_numpy(inp['hypm_weights']).float().to(
                    device)
            beam.reset()
            hidden = None
            probs, hidden = model(data, hidden)
            probs = probs.detach().cpu().numpy()
            while beam.beam(probs):
                data['seq'] = torch.tensor(
                    beam.live_samples, dtype=torch.long).to(device)
                data['seq'] = data['seq'][:, -1].expand(1, -1)
                data['word'] = data['word'][0].repeat(data['seq'].shape[1]).long().to(device)
                data['chars'] = data['chars'][0].expand(1, -1).repeat(
                    data['seq'].shape[1], 1)
                data['hypm'] = data['hypm'][0].expand(1, -1).repeat(
                    data['seq'].shape[1], 1)
                data['hypm_weights'] = data['hypm_weights'][0].expand(1, -1).repeat(
                    data['seq'].shape[1], 1)
                stack = []
                for i in range(model.n_layers):
                    hidden_layer = hidden[i][0].expand(1, -1).repeat(data['seq'].shape[1], 1)
                    stack.append(hidden_layer)
                hidden = torch.stack(stack, 0)
                probs, hidden = model(data, hidden)
                probs = probs.detach().cpu().numpy()
            line = [[voc.decode(i) for i in line if i not in [0, 2, 3]]
                    for line in beam.output]
            line = [' '.join(line) for line in line]
            word = voc.decode(int(data['word'][-1]))
            output_lines[word] = line
            output_scores[word] = beam.output_scores
    with open(output_save_path + '/output_lines.js', 'w') as fw_lines:
        fw_lines.write(json.dumps(output_lines))
    with open(output_save_path + '/output_scores.js', 'w') as fw_scores:
        fw_scores.write(json.dumps(output_scores))
