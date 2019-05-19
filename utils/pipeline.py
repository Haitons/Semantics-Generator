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
    for batch, inp in enumerate(tqdm(dataloader, desc='Epoch: %03d' % (epoch + 1), leave=False)):
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
        }
        if not model.params["pretrain"]:
            data["word"] = torch.from_numpy(
                inp['word']
            ).to(device)
            if model.is_ada:
                data["input"] = torch.from_numpy(
                    inp["input_adaptive"]
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
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, None, return_h=ac_re)
        else:
            output, hidden = model(data, None, return_h=ac_re)
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
    train_loss = np.mean(loss_epoch)
    train_ppl = np.exp(train_loss)
    return train_loss, train_ppl


def test(model, dataloader, device):
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
        total_loss = []
        for inp in tqdm(dataloader, desc='Evaluate model in definitions.', leave=False):
            data = {
                'seq': torch.t(torch.from_numpy(
                    inp['seq'])
                ).long().to(device),
            }
            if not model.params["pretrain"]:
                data["word"] = torch.from_numpy(
                    inp['word']
                ).to(device)
                if model.is_ada:
                    data["input"] = torch.from_numpy(
                        inp["input_adaptive"]
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
            output, hidden = model(data, None)
            loss = loss_fn(output, targets.contiguous().view(-1))
            total_loss.append(loss.item())
    return np.mean(total_loss), np.exp(np.mean(total_loss))


def generate(model, dataloader, voc, context_voc, tau, length, device, save, strategy="Greedy"):
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
    defsave = open(
        save + "gen.txt",
        "w"
    )
    for inp in tqdm(dataloader, desc='Generate definitions.', leave=False):
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
        }
        if not model.params["pretrain"]:
            data["word"] = torch.from_numpy(
                inp['word']
            ).to(device)
            if model.is_ada:
                data["input"] = torch.from_numpy(
                    inp["input_adaptive"]
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
        setence_list = []
        with torch.no_grad():
            for i in range(length):
                output, hidden = model(data, hidden)
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
            np.savetxt("weight.txt", setence_list)

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


def train_epoch_multi(epoch, dataloader, model, loss_fn, optimizer, device, clip_to, ac_re=False, alpha=0, beta=0):
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
    loss1_epoch = []
    loss2_epoch = []
    for batch, inp in enumerate(tqdm(dataloader, desc='Epoch: %03d' % (epoch + 1), leave=False)):
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
            'usage': torch.t(torch.from_numpy(
                inp['usage'])
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
        usage_targets = torch.t(torch.from_numpy(inp['usage_target'])).to(device)
        if ac_re:
            output1, hidden1, output2, hidden2, \
            rnn_hs1, dropped_rnn_hs1, rnn_hs2, dropped_rnn_hs2 = model(
                data, None, None, return_h=ac_re
            )
        else:
            output1, hidden1, output2, hidden2 = model(data, None, None, return_h=ac_re)
        loss1 = loss_fn(output1, targets.contiguous().view(-1))
        loss2 = loss_fn(output2, usage_targets.contiguous().view(-1))
        optimizer.zero_grad()
        if ac_re:
            # Activiation Regularization
            loss1 = loss1 + sum(
                alpha * dropped_rnn_h1.pow(2).mean() for dropped_rnn_h1 in dropped_rnn_hs1[-1:])
            loss2 = loss2 + sum(
                alpha * dropped_rnn_h2.pow(2).mean() for dropped_rnn_h2 in dropped_rnn_hs2[-1:])
            # Temporal Activation Regularization (slowness)
            loss1 = loss1 + sum(beta * (rnn_h1[1:] - rnn_h1[:-1]).pow(2).mean() for rnn_h1 in rnn_hs1[-1:])
            loss2 = loss2 + sum(beta * (rnn_h2[1:] - rnn_h2[:-1]).pow(2).mean() for rnn_h2 in rnn_hs2[-1:])
        loss = loss1 + loss2
        loss.backward()
        # `clip_grad_norm`
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_to)
        optimizer.step()
        loss_epoch.append(loss.item())
        loss1_epoch.append(loss1.item())
        loss2_epoch.append(loss2.item())
    train_loss = np.mean(loss_epoch)
    train_ppl = np.exp(train_loss)
    train_loss1 = np.mean(loss1_epoch)
    train_ppl1 = np.exp(train_loss1)
    train_loss2 = np.mean(loss2_epoch)
    train_ppl2 = np.exp(train_loss2)
    return train_loss, train_ppl, train_loss1, train_ppl1, train_loss2, train_ppl2


def test_multi(model, dataloader, device):
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
        for inp in tqdm(dataloader, desc='Evaluate model in definitions.', leave=False):
            data = {
                'seq': torch.t(torch.from_numpy(
                    inp['seq'])
                ).long().to(device),
                'usage': torch.t(torch.from_numpy(
                    inp['usage'])
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
            usage_targets = torch.t(torch.from_numpy(inp['usage_target'])).to(device)
            # targets = torch.t(torch.from_numpy(inp['target'])).to(device)
            # usage_targets = torch.t(torch.from_numpy(inp['usage_target'])).to(device)
            output1, hidden1, output2, hidden2 = model(data, None, None)
            loss1 = loss_fn(output1, targets.contiguous().view(-1))
            loss2 = loss_fn(output2, usage_targets.contiguous().view(-1))
            loss1_epoch.append(loss1.item())
            loss2_epoch.append(loss2.item())
        valid_loss1 = np.mean(loss1_epoch)
        valid_ppl1 = np.exp(valid_loss1)
        valid_loss2 = np.mean(loss2_epoch)
        valid_ppl2 = np.exp(valid_loss2)
    return valid_loss1, valid_ppl1, valid_loss2, valid_ppl2


def generate_definitions(model, dataloader, voc, context_voc, tau, length, device, save, strategy="Greedy"):
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
    defsave = open(
        save + "def.txt",
        "w"
    )
    for inp in tqdm(dataloader, desc='Generate definitions.', leave=False):
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
            'usage': torch.t(torch.from_numpy(
                inp['usage'])
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

        hidden1 = None
        hidden2 = None
        ret = []
        with torch.no_grad():
            for i in range(length):
                output1, hidden1, output2, hidden2 = model(data, hidden1, hidden2)
                if strategy == "Greedy":
                    word_weights = output.squeeze().div(1).exp().cpu()
                    # word_idx = torch.argmax(word_weights)
                    top_word = torch.topk(word_weights, 2)
                    word_idx = top_word[1][0] if top_word[1][0] != inp["word"][0].item() else top_word[1][1]
                elif strategy == "Multinomial":
                    word_weights1 = F.softmax(
                        output1 / tau, dim=1
                    ).multinomial(num_samples=1)
                    word_idx1 = word_weights1[0][0]
                    word_weights2 = F.softmax(
                        output2 / tau, dim=1
                    ).multinomial(num_samples=1)
                    word_idx2 = word_weights2[0][0]
                    # word_weights = F.softmax(
                    #     output / tau, dim=1
                    # ).multinomial(num_samples=2)
                    # word_idx=word_weights[0][0] if word_weights[0][0]!=inp["word"][0].item() else word_weights[0][1]
                if word_idx1 == constants.EOS_IDX:
                    break
                else:
                    data['seq'].fill_(word_idx1)
                    data['usage'].fill_(word_idx2)
                    word = word_idx1.item()
                    ret.append(voc.decode(word))
            output = " ".join(map(str, ret))
            defsave.write(output + "\n")


def generate_usages(model, dataloader, voc, context_voc, tau, length, device, save, strategy="Greedy"):
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
    usgsave = open(
        save + "usg.txt",
        "w"
    )
    for inp in tqdm(dataloader, desc='Generate usages.', leave=False):
        data = {
            'seq': torch.t(torch.from_numpy(
                inp['seq'])
            ).long().to(device),
            'usage': torch.t(torch.from_numpy(
                inp['usage'])
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
        usgsave.write("Word:" + def_word + "\n")
        usgsave.write("Context:")
        usgsave.write(" ".join(context) + "\n")
        usgsave.write("Usage:")

        hidden1 = None
        hidden2 = None
        ret = []
        with torch.no_grad():
            for i in range(length):
                output1, hidden1, output2, hidden2 = model(data, hidden1, hidden2)
                if strategy == "Greedy":
                    word_weights = output.squeeze().div(1).exp().cpu()
                    # word_idx = torch.argmax(word_weights)
                    top_word = torch.topk(word_weights, 2)
                    word_idx = top_word[1][0] if top_word[1][0] != inp["word"][0].item() else top_word[1][1]
                elif strategy == "Multinomial":
                    word_weights1 = F.softmax(
                        output1 / tau, dim=1
                    ).multinomial(num_samples=1)
                    word_idx1 = word_weights1[0][0]
                    word_weights2 = F.softmax(
                        output2 / tau, dim=1
                    ).multinomial(num_samples=1)
                    word_idx2 = word_weights2[0][0]
                    # word_weights = F.softmax(
                    #     output / tau, dim=1
                    # ).multinomial(num_samples=2)
                    # word_idx=word_weights[0][0] if word_weights[0][0]!=inp["word"][0].item() else word_weights[0][1]
                if word_idx2 == constants.EOS_IDX:
                    break
                else:
                    data['seq'].fill_(word_idx1)
                    data['usage'].fill_(word_idx2)
                    word = word_idx2.item()

                    ret.append(voc.decode(word))
            output = " ".join(map(str, ret))
            usgsave.write(output + "\n")



