# -*- coding: utf-8 -*-
import json
import re
import operator
from collections import Counter
from collections import defaultdict
from nltk.util import ngrams

pattern = re.compile(r'(?P<words>.+) (or|,|, or|of|and) (?P=words)\b')


def clean_repeated(text, max_len=6, min_len=1):
    global pattern
    s = pattern.search(text)
    new_text = text
    if s is not None:
        new_text = text[0:s.start()] + s.groups()[0]
        if s.end() < len(text):
            new_text = new_text + text[s.end():]
    text = new_text
    tokens = text.split()
    for n in range(max_len, min_len - 1, -1):
        if n > len(tokens):
            continue
        done = False
        while not done:
            ngram = Counter(ngrams(tokens, n))
            p = sorted(ngram.items(), key=operator.itemgetter(1), reverse=True)
            for k in p:
                if k[1] == 1:
                    done = True
                    break
                r = list(k[0])
                pos = [(i, i + len(r)) for i in range(len(tokens))
                       if tokens[i:i + len(r)] == r]
                prev_end = -1
                r_start = -1
                r_end = -1
                for start, end in pos:
                    if start <= prev_end:
                        if r_start == -1:
                            r_start = prev_end
                        r_end = end
                    prev_end = end
                if r_end != -1:
                    tokens = tokens[:r_start] + tokens[r_end:]
                    done = False
                    break
                else:
                    done = True
    return ' '.join(tokens)


def read_definition_file(gen_def_file):
    with open(gen_def_file) as fr:
        defs = json.loads(fr.read())
    ndefs = 0
    for word, definition in defs.items():
        definition = [clean_repeated(clean_repeated(d)) for d in definition]
        defs[word] = definition
        ndefs += len(definition)
    return defs, ndefs


def compute_score(word, definition, score, function_words=None):
    if function_words is None:
        function_words = set()
    definition = definition.replace('<unk>', 'UNK')
    tokens = definition.split(' ')
    new_score = score / (len(tokens) + 1)
    unigrams = Counter(ngrams(tokens, 1))
    unigram_penalty = sum(unigrams.values()) / float(len(unigrams.keys()))
    self_ref_penalty = 1
    if word in tokens:
        self_ref_penalty = 5
    return new_score * unigram_penalty * self_ref_penalty


def rerank(gen_def_file, gen_def_scores, rank_write_file):
    function_word_file = './data/function_words.txt'
    function_words = set()
    with open(function_word_file) as ifp:
        for line in ifp:
            function_words.add(line.strip())
    print('Reading Definitions...')
    defs, ndefs = read_definition_file(gen_def_file)
    print(" - {} words being defined".format(len(defs)))
    print(" - {} definitions".format(ndefs))
    with open(gen_def_scores) as fr:
        scores = json.loads(fr.read())

    rank_defs = defaultdict(list)
    for word, definition in defs.items():
        score = scores[word]
        assert len(definition) == len(score)
        for i, d in enumerate(definition):
            s = score[i]
            new_score = compute_score(word, d, s, function_words)
            rank_defs[word].append((d, new_score))

    for w, d_s in rank_defs.items():
        d_s = sorted(d_s, key=lambda x: x[1])[:1]
        rank_defs[w] = [d for d, s in d_s]

    with open(rank_write_file, 'w') as fw:
        fw.write(json.dumps(rank_defs))
    return 1


if __name__ == '__main__':
    gen_def_file = './output/output_lines.js'
    gen_def_scores = './output/output_scores.js'
    rank_write_file = './output/rerank_output.js'
    if rerank(gen_def_file, gen_def_scores, rank_write_file):
        print('All Done. No Error.')
