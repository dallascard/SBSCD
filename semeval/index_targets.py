import os
import re
import json
import string
from glob import glob
from optparse import OptionParser
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.misc import get_model_name, get_subdir


# Convert the target lemma indices to token indices based on do_alignment


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base directory: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                     help='model used for tokenization: default=%default')
    parser.add_option('--strip-accents', action="store_true", default=False,
                      help='Strip accents when tokenizing: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    strip_accents = options.strip_accents

    no_pos = False
    if lang == 'eng':
        no_pos = True

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    lemmas_dir = os.path.join(basedir, 'clean_lemmas')
    tokenized_dir = get_subdir(basedir, model_name, strip_accents)

    index_file = os.path.join(lemmas_dir, 'target_indices.json')
    if lang == 'eng':
        index_file_nopos = os.path.join(lemmas_dir, 'target_indices_nopos.json')

    match_file =  os.path.join(tokenized_dir, 'matching.jsonlist')
    print("Matching file:", match_file)
    
    if lang == 'eng' and no_pos:
        with open(index_file_nopos) as f:
            indices = json.load(f)
    else:
        with open(index_file) as f:
            indices = json.load(f)

    with open(match_file) as f:
        lines = f.readlines()
    print("Matched lines:", len(lines))

    print("Indexing lines by ID")
    lines = [json.loads(line) for line in lines]
    lines_by_id = {line['id']: line for line in lines}


    target_counter = Counter()
    target_indices_in_tokens = defaultdict(list)    

    valid_tokens_per_lemma = defaultdict(Counter)

    for key, values in indices.items():        
        lemma_counter = Counter()
        token_counter = Counter()
        for line_id, lemma_index in values:
            line = lines_by_id[line_id]
            tokens = line['tokens'].split()
            lemmas = line['lemmas'].split()
            alignment = line['alignment']
            mapping = {pair[1]: pair[0] for pair in alignment}
            lemma_counter[lemmas[lemma_index]] += 1
            token_index = mapping[lemma_index]
            if token_index == -1:
                token_counter['[NA]'] += 1
            else:
                token_counter[tokens[token_index]] += 1

        total = sum(token_counter.values())
        valid_tokens = set()
        for token, count in token_counter.items():
            token_lower = token.lower()
            if count / total >= 0.0002 and len(token) > 1:
                if token_lower[0] == key[0].lower():
                    valid_tokens.add(token)
                elif lang == 'lat':
                    if key == 'jus' and token_lower.startswith('i'):
                        valid_tokens.add(token)
                    elif key.startswith('v') and token_lower.startswith('u'):
                        valid_tokens.add(token)
                elif lang == 'ger':
                    if key == 'packen' and token_lower.startswith('g'):
                        valid_tokens.add(token)
                    elif key == 'Kubikmeter' and token_lower.startswith('c'):
                        valid_tokens.add(token)
                    elif key == 'überspannen' and token_lower.startswith('u'):
                        valid_tokens.add(token)
                elif lang == 'swe':
                    if key == 'kokärt' and token_lower.startswith('ä'):
                        valid_tokens.add(token)
              
        for line_id, lemma_index in values:
            line = lines_by_id[line_id]
            tokens = line['tokens'].split()
            lemmas = line['lemmas'].split()
            alignment = line['alignment']
            mapping = {pair[1]: pair[0] for pair in alignment}
            token_index = mapping[lemma_index]
            if token_index >= 0:
                token = tokens[token_index]
                if token in valid_tokens:
                    target_indices_in_tokens[key].append((line_id, token_index))
                    valid_tokens_per_lemma[key][token] += 1
        print(key, len(target_indices_in_tokens[key]), len(target_indices_in_tokens[key]) / len(values))
        target_counter[key] = len(target_indices_in_tokens[key])

    for target, count in target_counter.most_common():
        print(target, count)   

    if no_pos:
        outfile = os.path.join(tokenized_dir, 'target_indices_in_tokens_nopos.json')
    else:
        outfile = os.path.join(tokenized_dir, 'target_indices_in_tokens.json')

    with open(outfile, 'w') as f:
        json.dump(target_indices_in_tokens, f, indent=2)

    outfile = os.path.join(tokenized_dir, 'valid_tokens_per_lemma.json')
    with open(outfile, 'w') as f:
        json.dump(valid_tokens_per_lemma, f, indent=2)


if __name__ == '__main__':
    main()
