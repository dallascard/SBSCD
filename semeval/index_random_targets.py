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
    parser.add_option('--max-tokens', type=int, default=4000,
                      help='Limit the size of the index by pre-sampling: default=%default')
    parser.add_option('--n-terms', type=int, default=10000,
                      help='Number of words to sample: default=%default')
    parser.add_option('--seed', type=int, default=42,
                     help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    strip_accents = options.strip_accents
    max_tokens = options.max_tokens
    n_terms = options.n_terms
    seed = options.seed
    np.random.seed(seed)

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    lemmas_dir = os.path.join(basedir, 'clean_lemmas')
    tokenized_dir = get_subdir(basedir, model_name, strip_accents)

    if lang == 'eng':
        target_lemmas_file = os.path.join(lemmas_dir, 'target_indices_nopos.json')
    else:
        target_lemmas_file = os.path.join(lemmas_dir, 'target_indices.json')
    
    with open(target_lemmas_file) as f:
        target_indices = json.load(f)

    orig_target_lemmas = set(target_indices.keys())

    # Set the minimum count for random token inclusion to be half the min target lemma count
    min_target_lemma_count = min([len(indices) for indices in target_indices.values()])
    min_count = min_target_lemma_count // 2
    print("Setting min count to be", min_count)

    match_file =  os.path.join(tokenized_dir, 'matching.jsonlist')
    print("Matching file:", match_file)

    with open(match_file) as f:
        lines = f.readlines()
    print("Matched lines:", len(lines))
    lines = [json.loads(line) for line in lines]

    print("Counting tokens")
    token_counter_by_lemma = defaultdict(Counter)
    for line_i, line in tqdm(enumerate(lines)):
        tokens = line['tokens'].split()
        lemmas = line['lemmas'].split()
        alignment = line['alignment']
        for pair in alignment:
            token_index, lemma_index = pair
            if lemma_index != -1 and token_index != -1:
                try:
                    lemma = lemmas[lemma_index]
                except IndexError as e:
                    print(lemmas)
                    print(lemma_index)
                    raise e
                if lemma not in orig_target_lemmas:
                    token_counter_by_lemma[lemma][tokens[token_index]] += 1

    subset = {key: counter for key, counter in token_counter_by_lemma.items() if sum(counter.values()) >= min_count and len(key) > 1 and '/' not in key and re.match('.*[a-zA-Z0-9].*', key) is not None}
    print(len(subset))

    print("Filtering subsets for valid tokens")
    filtered_subset = defaultdict(Counter)
    valid_tokens_by_lemma = defaultdict(set)
    for lemma, token_counter in tqdm(subset.items()):
        total = sum(token_counter.values())
        lemma_lower = lemma.lower()
        for token, count in token_counter.items():
            token_lower = token.lower()
            if count / total >= 0.0002 and len(token) > 1:
                if token_lower[0] == lemma[0].lower():
                    valid_tokens_by_lemma[lemma].add(token)
                elif lang == 'lat':
                    if lemma_lower.startswith('j') and token_lower.startswith('i'):
                        valid_tokens_by_lemma[lemma].add(token)
                    elif lemma_lower.startswith('v') and token_lower.startswith('u'):
                        valid_tokens_by_lemma[lemma].add(token)
                elif lang == 'ger':
                    if lemma_lower.startswith('ü') and token_lower.startswith('u'):
                        valid_tokens_by_lemma[lemma].add(token)
                    elif lemma_lower.startswith('u') and token_lower.startswith('ü'):
                        valid_tokens_by_lemma[lemma].add(token)

        token_counter = Counter({token: count for token, count in token_counter.items() if token in valid_tokens_by_lemma[lemma]})
        filtered_subset[lemma] = token_counter

    print(len(filtered_subset))
    print("Filtering for lengthe once more")
    filtered_subset = {key: counter for key, counter in filtered_subset.items() if sum(counter.values()) >= min_count}
    print("Filtered", len(filtered_subset))

    if len(filtered_subset) > n_terms:
        target_lemmas = set(np.random.choice(list(filtered_subset.keys()), size=n_terms, replace=False))
    else:
        target_lemmas = set(filtered_subset.keys())

    print(len(target_lemmas))

    print("Indexing tokens")
    token_indices_by_lemma = defaultdict(list)
    for line_i, line in tqdm(enumerate(lines)):
        line_id = line['id']
        tokens = line['tokens'].split()
        lemmas = line['lemmas'].split()
        alignment = line['alignment']
        for pair in alignment:
            token_index, lemma_index = pair
            if lemma_index != -1 and token_index != -1 and lemma:
                token = tokens[token_index]
                lemma = lemmas[lemma_index]
                if lemma in target_lemmas and token in valid_tokens_by_lemma[lemma]:
                    token_indices_by_lemma[lemma].append((line_id, token_index))
    print(len(token_indices_by_lemma))

    print("Taking random sample of indices:")
    random_token_indices_by_lemma = defaultdict(list)
    for lemma in target_lemmas:
        token_indices = token_indices_by_lemma[lemma]
        if len(token_indices) > max_tokens:
            random_indices = np.random.choice(np.arange(len(token_indices)), size=max_tokens, replace=False)
            random_token_indices_by_lemma[lemma] = [token_indices[i] for i in random_indices]
        else:
            random_token_indices_by_lemma[lemma] = token_indices

    print(min(len(indices) for indices in random_token_indices_by_lemma.values()))
    print(max(len(indices) for indices in random_token_indices_by_lemma.values()))

    print("Saving indices")
    outfile = os.path.join(tokenized_dir, 'random_indices_in_tokens.json')

    with open(outfile, 'w') as f:
        json.dump(random_token_indices_by_lemma, f, indent=2)


if __name__ == '__main__':
    main()
