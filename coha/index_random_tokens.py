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


# Get a collection of random token indices


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                     help='model used for tokenization: default=%default')
    parser.add_option('--targets-file', type=str, default=None,
                     help='Targets file: default=%default')
    parser.add_option('--max-tokens', type=int, default=4000,
                      help='Limit the size of the index by pre-sampling: default=%default')
    parser.add_option('--n-terms', type=int, default=10000,
                      help='Number of words to sample: default=%default')
    parser.add_option('--min-count', type=int, default=9,
                      help='Restrict token selection to those with at least this many tokens: default=%default')
    parser.add_option('--seed', type=int, default=42,
                     help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    targets_file = options.targets_file
    max_tokens = options.max_tokens
    n_terms = options.n_terms
    min_count = options.min_count
    seed = options.seed
    np.random.seed(seed)

    model_name = get_model_name(model)

    # avoid re-indexing target words (if provided)
    targets = set()
    if targets_file is not None:            
        with open(targets_file) as f:
            target_lines = f.readlines()
        targets.update([line.strip().split('\t')[0] for line in target_lines])

    tokenized_dir = get_subdir(basedir, model_name)

    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')

    with open(tokenized_file) as f:
        lines = f.readlines()
    print("Tokenized lines:", len(lines))
    lines = [json.loads(line) for line in lines]

    print("Counting tokens")
    token_counter = Counter()
    for line_i, line in tqdm(enumerate(lines)):
        tokens = line['tokens']
        tokens = [re.sub('##', '', token) for token in tokens]
        for token in tokens:
            if token not in targets:        
                token_counter[token] += 1

    subset = Counter({key: count for key, count in token_counter.items() if count >= min_count and len(key) > 1 and '/' not in key and re.match('.*[a-zA-Z0-9].*', key) is not None})

    print("Most common tokens:")
    for term, count in subset.most_common(10):
        print(term, count)
    print(len(subset))

    if len(subset) > n_terms:        
        targets_subset = set(np.random.choice(list(subset.keys()), size=n_terms, replace=False))
        subset = Counter({key: count for key, count in subset.items() if key in targets_subset})

    print("Most common remainming:")
    for term, count in subset.most_common(10):
        print(term, count)
    print(len(subset))

    print("Indexing tokens")
    token_indices = defaultdict(list)
    for line_i, line in tqdm(enumerate(lines)):
        line_id = line['id']
        tokens = line['tokens']
        tokens = [re.sub('##', '', token) for token in tokens]
        for t_i, token in enumerate(tokens):
            if token in subset:
                token_indices[token].append((line_id, t_i))

    print(len(token_indices))

    print("Taking random sample of indices:")
    random_token_indices = defaultdict(list)
    for token, pairs in token_indices.items():
        if len(pairs) > max_tokens:
            random_indices = np.random.choice(np.arange(len(pairs)), size=max_tokens, replace=False)
            random_token_indices[token] = [pairs[i] for i in random_indices]
        else:
            random_token_indices[token] = pairs

    print(min([len(pairs) for pairs in random_token_indices.values()]))
    print(max([len(pairs) for pairs in random_token_indices.values()]))

    print("Saving indices")
    outfile = os.path.join(tokenized_dir, 'random_token_indices.json')

    with open(outfile, 'w') as f:
        json.dump(random_token_indices, f, indent=2)


if __name__ == '__main__':
    main()
