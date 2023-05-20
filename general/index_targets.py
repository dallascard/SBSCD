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


# Index target terms in the corpus


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                     help='model used for tokenization: default=%default')
    parser.add_option('--targets-file', type=str, default='targets.tsv',
                     help='Targets file: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    targets_file = options.targets_file

    model_name = get_model_name(model)

    tokenized_dir = get_subdir(basedir, model_name)
    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')
    with open(tokenized_file) as f:
        lines = f.readlines()

    with open(targets_file) as f:
        target_lines = f.readlines()
    targets = set([line.strip().split('\t')[0] for line in target_lines])

    target_counter = Counter()
    target_indices = defaultdict(list)    

    for line in tqdm(lines):
        line = json.loads(line)
        line_id = line['id']
        tokens = line['tokens']
        tokens = [re.sub('##', '', token) for token in tokens]        
        for t_i, token in enumerate(tokens):
            if token in targets:
                target_indices[token].append((line_id, t_i))
                target_counter[token] += 1

    targets_sorted = sorted(targets)    
    counts = [target_counter[target] for target in targets_sorted]
    order = np.argsort(counts)[::-1]
    for i in order:
        print(targets_sorted[i], counts[i])
    for target, count in target_counter.most_common():
        print(target, count)

    outfile = os.path.join(tokenized_dir, 'target_indices.json')
    with open(outfile, 'w') as f:
        json.dump(target_indices, f, indent=2)


if __name__ == '__main__':
    main()
