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
    parser.add_option('--lower', action="store_true", default=False,
                      help='Use lower cased input: default=%default')
    parser.add_option('--strip-accents', action="store_true", default=False,
                      help='Strip accents when tokenizing: default=%default')
    parser.add_option('--no-pos', action="store_true", default=False,
                      help='Use the targets with POS tags (for English only): default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    lower = options.lower
    strip_accents = options.strip_accents
    no_pos = options.no_pos

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    tokenized_dir = get_subdir(basedir, model_name, strip_accents)

    if lang == 'eng' and no_pos:
        lemma_index_file = os.path.join(basedir, 'clean_lemmas', 'target_indices_nopos.json')
    else:
        lemma_index_file = os.path.join(basedir, 'clean_lemmas', 'target_indices.json')

    with open(lemma_index_file) as f:
        lemma_indices = json.load(f)

    if lang == 'eng' and no_pos:
        index_file = os.path.join(tokenized_dir, 'target_indices_in_tokens_nopos.json')
    else:
        index_file = os.path.join(tokenized_dir, 'target_indices_in_tokens.json')

    tokenized_file =  os.path.join(tokenized_dir, 'all.jsonlist')

    with open(index_file) as f:
        indices = json.load(f)

    with open(tokenized_file) as f:
        lines = f.readlines()

    lines = [json.loads(line) for line in lines]
    lines_by_id = {line['id']: line for line in lines}
    source_by_id = {line['id']: line['source'] for line in lines}

    counter_by_source = defaultdict(Counter)

    for key, target_indices in indices.items():
        target_counter = Counter()
        for line_id, token_index in target_indices:
            try:
                token = lines_by_id[line_id]['tokens'][token_index]
                token = re.sub('##', '', token)
                target_counter[token] += 1
                counter_by_source[key][source_by_id[line_id]] += 1
            except KeyError as e:
                print(line_id)
                print(token_index)
                print(lines_by_id[line_id]['tokens'])
                raise e

        print(key)
        for k, c in target_counter.most_common():
            print(k, c)
        print(len(lemma_indices[key]), sum(target_counter.values()), '{:.3f}'.format(sum(target_counter.values())/len(lemma_indices[key])))
        print()

    print(np.mean([len(target_indices) for target_indices in indices.values()]), np.median([len(target_indices) for target_indices in indices.values()]))
    print(np.mean([min(counter_by_source[key].values()) for key in counter_by_source]), np.median([min(counter_by_source[key].values()) for key in counter_by_source]))

if __name__ == '__main__':
    main()
