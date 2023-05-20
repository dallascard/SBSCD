import os
import re
import json
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.misc import get_model_name, get_subdir


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base directory: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--target-terms', type=str, default=None,
                      help='Target term(s), comma-separated: default=%default')
    parser.add_option('--is-random', action="store_true", default=False,
                      help="Target it from random background terms: default=%default")
    parser.add_option('--no-pos', action="store_true", default=False,
                      help="Ignore part of speech tags (English only): default=%default")
    parser.add_option('--top-k', type=int, default=5,
                      help='Top-k terms to keep: default=%default')
    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    target_terms = options.target_terms
    top_k = options.top_k
    is_random = options.is_random
    no_pos = options.no_pos

    subdir = 'subs_masked'
    if no_pos:
        subdir += '_nopos'
    if is_random:
        subdir = 'random_subs_masked'

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    tokenized_dir = get_subdir(basedir, model_name)
    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')

    print("Loading tokenized data")
    lines = []
    with open(tokenized_file) as f:
        for line in tqdm(f):
            lines.append(json.loads(line))
    print(len(lines))

    # Do overall counts for lemmatized data
    source_by_id = {}
    for line in tqdm(lines):
        source_by_id[line['id']] = line['source']

    sub_counter_by_source = defaultdict(Counter)

    target_terms = target_terms.split(',')
    
    for target_term in target_terms:
        target_file = os.path.join(tokenized_dir, subdir, target_term + '_substitutes.jsonlist')

        with open(target_file) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            line_id = line['line_id']
            top_tokens = line['top_terms']
            source = source_by_id[line_id]
            top_k_tokens = top_tokens[:top_k]
            top_k_tokens = [re.sub('##', '', t) for t in top_k_tokens]
            sub_counter_by_source[source].update(top_k_tokens)

        print(target_term)        
        for source in sorted(sub_counter_by_source):
            print(source)
            counter = sub_counter_by_source[source]
            for term, count in counter.most_common(n=15):
                print(term, count)
            print()
            

if __name__ == '__main__':
    main()
