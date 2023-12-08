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
    parser.add_option('--data-dir', type=str, default=None,
                      help='Manually specify data directory (optional): default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                     help='model used for tokenization: default=%default')
    parser.add_option('--lemmas', action="store_true", default=False,
                      help='Use lemmas for indexing: default=%default')    
    parser.add_option('--pos', action="store_true", default=False,
                      help='Use POS tags for indexing: default=%default')    
    parser.add_option('--subdir', type=str, default='embeddings',
                     help='Subdirectory to save indices and subsequent experimental outputs: default=%default')    
    parser.add_option('--targets-file', type=str, default=None,
                     help='Targets file (.tsv or similar): default=%default')
    parser.add_option('--min-count', type=int, default=20,
                      help='Restrict token selection to those with at least this many tokens: default=%default')
    parser.add_option('--min-count-per-corpus', type=int, default=3,
                      help='Restrict token selection to those with at least this many tokens per corpus: default=%default')
    parser.add_option('--min-term-length', type=int, default=2,
                      help='Minimum term length in characters: default=%default')
    parser.add_option('--max-terms', type=int, default=None,
                      help='Number of words to sample: default=%default')
    parser.add_option('--max-tokens', type=int, default=4000,
                      help='Limit the size of the index by pre-sampling: default=%default')
    parser.add_option('--seed', type=int, default=42,
                     help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    tokenized_dir = options.data_dir
    model = options.model
    output_subdir = options.subdir
    targets_file = options.targets_file
    use_lemmas = options.lemmas
    use_pos_tags = options.pos
    min_count = options.min_count
    min_count_per_corpus = options.min_count_per_corpus
    min_term_length = options.min_term_length
    max_terms = options.max_terms
    max_tokens = options.max_tokens
    seed = options.seed


    np.random.seed(seed)

    model_name = get_model_name(model)

    print("Loading data")
    if tokenized_dir is None:
        tokenized_dir = get_subdir(basedir, model_name)
    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')
    tokenized_lines = []
    with open(tokenized_file) as f:
        for line in f:
            tokenized_lines.append(json.loads(line))
    
    if use_lemmas:
        lemmatized_file = os.path.join(tokenized_dir, 'lemmatized.jsonlist')    
        lemmatized_lines = []
        with open(lemmatized_file) as f:
            for line in f:
                lemmatized_lines.append(json.loads(line))

    # lightly decorate output subdirectory name with important options
    output_subdir += '_' + model_name
    if use_lemmas:
        output_subdir += '_lemmas'
    if use_pos_tags:
        output_subdir += '_pos'
    if not os.path.exists(os.path.join(tokenized_dir, output_subdir)):
        os.makedirs(os.path.join(tokenized_dir, output_subdir))


    config = {'model': model,
              'tokenized_dir': tokenized_dir,
              'model': model,
              'output_subdir': output_subdir,
              'targets_file': targets_file,
              'use_lemmas': use_lemmas,
              'use_pos_tags': use_pos_tags,
              'min_count': min_count,
              'min_count_per_corpus': min_count_per_corpus,
              'min_term_length': min_term_length,
              'max_terms': max_terms,
              'max_tokens': max_tokens,
              'seed': seed
              }
    
    with open(os.path.join(tokenized_dir, output_subdir, 'config_indexing.json'), 'w') as f:
        json.dump(config, f, indent=2)    

    targets = set()
    if targets_file is not None:
        with open(targets_file) as f:
            target_lines = f.readlines()
        targets = set([line.strip().split('\t')[0] for line in target_lines])

    multiword_targets = []
    multiword_target_dict = defaultdict(list)
    for target in targets:
        parts = target.split()
        if len(parts) > 1:
            multiword_targets.append(target)
            multiword_target_dict[parts[0]].append(parts[1:])

    term_counter = Counter()
    term_counter_per_corpus = defaultdict(Counter)
    term_indices = defaultdict(list)    

    print("Counting terms")
    converted_lines = []
    if len(targets) > 0:
        for line_i, line in enumerate(tqdm(tokenized_lines)):            
            source = line['corpus']
            if use_lemmas:
                if use_pos_tags:
                    tokens = [lemma + '_' + lemmatized_lines[line_i]['pos'][l_i] for l_i, lemma in enumerate(lemmatized_lines[line_i]['lemmas'])]
                else:
                    tokens = lemmatized_lines[line_i]['lemmas']
            else:
                if use_pos_tags:
                    tokens = [token + '_' + line['pos'][t_i] for t_i, token in enumerate(line['tokens'])]
                else:
                    tokens = line['tokens']
                tokens = [re.sub('##', '', token) for token in tokens]        

            # Count tokens, including multi-word targets
            skip = 0
            for t_i, token in enumerate(tokens):
                # skip this token if it was part of a multi-word target
                if skip > 0:
                    skip =- 1
                # first check for multi-word targets
                elif token in multiword_target_dict:
                    continuations = multiword_target_dict[token]
                    for continuation in continuations:
                        if continuation == tokens[t_i+1:t_i+1+len(continuation)]:
                            term_counter[token + ' ' + ' '.join(continuation)] += 1
                            term_counter_per_corpus[source][token + ' ' + ' '.join(continuation)] += 1
                            skip = len(continuation)
                    if skip > 0:
                        term_counter[token] += 1
                        term_counter_per_corpus[source][token] += 1
                else:
                    term_counter[token] += 1
                    term_counter_per_corpus[source][token] += 1

            converted_lines.append({'id' : line['id'], 'tokens' : tokens})

    sources = sorted(term_counter_per_corpus)

    print("\nTarget counts:")
    target_list = sorted(targets)
    target_counts = [term_counter[target] for target in target_list]
    order = np.argsort(target_counts)[::-1]
    for i in order:
        print(target_list[i], target_counts[i], min([term_counter_per_corpus[source][target_list[i]] for source in sources]))  
        if min([term_counter_per_corpus[source][target_list[i]] for source in sources]) < min_count_per_corpus:
            print("Warning: target term", target_list[i], "has too few instances in one of the corpora")
            for source in sources:
                print(source, term_counter_per_corpus[source][target_list[i]])

    valid_term_set = set([term for term, count in term_counter.items() 
                          if count >= min_count and                          
                          min([term_counter_per_corpus[source][term] for source in sources]) >= min_count_per_corpus and
                          len(term.split('_')[0]) >= min_term_length and
                          '/' not in term and
                          re.match('.*[a-zA-Z0-9].*', term.split('_')[0]) is not None])
   
    remaining_set = valid_term_set - set(targets)
    if max_terms is not None and len(remaining_set) > max_terms:
        remaining_set = set(np.random.choice(list(remaining_set), size=max_terms-len(targets), replace=False))

    full_target_set = set(targets).union(remaining_set)

    random_subset = np.random.choice(list(full_target_set), size=10, replace=False)
    print("\nRandom terms:")
    for term in random_subset:
        print(term, term_counter[term])

    print("\nIndexing target terms:")
    if len(targets) > 0:
        for line_i, line in enumerate(tqdm(converted_lines)):
            line_id = line['id']
            tokens = line['tokens']            
            skip = 0
            for t_i, token in enumerate(tokens):
                # skip this token if it was part of a multi-word target
                if skip > 0:
                    skip =- 1
                # first check for multi-word targets
                elif token in multiword_target_dict:
                    continuations = multiword_target_dict[token]
                    for continuation in continuations:
                        if continuation == tokens[t_i+1:t_i+1+len(continuation)]:
                            term_indices[token + ' ' + ' '.join(continuation)].append((line_id, t_i))
                            skip = len(continuation)
                    if skip > 0:
                        term_indices[token].append((line_id, t_i))
                elif token in full_target_set:
                    term_indices[token].append((line_id, t_i))
                else:
                    pass
                
    print("\nTaking random sample of indices:")
    n_over_max_tokens = 0    
    total_tokens = 0
    final_token_indices_by_term = defaultdict(list)
    for term in term_indices:
        indices = term_indices[term]
        if len(indices) > max_tokens:
            n_over_max_tokens += 1            
            total_tokens += max_tokens
            random_indices = np.random.choice(np.arange(len(indices)), size=max_tokens, replace=False)
            final_token_indices_by_term[term] = [indices[i] for i in random_indices]
        else:
            final_token_indices_by_term[term] = indices
            total_tokens += len(indices)

    print("\nNumber of terms to be embedded:", len(final_token_indices_by_term))
    print("Total tokens to be embedded:", total_tokens)
    print("Number of terms sampled down to max_tokens:", n_over_max_tokens)
    print("Min instances:", min([len(indices) for indices in final_token_indices_by_term.values()]))
    print("Max instances:", max([len(indices) for indices in final_token_indices_by_term.values()]))

    outfile = os.path.join(tokenized_dir, output_subdir, 'target_indices.json')
    
    with open(outfile, 'w') as f:
        json.dump(final_token_indices_by_term, f, indent=2)


if __name__ == '__main__':
    main()
