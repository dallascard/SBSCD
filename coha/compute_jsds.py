import os
import re
import json
import unicodedata
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.misc import get_model_name, get_subdir
from common.compute_jsd import compute_jsd_from_counters


# Compute JSD for target terms, scaled by background terms

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--top-k', type=int, default=5,
                      help='Top-k terms to keep: default=%default')
    parser.add_option('--window-factor', type=float, default=2.0,
                      help='Factor by which to get nearby terms: default=%default')
    parser.add_option('--source-field', type=str, default='group',
                      help='Field to use for splitting into 2 subsets: default=%default')
    parser.add_option('--compute-all', action="store_true", default=False,
                      help="Compute scaled JSD for all terms (including random): default=%default")

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    top_k = options.top_k
    window_factor = options.window_factor
    source_field = options.source_field
    compute_all = options.compute_all

    targets_subdir = 'subs_masked'
    random_subdir = 'random_subs_masked'

    model_name = get_model_name(model)

    tokenized_dir = get_subdir(basedir, model_name)
    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')

    targets_dir = os.path.join(tokenized_dir, targets_subdir)
    if os.path.exists(targets_dir):
        print("Loading target substitutes from", targets_dir)
        target_files = sorted(glob(os.path.join(targets_dir, '*.jsonlist')))
        print(len(target_files), "target files")
    else:
        print("Target dir does not exist:", targets_dir)
        target_files = []

    if len(target_files) == 0 and not compute_all:
        raise RuntimeError("No target files found; aborting")

    # Do overall counts for lemmatized data
    source_by_id = {}
    raw_source_counter = Counter()
        
    # repeat for tokenized data if using random tokens for comparison
    raw_token_counter = Counter()
    token_counter_by_source = defaultdict(Counter)
    with open(tokenized_file) as f:
        tokenized_lines = f.readlines()
        for line in tqdm(tokenized_lines):
            line = json.loads(line)
            line_id = line['id']
            source = line[source_field]
            tokens = line['tokens']
            tokens  = [re.sub('##', '', token) for token in tokens]
            raw_source_counter[source] += 1
            source_by_id[line_id] = source
            raw_token_counter.update(tokens)
            token_counter_by_source[source].update(tokens)

    for s, c in raw_source_counter.most_common():
        print(s, c)

    # Get the list of sources
    sources = sorted(raw_source_counter)
    assert len(sources) == 2

    output_df = pd.DataFrame()

    if len(target_files) > 0:
        print("Loading target substitutes")
        # Load the subsitutes for target terms
        targets_sub_counter_by_term_by_source = {source: defaultdict(Counter) for source in raw_source_counter}
        targets_counter_by_term = Counter()

        term_counter_by_source = defaultdict(Counter)

        for infile in target_files:
            count_by_source = Counter()
            basename = os.path.basename(infile)        
            parts = basename.split('_')
            if len(parts) == 2:
                term = parts[0]
            else:
                term = '_'.join(parts[:-1])

            with open(infile) as f:
                lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                line_id = line['line_id']
                top_tokens = line['top_terms']
                source = source_by_id[line_id]
                top_k_tokens = top_tokens[:top_k]
                top_k_tokens = [re.sub('##', '', t) for t in top_k_tokens]
                targets_sub_counter_by_term_by_source[source][term].update(top_k_tokens)
                count_by_source[source] += 1
                targets_counter_by_term[term] += 1
                term_counter_by_source[source][term] += 1
            print(term, len(targets_sub_counter_by_term_by_source[sources[0]][term]), len(targets_sub_counter_by_term_by_source[sources[1]][term]))

        print("Computing JSDs")
        target_terms = sorted(targets_sub_counter_by_term_by_source[sources[0]])
        target_jsds = []
        for term in target_terms:
            jsd = compute_jsd_from_counters(targets_sub_counter_by_term_by_source[sources[0]][term], targets_sub_counter_by_term_by_source[sources[1]][term])
            print(term, jsd)
            target_jsds.append(float(jsd))

        raw_jsds_by_term = dict(zip(target_terms, target_jsds))

        print(len(target_terms), len(target_jsds))

        if not compute_all:
            output_df['term'] = target_terms
            output_df['jsd'] = target_jsds
            output_df['count_' + sources[0]] = [term_counter_by_source[sources[0]][term] for term in target_terms]
            output_df['count_' + sources[1]] = [term_counter_by_source[sources[1]][term] for term in target_terms]

    scaled_jsds_by_term = None
    if random_subdir is not None:
        print("\nRepeating with random term correction")
        print("\nLoading random subsitutes")
        random_files = sorted(glob(os.path.join(tokenized_dir, random_subdir, '*.jsonlist')))
        print(len(random_files), "random files")

        random_sub_counter_by_term_by_source_random = {source: defaultdict(Counter) for source in raw_source_counter}
        random_counter_by_term_random = Counter()

        for infile in tqdm(random_files):
            count_by_source = Counter()
            counter_by_source = defaultdict(Counter)
            basename = os.path.basename(infile)
            term = basename.split('_')[0]
            with open(infile) as f:
                lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                line_id = line['line_id']
                top_tokens = line['top_terms']
                source = source_by_id[line_id]
                count_by_source[source] += 1
                top_k_tokens = top_tokens[:top_k]
                top_k_tokens = [re.sub('##', '', t) for t in top_k_tokens]
                counter_by_source[source].update(top_k_tokens)
                random_counter_by_term_random[term] += 1

            if count_by_source[sources[0]] > 0 and count_by_source[sources[1]] > 0:
                for source in sources:
                    random_sub_counter_by_term_by_source_random[source][term].update(counter_by_source[source])

        print(len(random_sub_counter_by_term_by_source_random[sources[0]]),len(random_sub_counter_by_term_by_source_random[sources[1]]) )

        random_terms = sorted(random_sub_counter_by_term_by_source_random[sources[0]])
        random_terms_clean = [re.sub('##', '', term) for term in random_terms]

        print(len(random_terms))
        print(random_terms[-5:])
        random_jsds = []
        for term in tqdm(random_terms):
            jsd = compute_jsd_from_counters(random_sub_counter_by_term_by_source_random[sources[0]][term], random_sub_counter_by_term_by_source_random[sources[1]][term])
            random_jsds.append(jsd)

        relative_jsds = []
        n_neighbours = []

        all_terms_clean = target_terms + random_terms_clean
        all_jsds = target_jsds + random_jsds

        random_term_counts = [raw_token_counter[term] for term in random_terms_clean]

        order = np.argsort(random_term_counts)
        print("Least common background terms:")
        for i in order[:25]:
            print(random_terms_clean[i], random_term_counts[i])

        all_jsds_by_term = dict(zip(all_terms_clean, all_jsds))

        if compute_all:
            to_process = all_terms_clean
        else:
            to_process = target_terms

        nearby_by_target = defaultdict(list)
        for target_i, target in tqdm(enumerate(to_process)):
            target_count = raw_token_counter[target]
            target_jsd = all_jsds[target_i]
            nearby_terms = [term for term in all_terms_clean if target_count/window_factor <= raw_token_counter[term] <= target_count*window_factor and term != target]

            nearby_by_target[target] = nearby_terms
            n_neighbours.append(len(nearby_terms))
            nearby_jsds = np.array([all_jsds_by_term[term] for term in nearby_terms])
            quartile = float(np.mean(target_jsd >= nearby_jsds))
            relative_jsds.append(quartile)

        order = np.argsort(n_neighbours)
        print('min neihgbours')
        for i in order[:5]:
            target = target_terms[i]
            print(target, n_neighbours[i], relative_jsds[i])

        print('max neihgbours')
        for i in order[-5:]:
            target = target_terms[i]
            print(target, n_neighbours[i], relative_jsds[i])

        scaled_jsds_by_term = dict(zip(target_terms, relative_jsds))

        if compute_all:
            output_df['term'] = all_terms_clean
            output_df['jsd'] = all_jsds
            output_df['count_' + sources[0]] = [term_counter_by_source[sources[0]][term] for term in all_terms_clean]
            output_df['count_' + sources[1]] = [term_counter_by_source[sources[1]][term] for term in all_terms_clean]
        output_df['neighbours'] = n_neighbours
        output_df['scaled_jsd'] = relative_jsds


    output_df.sort_values(by='scaled_jsd', ascending=False, inplace=True)

    output_df.to_csv(os.path.join(tokenized_dir, 'jsd_scores.csv'), index=False)

    with open(os.path.join(tokenized_dir, 'jsd_scores.json'), 'w') as f:
        json.dump({'raw': raw_jsds_by_term, 'scaled': scaled_jsds_by_term}, f)



if __name__ == '__main__':
    main()
