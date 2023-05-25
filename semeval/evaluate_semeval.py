import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from common.misc import get_model_name, get_subdir
from common.compute_jsd import compute_jsd_from_counters


# Do the evaluation of substitute method, with and without scaling by background term

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base directory: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--strip-accents', action="store_true", default=False,
                      help='Strip accents when tokenizing: default=%default')
    parser.add_option('--top-k', type=int, default=5,
                      help='Top-k terms to keep: default=%default')
    parser.add_option('--window-factor', type=float, default=2.0,
                      help='Factor by which to get nearby terms: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    strip_accents = options.strip_accents
    top_k = options.top_k
    window_factor = options.window_factor

    no_pos = False
    if lang == 'eng':
        no_pos = True

    targets_subdir = 'subs_masked'
    if no_pos:
        targets_subdir += '_nopos'
    random_subdir = 'random_subs_masked'

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    lemmas_dir = os.path.join(basedir, 'clean_lemmas')
    lemmas_file = os.path.join(lemmas_dir, 'all.jsonlist')
    tokenized_dir = get_subdir(basedir, model_name, strip_accents)
    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')

    print("Loading labels")
    label_file = os.path.join(basedir, 'truth', 'graded.txt')
    label_df = pd.read_csv(label_file, header=None, index_col=None, sep='\t')
    label_df.columns = ['term', 'score']

    target_words = label_df['term'].values

    if no_pos:
        target_words = [term.split('_')[0] for term in target_words]

    target_shifts = label_df['score'].values
    target_shift_by_word = dict(zip(target_words, target_shifts))
    print(len(target_words), len(target_shift_by_word))

    targets_dir = os.path.join(tokenized_dir, targets_subdir, '*.jsonlist')
    print("Loading target substitutes from", targets_dir)
    target_files = sorted(glob(targets_dir))
    print(len(target_files), "target files")
    if len(target_files) == 0:
        raise Exception("No target files found in", targets_dir)

    print("Loading tokenized data")
    lemma_lines = []
    with open(lemmas_file) as f:
        for line in tqdm(f):
            lemma_lines.append(json.loads(line))
    print(len(lemma_lines))

    # Do overall counts for lemmatized data
    source_by_id = {}
    raw_source_counter = Counter()
    raw_lemma_counter = Counter()
    for line in tqdm(lemma_lines):
        raw_source_counter[line['source']] += 1
        source_by_id[line['id']] = line['source']
        text = line['text']
        lemmas = text.split()
        raw_lemma_counter.update(lemmas)

    for s, c in raw_source_counter.most_common():
        print(s, c)
        
    for t, c in raw_lemma_counter.most_common(n=3):
        print(t, c)          

    # repeat for tokenized data if using random tokens for comparison
    raw_token_counter = Counter()
    with open(tokenized_file) as f:
        tokenized_lines = f.readlines()
        for line in tqdm(tokenized_lines):
            line = json.loads(line)
            tokens = line['tokens']
            tokens  = [re.sub('##', '', token) for token in tokens]
            raw_token_counter.update(tokens)

    # Get the list of sources
    sources = sorted(raw_source_counter)
    assert len(sources) == 2

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
            #if exclude_self:
            #    top_k_tokens = [t for t in top_k_tokens if t != term]
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
        target_jsds.append(jsd)    

    output_df = pd.DataFrame()
    output_df['term'] = target_terms
    output_df['jsd'] = target_jsds
    output_df['count_' + sources[0]] = [term_counter_by_source[sources[0]][t] for t in target_terms]
    output_df['count_' + sources[1]] = [term_counter_by_source[sources[1]][t] for t in target_terms]

    print(len(target_terms), len(target_jsds))

    target_shifts_sorted = [target_shift_by_word[re.sub('##', '', t)] for t in target_terms]   

    print("\nUncorrected results:")
    print(len(target_shifts_sorted), len(target_jsds))
    print(pearsonr(target_shifts_sorted, target_jsds))
    print(spearmanr(target_shifts_sorted, target_jsds))

    print("\nRepeating with random term correction")
    print("\nLoading random subsitutes")
    random_files = sorted(glob(os.path.join(tokenized_dir, random_subdir, '*.jsonlist')))
    print(len(random_files), "random files")

    if len(random_files) == 0:
        raise RuntimeError("No files found in", random_files)

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

    random_term_counts = [raw_lemma_counter[term] for term in random_terms_clean]

    order = np.argsort(random_term_counts)
    print("Least common background terms:")
    for i in order[:5]:
        print(random_terms_clean[i], random_term_counts[i])

    all_jsds_by_term = dict(zip(all_terms_clean, all_jsds))

    nearby_by_target = defaultdict(list)
    for target_i, target in tqdm(enumerate(target_terms)):
        if not no_pos:
            target = target.split('_')[0]
        target_count = raw_lemma_counter[target]
        target_jsd = target_jsds[target_i]
        nearby_terms = [term for term in all_terms_clean if target_count/window_factor <= raw_lemma_counter[term] <= target_count*window_factor and term != target]

        nearby_by_target[target] = nearby_terms
        n_neighbours.append(len(nearby_terms))
        nearby_jsds = np.array([all_jsds_by_term[term] for term in nearby_terms])
        quartile = np.mean(target_jsd >= nearby_jsds)
        relative_jsds.append(quartile)
        print(target, target_count, target_jsd, len(nearby_terms), quartile, target_shifts_sorted[target_i])

    order = np.argsort(n_neighbours)
    print('min neihgbours')
    for i in order[:5]:
        target = target_terms[i]
        print(target, n_neighbours[i], relative_jsds[i])

    print('max neihgbours')
    for i in order[-5:]:
        target = target_terms[i]
        print(target, n_neighbours[i], relative_jsds[i])

    output_df['scaled_jsd'] = relative_jsds

    print("Scaled")
    print(len(target_shifts_sorted), len(relative_jsds))
    print(pearsonr(target_shifts_sorted, relative_jsds))
    print(spearmanr(target_shifts_sorted, relative_jsds))

    output_df.sort_values(by='scaled_jsd', ascending=False, inplace=True)
    
    output_file = os.path.join(tokenized_dir, 'jsd_by_term.csv')
    output_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()
