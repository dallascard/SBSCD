import os
import re
import json
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
    parser.add_option('--infile', type=str, default=None,
                      help='Override data file (optional): default=%default')    
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--lemmas', action="store_true", default=False,
                      help='Use lemmas for indexing: default=%default')    
    parser.add_option('--pos', action="store_true", default=False,
                      help='Use POS tags for indexing: default=%default')        
    parser.add_option('--subdir', type=str, default='embeddings',
                     help='Subdirectory name given to index_terms.py: default=%default')        
    parser.add_option('--top-k', type=int, default=5,
                      help='Top-k terms to keep: default=%default')
    parser.add_option('--window-factor', type=float, default=2.0,
                      help='Factor by which to get nearby terms: default=%default')
    parser.add_option('--targets-file', type=str, default=None,
                      help='Optionally, limit the evaluation to a list of targets in a .tsv file (first column): default=%default')    
    parser.add_option('--header', action="store_true", default=False,
                      help="Set if targets-file has a header to ignore: default=%default")

    (options, args) = parser.parse_args()

    basedir = options.basedir
    infile = options.infile
    model = options.model
    subdir = options.subdir
    use_lemmas = options.lemmas
    use_pos_tags = options.pos
    top_k = options.top_k
    window_factor = options.window_factor
    targets_file = options.targets_file
    header = options.header
    
    if targets_file is not None:
        targets_df = pd.read_csv(targets_file, sep='\t', index_col=None, header=None)
        target_words = set(targets_df[0].values)
        if header:
            target_words = set(target_words[1:])

    model_name = get_model_name(model)

    subdir += '_' + model_name
    if use_lemmas:
        subdir += '_lemmas'
    if use_pos_tags:
        subdir += '_pos'

    with open(os.path.join(basedir, subdir, 'config_jsd.json'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)

    substitutes_dir = os.path.join(basedir, subdir, 'subs_masked')

    print("Loading data")
    if infile is None:
        tokenized_dir = get_subdir(basedir, model_name)
        tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')
    else:
        tokenized_file = infile
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

    # Count terms
    source_by_id = {}
    raw_source_counter = Counter()
    term_counter = Counter()
    tokens_by_line_id = {}
        
    print("Counting terms")
    for line_i, line in enumerate(tqdm(tokenized_lines)):       
        line_id = line['id']
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
        term_counter.update(tokens)
        tokens_by_line_id[line_id] = tokens
        source = line['corpus']
        raw_source_counter[source] += 1
        source_by_id[line_id] = source

    for s, c in raw_source_counter.most_common():
        print(s, c)

    # Get the list of sources
    sources = sorted(raw_source_counter)
    assert len(sources) == 2

    output_df = pd.DataFrame()

    scaled_jsds_by_term = None
    target_files = sorted(glob(os.path.join(substitutes_dir, '*.jsonlist')))
    print(len(target_files), "target files")

    term_counter_by_source = defaultdict(Counter)
    sub_counter_by_term_by_source = {source: defaultdict(Counter) for source in raw_source_counter}

    for infile in tqdm(target_files):
        count_by_source = Counter()
        sub_counter_by_source = defaultdict(Counter)
        basename = os.path.basename(infile)
        term = basename[:-len('.jsonlist')]
        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            line_id = line['line_id']
            top_tokens = line['top_terms']
            source = source_by_id[line_id]
            count_by_source[source] += 1
            term_counter_by_source[source][term] += 1
            top_k_tokens = top_tokens[:top_k]
            top_k_tokens = [re.sub('##', '', t) for t in top_k_tokens]
            sub_counter_by_source[source].update(top_k_tokens)

        # only keep terms that exist in both corpora
        if count_by_source[sources[0]] > 0 and count_by_source[sources[1]] > 0:
            for source in sources:
                sub_counter_by_term_by_source[source][term].update(sub_counter_by_source[source])

    print(len(sub_counter_by_term_by_source[sources[0]]), len(sub_counter_by_term_by_source[sources[1]]) )

    # get those terms that exist in both corpora
    valid_terms = sorted(sub_counter_by_term_by_source[sources[0]])
    valid_terms_clean = [re.sub('##', '', term) for term in valid_terms]

    if targets_file is not None:
        valid_terms = [term for t_i, term in enumerate(valid_terms) if valid_terms_clean[t_i] in target_words]
        valid_terms_clean = [term for term in valid_terms_clean if term in target_words]

    print(len(valid_terms))
    print(valid_terms[-5:])
    jsds = []
    for term in tqdm(valid_terms):
        jsd = compute_jsd_from_counters(sub_counter_by_term_by_source[sources[0]][term], sub_counter_by_term_by_source[sources[1]][term])
        jsds.append(jsd)

    relative_jsds = []
    n_neighbours = []

    valid_term_counts = [term_counter[term] for term in valid_terms_clean]

    order = np.argsort(valid_term_counts)
    print("Least common background terms:")
    for i in order[:25]:
        print(valid_terms_clean[i], valid_term_counts[i])

    jsds_by_term = dict(zip(valid_terms_clean, jsds))

    nearby_by_target = defaultdict(list)
    for target_i, target in tqdm(enumerate(valid_terms_clean)):
        target_count = term_counter[target]
        target_jsd = jsds_by_term[target]
        nearby_terms = [term for term in valid_terms_clean if target_count/window_factor <= term_counter[term] <= target_count*window_factor and term != target]

        nearby_by_target[target] = nearby_terms
        n_neighbours.append(len(nearby_terms))
        nearby_jsds = np.array([jsds_by_term[term] for term in nearby_terms])
        quartile = float(np.mean(target_jsd >= nearby_jsds))
        relative_jsds.append(quartile)

    order = np.argsort(n_neighbours)
    print('min neihgbours')
    for i in order[:5]:
        target = valid_terms[i]
        print(target, n_neighbours[i], relative_jsds[i])

    print('max neihgbours')
    for i in order[-5:]:
        target = valid_terms[i]
        print(target, n_neighbours[i], relative_jsds[i])

    scaled_jsds_by_term = dict(zip(valid_terms, relative_jsds))

    output_df['term'] = valid_terms_clean
    output_df['jsd'] = jsds
    output_df['count_' + sources[0]] = [term_counter_by_source[sources[0]][term] for term in valid_terms_clean]
    output_df['count_' + sources[1]] = [term_counter_by_source[sources[1]][term] for term in valid_terms_clean]
    output_df['neighbours'] = n_neighbours
    output_df['scaled_jsd'] = relative_jsds

    output_df.sort_values(by='scaled_jsd', ascending=False, inplace=True)

    outfile = os.path.join(basedir, subdir, 'jsd_scores')

    output_df.to_csv(outfile + '.csv', index=False)

    with open(outfile + '.json', 'w') as f:
        json.dump(scaled_jsds_by_term, f)



if __name__ == '__main__':
    main()


