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


# Get the top replacements for each period

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
    parser.add_option('--output-subdir', type=str, default='summary',
                     help='Output subdirectory name: default=%default')        
    parser.add_option('--top-k', type=int, default=5,
                      help='Top-k terms to keep: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    infile = options.infile
    model = options.model
    use_lemmas = options.lemmas
    use_pos_tags = options.pos
    subdir = options.subdir
    output_subdir = options.output_subdir
    top_k = options.top_k

    model_name = get_model_name(model)
    tokenized_dir = get_subdir(basedir, model_name)

    subdir += '_' + model_name
    if use_lemmas:
        subdir += '_lemmas'
    if use_pos_tags:
        subdir += '_pos'

    substitutes_dir = os.path.join(tokenized_dir, subdir, 'subs_masked')

    output_dir = os.path.join(tokenized_dir, subdir, output_subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'config_gather.json'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)

    print("Loading data")
    if infile is None:
        tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')
    else:
        tokenized_file = infile
    tokenized_lines = []
    with open(tokenized_file) as f:
        for line in f:
            tokenized_lines.append(json.loads(line))
    
    """
    if use_lemmas:
        lemmatized_file = os.path.join(tokenized_dir, 'lemmatized.jsonlist')    
        lemmatized_lines = []
        with open(lemmatized_file) as f:
            for line in f:
                lemmatized_lines.append(json.loads(line))
    """

    # Count terms
    source_by_id = {}
    raw_source_counter = Counter()
    #term_counter = Counter()
    #tokens_by_line_id = {}
        
    print("Counting terms")
    for line_i, line in enumerate(tqdm(tokenized_lines)):       
        line_id = line['id']
        """
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
        """
        source = line['corpus']
        raw_source_counter[source] += 1
        source_by_id[line_id] = source

    for s, c in raw_source_counter.most_common():
        print(s, c)

    # Get the list of sources
    sources = sorted(raw_source_counter)
    assert len(sources) == 2

    print("Reading target files from", substitutes_dir)
    target_files = sorted(glob(os.path.join(substitutes_dir, '*.jsonlist')))
    print(len(target_files), "target files")

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
            top_k_tokens = top_tokens[:top_k]
            top_k_tokens = [re.sub('##', '', t) for t in top_k_tokens]
            sub_counter_by_source[source].update(top_k_tokens)

        # only keep terms that exist in both corpora
        if count_by_source[sources[0]] > 0 and count_by_source[sources[1]] > 0:
            dfs = []
            for source in sources:
                df = pd.DataFrame()
                subs = []
                counts = []
                for sub, count in sub_counter_by_source[source].most_common():
                    subs.append(sub)
                    counts.append(count)
                df[source] = subs
                df[source + '_count'] = counts
                dfs.append(df)

            df = pd.concat(dfs, axis=1)

            # Save the overall most common replacements to file
            outfile = os.path.join(output_dir, term + '.tsv')
            df.to_csv(outfile, sep='\t', index=False)


if __name__ == '__main__':
    main()


