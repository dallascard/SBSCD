import os
import json
from glob import glob
from optparse import OptionParser

import pandas as pd
from scipy.stats import spearmanr, pearsonr

from common.misc import get_model_name, get_subdir


# Evaluate rank correlation against human judgements for a subset of words

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--lemmas', action="store_true", default=False,
                      help='Use lemmas for indexing: default=%default')    
    parser.add_option('--pos', action="store_true", default=False,
                      help='Use POS tags for indexing: default=%default')        
    parser.add_option('--subdir', type=str, default='embeddings',
                     help='Subdirectory name given to index_terms.py: default=%default')        
    parser.add_option('--targets-file', type=str, default=None,
                      help='Target two-column .tsv file containing target words and human scores: default=%default')
    parser.add_option('--header', action="store_true", default=False,
                      help="Set if targets-file has a header to ignore: default=%default")

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    subdir = options.subdir
    use_lemmas = options.lemmas
    use_pos_tags = options.pos
    targets_file = options.targets_file
    header = options.header

    model_name = get_model_name(model)

    tokenized_dir = get_subdir(basedir, model_name)

    subdir += '_' + model_name
    if use_lemmas:
        subdir += '_lemmas'
    if use_pos_tags:
        subdir += '_pos'

    with open(os.path.join(basedir, subdir, 'config_evaluate.json'), 'w') as f:
        json.dump(options.__dict__, f, indent=2)

    jsd_file = os.path.join(basedir, subdir, 'jsd_scores.json')
    
    with open(jsd_file) as f:
        jsd_data = json.load(f)

    targets_df = pd.read_csv(targets_file, sep='\t', index_col=None, header=None)
    target_words = targets_df[0].values
    target_values = targets_df[1].values
    if header:
        target_words = target_words[1:]
        target_values = target_values[1:]

    excluded = []
    words_found = []
    scaled_jsd_values = []
    target_scores_subset = []
    for word_i, word in enumerate(target_words):
        if word in jsd_data:
            scaled_jsd_values.append(jsd_data[word])
            words_found.append(word)
            target_scores_subset.append(target_values[word_i])
        else:
            excluded.append(word)

    print("{:d} word excluded: {:s}".format(len(excluded), ', '.join(excluded)))
    
    print("Scaled results:")
    print("Pearson:", pearsonr(target_scores_subset, scaled_jsd_values))
    print("Spearman:", spearmanr(target_scores_subset, scaled_jsd_values))


if __name__ == '__main__':
    main()


