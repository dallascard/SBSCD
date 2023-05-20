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

# Recombine tokenized data into plan text for continued MLM pretraining


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base directory: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--strip-accents', action="store_true", default=False,
                      help='Strip accents when tokenizing: default=%default')
    parser.add_option('--val-frac', type=float, default=0.05,
                      help='Fraction to use for validation: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    strip_accents = options.strip_accents
    output_subdir = 'mlm_pretraining'
    val_frac = options.val_frac
    seed = options.seed    
    np.random.seed(seed)

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    tokenized_dir = get_subdir(basedir, model_name, strip_accents)    
    tokenized_file =  os.path.join(tokenized_dir, 'all.jsonlist')
    print("Tokenized file:", tokenized_file)

    outdir = get_subdir(basedir, model_name, strip_accents, prefix=output_subdir)
    print("Using outdir", outdir)
    if not os.path.exists(outdir):        
        os.makedirs(outdir)

    outlines = []
    outlines_val = []

    with open(tokenized_file) as f:
        lines = f.readlines()
    print(len(lines))

    for line in tqdm(lines):            
        line = json.loads(line)            

        tokens = line['tokens']      
        text = ' '.join(tokens)
        text = re.sub('##', '', text)        
        tokens = text.split()              

        if val_frac > 0 and np.random.rand() <= val_frac:
            outlines_val.append(' '.join(tokens))
        else:
            outlines.append(' '.join(tokens))

    print(len(outlines), len(outlines_val))
    print()

    if len(outlines) > 0:
        outfile = os.path.join(outdir, 'all_raw_train.txt')

        order = np.arange(len(outlines))
        np.random.shuffle(order)
        with open(outfile, 'w') as f:
            for i in order:
                f.write(outlines[i] + '\n')

    if len(outlines_val) > 0:
        outfile = os.path.join(outdir, 'all_raw_val.txt')

        order = np.arange(len(outlines_val))
        np.random.shuffle(order)
        with open(outfile, 'w') as f:
            for i in order:
                f.write(outlines_val[i] + '\n')


if __name__ == '__main__':
    main()
