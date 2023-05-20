import os
import re
import json
from glob import glob
from subprocess import call
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
    parser.add_option('--epochs', type=int, default=5,
                      help='Number of epochs: default=%default')
    parser.add_option('--per-gpu', type=int, default=32,
                      help='Batch size: default=%default')
    parser.add_option('--cache-dir', type=str, default='/data/dalc/cache/huggingface/',
                      help='Cache directory: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    strip_accents = options.strip_accents
    epochs = options.epochs
    per_gpu = options.per_gpu
    cache_dir = options.cache_dir

    subdir_prefix = 'mlm_pretraining'

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    data_dir = get_subdir(basedir, model_name, strip_accents, prefix=subdir_prefix)
    print("Using data dir", data_dir)

    output_dir = os.path.join(data_dir, 'model')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cmd = ['python',
           '-m',
           'common.run_mlm',
           '--model_name_or_path', model,
           '--model_type', 'bert',
           '--cache_dir', cache_dir,
           '--train_file', os.path.join(data_dir, 'all_raw_train.txt'),
           '--validation_file', os.path.join(data_dir, 'all_raw_val.txt'),
           '--do_train',
           '--do_eval',
           '--max_seq_length', '256',
           '--per_device_train_batch_size', str(per_gpu),
           '--per_device_eval_batch_size', str(per_gpu),
           '--output_dir', output_dir,
           '--overwrite_cache',
           '--overwrite_output_dir',
           '--num_train_epochs', str(epochs),
           '--logging_dir', os.path.join(data_dir, 'logs'),
           '--save_total_limit', '0'
           ]

    with open(os.path.join(output_dir, 'my_cmd.txt'), 'w') as f:
        f.write(' '.join(cmd))

    print(cmd)
    call(cmd)

    with open(os.path.join(output_dir, 'my_cmd.txt'), 'w') as f:
        f.write(' '.join(cmd))

if __name__ == '__main__':
    main()
