import os
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast

from common.misc import get_model_name, get_subdir


# Tokenize text

def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base directory: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--strip-accents', action="store_true", default=False,
                      help='Strip accents when tokenizing: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model_name_or_path = options.model
    strip_accents = options.strip_accents
    
    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    model_name = get_model_name(model_name_or_path)

    indir = os.path.join(basedir, 'clean_tokens')
    outdir = get_subdir(basedir, model_name, strip_accents)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading model")
    tokenizer_class = BertTokenizerFast

    # Load pretrained model/tokenizer
    if lang == 'ger':
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        if strip_accents:
            tokenizer.backend_tokenizer.normalizer.strip_accents = True
        else:
            tokenizer.backend_tokenizer.normalizer.strip_accents = False
    else:
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

    infile = os.path.join(indir, 'all.jsonlist')

    printed_empty = False

    token_counter = Counter()

    outlines = []

    empty_docs = 0
    emtpy_after_tokenization = 0
    outlines = []
    basename = os.path.basename(infile)
    print(infile)
    with open(infile) as f:
        lines = f.readlines()
    print(len(lines))

    max_n_pieces = 0
    for line in tqdm(lines):
        line = json.loads(line)
        line_id = line['id']
        # drop the header
        if 'text' not in line:
            print('No text in', line_id)
        else:
            text = line['text'].strip()

            if len(text) == 0:
                empty_docs += 1
                if not printed_empty:
                    print("Empty body in", line_id)
                    print(line)
                    printed_empty = True
            else:
                # convert to tokens using BERT
                #raw_pieces = [tokenizer.ids_to_tokens[i] for i in tokenizer.encode(text, add_special_tokens=False)]
                raw_pieces = tokenizer(text, add_special_tokens=False).tokens()
                max_n_pieces = max(max_n_pieces, len(raw_pieces))

                rejoined_pieces = []
                # rejoin into concatenated words
                if len(raw_pieces) == 0:
                    emtpy_after_tokenization += 1
                    print("No tokens after tokenization in", line_id)
                    print(text)
                else:
                    for p_i, piece in enumerate(raw_pieces):
                        if p_i == 0:
                            rejoined_pieces.append(piece)
                        elif piece.startswith('##'):
                            rejoined_pieces[-1] += piece
                        else:
                            rejoined_pieces.append(piece)
                    
                    line['tokens'] = rejoined_pieces

                    token_counter.update(rejoined_pieces)

                    outlines.append(line)

    print(basename, len(lines), empty_docs, emtpy_after_tokenization, len(outlines), len(lines) - empty_docs -len(outlines))
    outfile = os.path.join(outdir, 'all.jsonlist')
    with open(outfile, 'w') as fo:
        for line in outlines:
            fo.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
