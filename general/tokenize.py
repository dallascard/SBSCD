import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast

from common.misc import get_model_name, get_subdir


# Tokenize a corpus

def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--infile', type=str, default='/data/dalc/COHA/clean/all.jsonlist',
                      help='File with corpus to be tokenized: default=%default')
    parser.add_option('--id-field', type=str, default='id',
                      help='ID field name: default=%default')
    parser.add_option('--text-field', type=str, default='text',
                      help='Text field name: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    infile = options.infile
    id_field = options.id_field
    text_field = options.text_field
    model_name_or_path = options.model
    
    model_name = get_model_name(model_name_or_path)

    outdir = get_subdir(basedir, model_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("Loading model")
    tokenizer_class = BertTokenizerFast

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

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
        line_id = line[id_field]
        line['id'] = line_id
        # drop the header
        if text_field not in line:
            print('No text in', line_id)
        else:
            text = line[text_field].strip()
            text = re.sub('#+', '#', text)
            text = re.sub('_', ' ', text)

            if len(text) == 0:
                empty_docs += 1
                if not printed_empty:
                    print("Empty body in", line_id)
                    print(line)
                    printed_empty = True
            else:
                # convert to tokens using BERT
                raw_pieces = tokenizer(text, add_special_tokens=False).tokens()
                max_n_pieces = max(max_n_pieces, len(raw_pieces))

                rejoined_pieces = []
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
