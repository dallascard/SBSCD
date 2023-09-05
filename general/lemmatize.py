import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import spacy
from tqdm import tqdm

from common.misc import get_model_name, get_subdir


# Optionally lemmatize the tokenized text, and or add part of speech tags

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model name or path: default=%default')
    parser.add_option('--spacy', type=str, default='en_core_web_sm',
                      help='spaCy model: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    spacy_model = options.spacy

    print("Loading spacy")
    nlp = spacy.load(spacy_model)

    model_name = get_model_name(model)

    tokenized_dir = get_subdir(basedir, model_name)

    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')
    outfile = os.path.join(tokenized_dir, 'lemmatized.jsonlist')

    with open(tokenized_file) as f:
        lines = f.readlines()
    print("Tokenized lines:", len(lines))

    outlines = []
    for line in tqdm(lines):
        line = json.loads(line)
        mlm_tokens = [re.sub('##', '', token) for token in line['tokens']]
        text = ' '.join(mlm_tokens)
        parsed = nlp(text)
        spacy_tokens = []
        tokens = []        
        lemmas = []
        pos_tags = []
        for token in parsed:
            if len(spacy_tokens) == 0:
                spacy_tokens.append(token)
                tokens.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)
            elif token.idx == spacy_tokens[-1].idx + len(spacy_tokens[-1].text):
                spacy_tokens.append(token)
                tokens[-1] += token.text
                lemmas[-1] += token.lemma_                
            else:
                spacy_tokens.append(token)
                tokens.append(token.text)
                lemmas.append(token.lemma_)
                pos_tags.append(token.pos_)
    
        try:
            assert len(mlm_tokens) == len(tokens)
            assert len(mlm_tokens) == len(lemmas)
            assert len(mlm_tokens) == len(pos_tags)
        except AssertionError as e:
            for i , token in enumerate(mlm_tokens):
                if i >= len(tokens):
                    print(token, 'None', 'None', 'None')
                else:
                    print(token, tokens[i], lemmas[i], pos_tags[i])
            break
        outline = {'id': line['id'], 'lemmas': lemmas, 'pos': pos_tags}
        outlines.append(outline)

    with open(outfile, 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    main()
