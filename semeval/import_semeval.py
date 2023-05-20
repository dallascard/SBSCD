import os
import re
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

from tqdm import tqdm

# Import SemEval (tokens and lemmas) and do the following processing:
# - convert #+ to #
# - replace underscores with spaces (except English, where they are used for POS tags)
# - fix some special characters and diacritics for German
# - drop URLs (starting with http)
# Finally, also identify the indices of the targets in the lemmatized data


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base dir: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang

    lang_dir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    # Read in the targets
    targets_file = os.path.join(lang_dir, 'targets.txt')
    with open(targets_file) as f:
        lines = f.readlines()
    targets = set([line.strip() for line in lines])
    target_indices = defaultdict(list)
    
    # For English, index the targets both with and without POS tags
    if lang == 'eng':
        targets_nopos = set([token.split('_')[0] for token in targets])
        target_indices_nopos = defaultdict(list)

    # First deal with lemmas, and index the targets
    outdir_lemmas_dir = os.path.join(lang_dir, 'clean_lemmas')
    if not os.path.exists(outdir_lemmas_dir):
        os.makedirs(outdir_lemmas_dir)

    # Read in the lemma files
    files = sorted(glob(os.path.join(lang_dir, 'corpus*', 'lemma', '*.txt')))
    print(files)
    assert len(files) == 2

    lemma_lines_by_name = Counter()

    outlines = []
    for f_i, infile in enumerate(files):
        print(infile)
        basename = os.path.basename(infile)
        name = basename.split('.')[0]
        with open(infile) as f:
            lines = f.readlines()
        lemma_lines_by_name[name] = len(lines)
        for l_i, line in enumerate(tqdm(lines)):
            # Assign line IDs based on line number
            line_id = name + '_' + str(l_i).zfill(8)

            text = line.strip()
            # replace multiple pound signs with a single one
            text = re.sub('#+', '#', text)

            if lang == 'ger':
                # Replace german long S's with regular S's, and fix some diacritics
                text = re.sub('ſ', 's', text)
                text = re.sub('uͤ', 'u', text)
                text = re.sub('oͤ', 'o', text)
                text = re.sub('aͤ', 'a', text)

            # If dealing with English lemmas, index first, then drop the POS tags on decorated tokens
            if lang == 'eng':
                tokens = text.split()
                # index the targets
                for t_i, token in enumerate(tokens):
                    if token in targets:
                        target_indices[token].append((line_id, t_i))
                # Drop the POS tags
                tokens = [t.split('_')[0] for t in tokens]
                # index again, ignoring POS tags
                for t_i, token in enumerate(tokens):
                    if token in targets_nopos:
                        target_indices_nopos[token].append((line_id, t_i))
                text = ' '.join(tokens)
            # Otherwise, replace underscores with spaces first, then index
            else:
                text = re.sub('_', ' ', text)
                tokens = text.split()
                # Drop URLs
                tokens = [t for t in tokens if not t.startswith('http')]
                for t_i, token in enumerate(tokens):
                    if token in targets:
                        target_indices[token].append((line_id, t_i))
            
            outlines.append({'id': line_id, 'text': text, 'source': name})

    with open(os.path.join(outdir_lemmas_dir, 'all.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')

    outfile = os.path.join(outdir_lemmas_dir, 'target_indices.json')
    with open(outfile, 'w') as f:
        json.dump(target_indices, f)

    if lang == 'eng':
        outfile = os.path.join(outdir_lemmas_dir, 'target_indices_nopos.json')
        with open(outfile, 'w') as f:
            json.dump(target_indices_nopos, f)

    for target in sorted(target_indices):
        if lang == 'eng':
            print(target, len(target_indices[target]), len(target_indices_nopos[target.split('_')[0]]))
        else:
            print(target, len(target_indices[target]))

    # Now repeat for the original tokens
    outdir_tokens_dir = os.path.join(lang_dir, 'clean_tokens')
    if not os.path.exists(outdir_tokens_dir):
        os.makedirs(outdir_tokens_dir)

    # Read in the orignal files
    files = sorted(glob(os.path.join(lang_dir, 'corpus*', 'token', '*.txt')))
    print(files)
    assert len(files) == 2

    outlines = []
    for f_i, infile in enumerate(files):
        print(infile)
        basename = os.path.basename(infile)
        name = basename.split('.')[0]
        with open(infile) as f:
            lines = f.readlines()

        assert len(lines) == lemma_lines_by_name[name]
        for l_i, line in enumerate(tqdm(lines)):
            # Assign line IDs based on line number
            line_id = name + '_' + str(l_i).zfill(8)

            text = line.strip()
            # replace multiple pound signs with a single one
            text = re.sub('#+', '#', text)

            if lang == 'ger':
                # Replace german long S's with regular S's, and fix some diacritics
                text = re.sub('ſ', 's', text)
                text = re.sub('uͤ', 'u', text)
                text = re.sub('oͤ', 'o', text)
                text = re.sub('aͤ', 'a', text)

            # Replace underscores with spaces
            text = re.sub('_', ' ', text)
            tokens = text.split()
            # Drop URLs
            tokens = [t for t in tokens if not t.startswith('http')]
            # Rejoin into a string
            text = ' '.join(tokens)
            
            outlines.append({'id': line_id, 'text': text, 'source': name})

    with open(os.path.join(outdir_tokens_dir, 'all.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')


if __name__ == '__main__':
    main()
