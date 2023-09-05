import os
import re
from glob import glob
from optparse import OptionParser


# Convert the targets to the format used by the general code in this repo

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base dir: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = 'eng'

    lang_dir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    # Read in the targets
    targets_file = os.path.join(lang_dir, 'truth', 'graded.txt')
    with open(targets_file) as f:
        lines = f.readlines()

    # replace SemEval tags iwth spaCy tags
    fixed_tags = []
    for line in lines:
        line = re.sub('_nn', '_NOUN', line)
        line = re.sub('_vb', '_VERB', line)
        fixed_tags.append(line)
    
    with open(os.path.join(lang_dir, 'targets_pos.txt'), 'w') as f:
        for line in fixed_tags:
            f.write(line)
    
    # also make a version with no tags
    untagged = []
    for line in lines:
        line = re.sub('_nn', '', line)
        line = re.sub('_vb', '', line)
        untagged.append(line)

    with open(os.path.join(lang_dir, 'targets_nopos.txt'), 'w') as f:
        for line in untagged:
            f.write(line)


if __name__ == '__main__':
    main()
