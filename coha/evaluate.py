import os
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

from scipy.stats import spearmanr, pearsonr

from common.misc import get_model_name, get_subdir


## TO DO; revise

# Do the evaluation of substitute method, with and without scaling by background term

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/COHA/',
                      help='Base directory: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                      help='Model that was used for tokenization: default=%default')
    parser.add_option('--targets-file', type=str, default='GEMS_evaluation_dataset.csv.mean.tsv',
                      help='File with average GEMS ratings: default=%default')
    (options, args) = parser.parse_args()

    basedir = options.basedir
    model = options.model
    targets_file = options.targets_file

    model_name = get_model_name(model)

    tokenized_dir = get_subdir(basedir, model_name)
    jsd_file = os.path.join(tokenized_dir, 'jsd_scores.json')

    with open(jsd_file) as f:
        jsd_scores = json.load(f)

    with open(targets_file) as f:
        lines = f.readlines()

    target_words = [line.split('\t')[0] for line in lines]
    target_scores = [float(line.split('\t')[1]) for line in lines]

    print("Excluding extracellular:")

    target_scores_subset = [target_scores[i] for i, word in enumerate(target_words) if word != 'extracellular']
    jsd_subset = [jsd_scores['raw'][word] for word in target_words if word != 'extracellular']    
   
    print("\nUncorrected results:")
    print("Lengths:", len(target_scores_subset), len(jsd_subset))
    print("Pearson:", pearsonr(target_scores_subset, jsd_subset))
    print("Spearman:", spearmanr(target_scores_subset, jsd_subset))

    if jsd_scores['scaled'] is not None:
        jsd_subset = [jsd_scores['scaled'][word] for word in target_words if word != 'extracellular']
    
        print("Scaled results:")
        print("Lengths:", len(target_scores_subset), len(jsd_subset))
        print("Pearson:", pearsonr(target_scores_subset, jsd_subset))
        print("Spearman:", spearmanr(target_scores_subset, jsd_subset))

    print()
    print("Excluding extracellular, assay, mediaeval, and sulphate:")

    exclude = {'extracellular', 'assay', 'mediaeval', 'sulphate'}
    target_scores_subset = [target_scores[i] for i, word in enumerate(target_words) if word not in exclude]
    jsd_subset = [jsd_scores['raw'][word] for word in target_words if word not in exclude]
   
    print("\nUncorrected results:")
    print("Lengths:", len(target_scores_subset), len(jsd_subset))
    print("Pearson:", pearsonr(target_scores_subset, jsd_subset))
    print("Spearman:", spearmanr(target_scores_subset, jsd_subset))

    if jsd_scores['scaled'] is not None:
        jsd_subset = [jsd_scores['scaled'][word] for word in target_words if word not in exclude]
    
        print("Scaled results:")
        print("Lengths:", len(target_scores_subset), len(jsd_subset))
        print("Pearson:", pearsonr(target_scores_subset, jsd_subset))
        print("Spearman:", spearmanr(target_scores_subset, jsd_subset))

    print()

if __name__ == '__main__':
    main()
