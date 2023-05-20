import os
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm


# Take the average of the individual GEMS annotations


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--gems-file', type=str, default=None,
                      help='.csv file with GEMS annotations: default=%default')

    (options, args) = parser.parse_args()

    gems_file = options.gems_file
    
    df = pd.read_csv(gems_file, index_col=0, header=0)

    gems_scores = df[['p1', 'p2', 'p3', 'p4', 'p5']].values
    gems_means = gems_scores.mean(axis=1)
    gems_index = list(df.index)

    with open(gems_file + '.mean.tsv', 'w') as f:
        for word_i, word in enumerate(gems_index):
            f.write(word + '\t' + str(gems_means[word_i]) + '\n')
    

if __name__ == '__main__':
    main()
