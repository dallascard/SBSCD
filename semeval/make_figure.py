
import os
import re
import json
from glob import glob
from subprocess import call
from collections import Counter, defaultdict
from optparse import OptionParser

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
from scipy.stats import spearmanr, pearsonr

from common.misc import get_model_name, get_subdir
from common.compute_jsd import compute_jsd_from_counters


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
    parser.add_option('--top-k', type=int, default=5,
                      help='Top-k terms to keep: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    strip_accents = options.strip_accents
    topk = options.top_k

    targets_subdir = 'subs_masked'
    if lang == 'eng':
        targets_subdir += '_nopos'
    random_subdir = 'random_subs_masked'

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    lemmas_dir = os.path.join(basedir, 'clean_lemmas')
    lemmas_file = os.path.join(lemmas_dir, 'all.jsonlist')
    tokenized_dir = get_subdir(basedir, model_name, strip_accents)
    tokenized_file = os.path.join(tokenized_dir, 'all.jsonlist')
    matching_file = os.path.join(tokenized_dir, 'matching.jsonlist')

    outdir = os.path.join(tokenized_dir, 'plots')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    target_files = sorted(glob(os.path.join(tokenized_dir, targets_subdir, '*.jsonlist')))
    print(len(target_files), "target files")
    if len(target_files) == 0:
        raise Exception("No target files found in", targets_subdir)

    with open(tokenized_file) as f:
        lines = f.readlines()
    lines = [json.loads(line) for line in lines]
    len(lines)

    source_by_id = {}
    source_counter = Counter()
    for line in tqdm(lines):
        source_counter[line['source']] += 1
        source_by_id[line['id']] = line['source']

    for s, c in source_counter.most_common():
        print(s, c)

    print("Loading lemmatized files file")    
    with open(lemmas_file) as f:
        lines = f.readlines()
    len(lines)
    
    token_counter = Counter()
    for line in tqdm(lines):
        line = json.loads(line)
        text = line['text']
        lemmas = text.split()
        token_counter.update(lemmas)        

    for t, c in token_counter.most_common(n=15):
        print(t, c)    

    sources = sorted(source_counter)
    sub_counter_by_term_by_source = {source: defaultdict(Counter) for source in source_counter}
    counter_by_term = Counter()

    top_subs_lists_by_term_source1 = defaultdict(list)
    top_subs_lists_by_term_source2 = defaultdict(list)

    for infile in target_files:
        count_by_source = Counter()
        basename = os.path.basename(infile)
        term = basename.split('_')[0]
        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            line_id = line['line_id']
            top_tokens = line['top_terms']
            source = source_by_id[line_id]
            sub_counter_by_term_by_source[source][term].update(top_tokens[:topk])
            count_by_source[source] += 1
            counter_by_term[term] += 1
            if source == sources[0]:
                top_subs_lists_by_term_source1[term].append(top_tokens[:topk])
            else:
                top_subs_lists_by_term_source2[term].append(top_tokens[:topk])
        print(term, len(sub_counter_by_term_by_source[sources[0]][term]), len(sub_counter_by_term_by_source[sources[1]][term]))        

    terms = sorted(sub_counter_by_term_by_source[sources[0]])
    jsds = []
    for term in terms:
        jsd = compute_jsd_from_counters(sub_counter_by_term_by_source[sources[0]][term], sub_counter_by_term_by_source[sources[1]][term])
        jsds.append(jsd)

    print("Loading labels")
    label_file = os.path.join(basedir, 'truth', 'graded.txt')
    label_df = pd.read_csv(label_file, header=None, index_col=None, sep='\t')
    label_df.columns = ['term', 'score']

    target_words = label_df['term'].values

    if lang == 'eng':
        target_words = [term.split('_')[0] for term in target_words]

    target_shifts = label_df['score'].values
    target_shift_by_word = dict(zip(target_words, target_shifts))
    print(len(target_words), len(target_shift_by_word))

    df_terms = sorted(target_words)
    for t_i, term in enumerate(terms):
        print(term, df_terms[t_i])

    shifts_sorted = [target_shift_by_word[re.sub('##', '', t)] for t in terms]

    df2 = pd.DataFrame()
    df2['term'] = terms
    df2['jsd'] = jsds
    df2['shift'] = shifts_sorted

    random_files = sorted(glob(os.path.join(tokenized_dir, random_subdir, '*.jsonlist')))
    print(len(random_files), "random files")
    if len(random_files) == 0:
        raise Exception("No target files found in", random_subdir)

    sub_counter_by_term_by_source_random = {source: defaultdict(Counter) for source in source_counter}

    counter_by_term_random = Counter()

    for infile in tqdm(random_files):
        count_by_source = Counter()
        counter_by_source = defaultdict(Counter)
        basename = os.path.basename(infile)
        term = basename.split('_')[0]
        with open(infile) as f:
            lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            line_id = line['line_id']
            top_tokens = line['top_terms']
            source = source_by_id[line_id]
            count_by_source[source] += 1
            counter_by_source[source].update(top_tokens[:topk])
            counter_by_term_random[term] += 1

        if count_by_source[sources[0]] > 0 and count_by_source[sources[1]] > 0:
            for source in sources:
                sub_counter_by_term_by_source_random[source][term].update(counter_by_source[source])

    print(len(sub_counter_by_term_by_source_random[sources[0]]),len(sub_counter_by_term_by_source_random[sources[1]]) )

    terms_random = sorted(sub_counter_by_term_by_source_random[sources[0]])
    jsds_random = []
    for term in tqdm(terms_random):
        jsd = compute_jsd_from_counters(sub_counter_by_term_by_source_random[sources[0]][term], sub_counter_by_term_by_source_random[sources[1]][term])
        jsds_random.append(jsd)    

    df3 = pd.DataFrame()
    df3['term'] = terms + terms_random
    df3['JSD'] = jsds + jsds_random
    df3['Count in corpus'] = [token_counter[t] for t in terms] + [token_counter[t] for t in terms_random]
    df3['subset'] = ['target'] * len(terms) + ['random'] * len(terms_random)
    df3['opacity'] = [1.0] * len(terms) + [0.2] * len(terms_random)

    relative_jsds = []
    n_neighbours = []

    all_terms = terms + terms_random
    all_jsds = jsds + jsds_random
    all_counts = [counter_by_term[t] for t in terms] + [counter_by_term_random[t] for t in terms_random]

    all_jsds_by_term = dict(zip(all_terms, all_jsds))

    for target_i, target in tqdm(enumerate(terms)):
        target_count = token_counter[target]
        target_jsd = jsds[target_i]
        nearby_terms = [term for term in all_terms if target_count/2 <= token_counter[term] <= target_count*2 and term != target]
        n_neighbours.append(len(nearby_terms))
        nearby_jsds = np.array([all_jsds_by_term[term] for term in nearby_terms])
        quartile = np.mean(target_jsd >= nearby_jsds)
        relative_jsds.append(quartile)
        print(target, target_count, target_jsd, len(nearby_terms), quartile, shifts_sorted[target_i])

    print(min(n_neighbours))    

    df4 = pd.DataFrame()
    df4['term'] = terms
    df4['jsd'] = jsds
    df4['relative_jsd'] = relative_jsds
    df4['score'] = shifts_sorted

    print(len(shifts_sorted), len(relative_jsds))
    print(pearsonr(shifts_sorted, relative_jsds))
    print(spearmanr(shifts_sorted, relative_jsds))

    ols_model = sm.OLS(df4['score'].values, sm.add_constant(df4['relative_jsd']))
    est = ols_model.fit()

    fig, axes = plt.subplots(nrows=2, figsize=(5, 6))
    plt.subplots_adjust(hspace=0.35)

    ax = axes[0]

    subset = df3[df3['subset'] == 'random']
    ax.scatter(subset['Count in corpus'].values, subset['JSD'].values, label='random', alpha=0.2, s=10, rasterized=True)

    subset = df3[df3['subset'] == 'target']
    ax.scatter(subset['Count in corpus'].values, subset['JSD'].values, label='target', alpha=0.8, s=10, marker='s', rasterized=True)

    if lang == 'eng':
        labeled = {'tree', 'plane', 'gas', 'graft', 'tip', 'ounce', 'head', 'bit', 'fiction'}
    else:
        labeled = set()

    for word in labeled:
        row = subset[subset['term'] == word]
        ax.text(row['Count in corpus'], row['JSD'], re.sub('##', '', word))

    ax.set_xscale('log')
    ax.set_ylim(0.2, 1.05)
    ax.set_xlim(100, 3e5)
    ax.set_xlabel('Count in corpus (log scale)')
    ax.set_ylabel('Raw JSD')
    ax.legend()

    ax = axes[1]

    ax.scatter(df4['relative_jsd'], df4['score'], s=10, rasterized=True)

    for word in labeled:
        row = df4[df4['term'] == word]
        if word == 'tree':
            ax.text(row['relative_jsd']+0.01, row['score']-0.05, re.sub('##', '', word))
        else:
            ax.text(row['relative_jsd']+0.01, row['score']+0.01, re.sub('##', '', word))

    X = pd.DataFrame()
    x_pred = np.linspace(0, 1, 20)
    X['relative_jsd'] = x_pred
    y_pred = est.predict(sm.add_constant(X))
    ax.plot(x_pred,y_pred, alpha=0.5)

    pred = est.get_prediction(sm.add_constant(X)).summary_frame()
    ax.fill_between(x_pred,pred['mean_ci_lower'],pred['mean_ci_upper'], color='C0', alpha=0.1)

    ax.plot([0, 1], [0, 1], 'k:', alpha=0.25)

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylabel('Human rating')
    ax.set_xlabel('Scaled JSD')
    outfile = os.path.join(outdir, 'jsd_combined_' + lang + '.jpg')
    plt.savefig(outfile, bbox_inches='tight', dpi=1200)
    plt.show()


if __name__ == '__main__':
    main()
