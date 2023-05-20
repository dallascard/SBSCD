import os
import re
import sys
import json
from glob import glob
from collections import Counter, defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import networkx.algorithms.community as nx_comm

from common.stopwords import get_stopwords
from common.misc import get_model_name, get_subdir


# Do clustering of replacements

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base directory: default=%default')
    parser.add_option('--subdir', type=str, default='subs_masked_nopos',
                      help='Sub directory with sub files: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                     help='model used for tokenization: default=%default')
    parser.add_option('--strip-accents', action="store_true", default=False,
                      help='Strip accents when tokenizing: default=%default')
    parser.add_option('--target-term', type=str, default=None,
                      help='Only embed one particular term: default=%default')
    parser.add_option('--resolution', type=float, default=1,
                      help='Resolution: default=%default')
    parser.add_option('--top-k', type=int, default=5,
                      help='Top-k terms to keep: default=%default')
    parser.add_option('--min-count', type=int, default=1,
                      help='Min count to add edges: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--include-self', action="store_true", default=False,
                      help="Include the token itself as a possible substitute: default=%default")

    (options, args) = parser.parse_args()

    basedir = options.basedir
    subdir = options.subdir
    lang = options.lang
    model = options.model
    target_term = options.target_term
    resolution = options.resolution
    top_k = options.top_k
    min_count = options.min_count
    seed = options.seed
    exclude_self = not options.include_self

    np.random.seed(seed)

    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    tokenized_dir = get_subdir(basedir, model_name)

    outdir = os.path.join(tokenized_dir, 'clusters')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    indir = os.path.join(tokenized_dir, subdir)

    all_files = sorted(glob(os.path.join(indir, '*.jsonlist')))
    if target_term is None:
        sub_files = sorted(glob(os.path.join(indir, '*.jsonlist')))
    else:
        sub_files = [os.path.join(indir, target_term + '_substitutes.jsonlist')]
    print(len(sub_files))
    
    sub_counter_by_term = defaultdict(Counter)

    print("Counting replacements")
    for infile in tqdm(all_files):
        basename = os.path.basename(infile)
        target = '_'.join(basename.split('_')[:-1])

        with open(infile) as f:
            lines = f.readlines()

        for line in lines:
            line = json.loads(line)
            lemma = target
            top_terms = line['top_terms'][:top_k+1]
            if exclude_self and lemma in top_terms:
                top_terms.remove(lemma)
            else:
                top_terms = top_terms[:top_k]
            sub_counter_by_term[target].update(top_terms)

    print("Doing clustering")
    for infile in sub_files:
        print(infile)
        basename = os.path.basename(infile)
        target = '_'.join(basename.split('_')[:-1])

        with open(infile) as f:
            lines = f.readlines()
        lines = [json.loads(line) for line in lines]

        edge_counter = Counter()
        node_counter = Counter()

        for line in lines:
            lemma = target
            top_terms = line['top_terms'][:top_k+1]
            if exclude_self and lemma in top_terms:
                top_terms.remove(lemma)
            else:
                top_terms = top_terms[:top_k]

            for t_i, term1 in enumerate(top_terms[:-1]):
                node_counter[term1] += 1
                for term2 in top_terms[t_i+1:]:
                    edge_counter[term1 + '||' + term2] += 1

        if len(sub_files) == 1:
            for pair, count in edge_counter.most_common(n=10):
                print(pair, count)
            print(len(edge_counter), sum(edge_counter.values()))

        if len(sub_files) == 1:
            print("Constructing graph")
        G = nx.Graph()

        for pair, count in edge_counter.items():
            if count >= min_count:
                term1, term2 = pair.split('||')
                G.add_edge(term1, term2, weight=count)

        if len(sub_files) == 1:
            print("Doing community detection")

        failed = False
        try:
            clusters = nx_comm.louvain_communities(G, seed=seed, resolution=resolution)
        except Exception as e:
            print(e)
            failed = True
            print("Skipping", target)

        if not failed:
            print(len(clusters))

            clusters_sorted = []
            cluster_counters = []

            sizes = [len(cluster) for cluster in clusters]
            order = np.argsort(sizes)[::-1]
            for i in order:
                cluster = clusters[i]
                clusters_sorted.append(sorted(cluster))
                counter = Counter({term:sub_counter_by_term[target][term] for term in cluster})
                cluster_counters.append(counter)
                if len(sub_files) == 1:
                    print(len(cluster), counter.most_common(n=5))            

            cluster_assignments = []
            count_by_cluster = Counter()

            for line in lines:
                line_id = line['line_id']
                top_terms = line['top_terms'][:top_k+1]
                if exclude_self and lemma in top_terms:
                    top_terms.remove(lemma)
                else:
                    top_terms = top_terms[:top_k]
                scores = []
                for cluster in clusters_sorted:
                    scores.append(jaccard_similarity(set(cluster), set(top_terms)))

                most_similar = int(np.argmax(scores))
                cluster_assignments.append({'id': line_id, 'cluster': most_similar})
                count_by_cluster[most_similar] += 1

            if len(sub_files) == 1:
                print("Active clusters")
                for cluster_index, count in count_by_cluster.most_common():
                    cluster = clusters_sorted[cluster_index]
                    counter = Counter({term:sub_counter_by_term[target][term] for term in cluster})
                    print(cluster_index, count, counter.most_common(n=5))
                

            with open(os.path.join(outdir, target + '_clusters.jsonlist'), 'w') as f:
                for c_i, cluster in enumerate(cluster_counters):
                    f.write(str(c_i) + '\t' + str(count_by_cluster[c_i]) + '\t' + json.dumps(cluster.most_common()) + '\n')
                #json.dump(clusters_sorted, f)

            with open(os.path.join(outdir, target + '_assignments.json'), 'w') as f:
                for line in cluster_assignments:
                    f.write(json.dumps(line) + '\n')


def jaccard_similarity(a, b):
    return (len(a.intersection(b)) / len(a.union(b)))


if __name__ == '__main__':
    main()
