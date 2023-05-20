import os
import re
import json
import string
from glob import glob
from optparse import OptionParser
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.misc import get_model_name, get_subdir


# Align the lemmatized and original corpora from SemEval


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--basedir', type=str, default='/data/dalc/SemEval/2020/task1_semantic_change/',
                      help='Base directory: default=%default')
    parser.add_option('--lang', type=str, default='eng',
                      help='Language [eng|ger|lat|swe]: default=%default')
    parser.add_option('--model', type=str, default='bert-large-uncased',
                     help='model used for tokenization: default=%default')
    parser.add_option('--strip-accents', action="store_true", default=False,
                      help='Strip accents when tokenizing: default=%default')

    (options, args) = parser.parse_args()

    basedir = options.basedir
    lang = options.lang
    model = options.model
    strip_accents = options.strip_accents


    model_name = get_model_name(model)

    basedir = os.path.join(basedir, 'semeval2020_ulscd_' + lang)

    lemmas_dir = os.path.join(basedir, 'clean_lemmas')
    tokenized_dir = get_subdir(basedir, model_name, strip_accents)

    lemmas_file = os.path.join(lemmas_dir, 'all.jsonlist')
    tokenized_file =  os.path.join(tokenized_dir, 'all.jsonlist')
    print("Tokenized file:", tokenized_file)
        
    with open(lemmas_file) as f:
        lines = f.readlines()
    print("Lemma lines:", len(lines))

    line_ids = []
    lemmatized_tokens = []
    for line in tqdm(lines):
        line = json.loads(line)
        text = line['text'].lower()
        tokens = text.split()
        lemmatized_tokens.append(tokens)
        line_ids.append(line['id'])

    with open(tokenized_file) as f:
        lines = f.readlines()
    print("Tokenized lines:", len(lines))

    tokenized_tokens = []
    for line in tqdm(lines):
        line = json.loads(line)
        tokens = line['tokens']
        tokens = [re.sub('##', '', token).lower() for token in tokens]
        tokenized_tokens.append(tokens)

    assert len(lemmatized_tokens) == len(tokenized_tokens)

    tokenized_strings = []
    lemmatized_strings = []
    scores = []
    word_match_scores = []
    alignments = []

    outlines = []

    # process each tokenized line
    for line_i, tokens in tqdm(enumerate(tokenized_tokens)):
        # get the corresponding lemmatized line
        lemmas = lemmatized_tokens[line_i]

        tokens_clean, tokens_removed = remove_punctuation(tokens)
        lemmas_clean, lemmas_removed = remove_punctuation(lemmas)

        alignment = align_lists(tokens_clean, lemmas_clean)

        corrected_alignment = fix_alignment(tokens_removed, lemmas_removed, alignment)

        # score the mapping
        try:
            score = score_alignment(tokens, lemmas, corrected_alignment)
        except IndexError as e:
            print("Index error:", e)
            print(line_i)
            print(tokens)
            print(lemmas)
            print(tokens_clean)
            print(lemmas_clean)
            print(alignment)
            print(corrected_alignment)
            raise e
        word_match_score = score_word_matching(tokens, lemmas, corrected_alignment)
        tokenized_strings.append(' '.join(tokens))
        lemmatized_strings.append(' '.join(lemmas))
        scores.append(score)
        word_match_scores.append(word_match_score)
        alignments.append(corrected_alignment)
    
        outlines.append({'id': line_ids[line_i], 'tokens': ' '.join(tokens), 'lemmas': ' '.join(lemmas), 'score': score, 'word_match_score': word_match_score, 'alignment': sorted(corrected_alignment)})

    df = pd.DataFrame()
    df['id'] = line_ids
    df['tokens'] = tokenized_strings
    df['lemmas'] = lemmatized_strings
    df['score'] = scores
    df['word_match_score'] = word_match_scores
    df['alignment'] = alignments
    
    df.to_csv(os.path.join(tokenized_dir, 'matching.csv'))

    with open(os.path.join(tokenized_dir, 'matching.jsonlist'), 'w') as f:
        for line in outlines:
            f.write(json.dumps(line) + '\n')
    

def score_word_matching(list_one, list_two, mapping):
    # Score the alignment of two lists, using edit distance
    score = 0
    for i, pair in enumerate(mapping):
        index1, index2 = pair
        if index1 < 0:
            score += 1
        elif index2 < 0:
            score += 1
        elif list_one[index1] is None or list_two[index2] is None:
            score += 1
        elif list_one[index1] != list_two[index2]:
            score += 1
    return score


def score_alignment(list_one, list_two, mapping, debug=False, ignore_chars=',.;?!:()"'):
    # Score the alignment of two lists, using edit distance
    score = 0
    for i, pair in enumerate(mapping):
        index1, index2 = pair
        if index1 < 0:
            score += len(list_two[index2])
            if debug:
                print('---', list_two[index2], len(list_two[index2]))
        elif index2 < 0:
            score += len(list_one[index1])
            if debug:
                print(list_one[index1], '---', len(list_one[index1]))
        elif list_one[index1] is None:
            if list_two[index2] in ignore_chars:                
                dist = 0
            else:
                dist = len(list_two[index2])
            score += dist
            if debug:
                print('---', list_two[index2], score)
        elif list_two[index2] is None:
            if list_one[index1] in ignore_chars:
                dist = 0
            else:
                dist = len(list_one[index1])
            score += dist
            if debug:
                print(list_one[index1], '---', dist)
        elif list_one[index1] != list_two[index2]:
            dist = levenshteinDistance(list_one[index1], list_two[index2])+1
            if debug:
                print(list_one[index1], list_two[index2], dist)
            score += dist
        elif debug:
            print(list_one[index1], list_two[index2], 0)
    return score
        
        
def levenshteinDistance(s1, s2):
    # Compute the edit distance between the strings

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    max_dist = max(len(s1), len(s2))
    return min(max_dist, distances[-1])


def remove_punctuation(tokens, punct=',.;?!:()"'):
    # remove punctuation from a list of tokens
    removed = {}
    for t_i, t in enumerate(tokens):
        if t in punct:
            removed[t_i] = t
    clean = [t for t_i, t in enumerate(tokens) if t_i not in removed]
    return clean, removed


def fix_alignment(tokens_removed, lemmas_removed, alignment):

    for t_i, token in tokens_removed.items():
        corrected = []
        for pair in alignment:            
            if pair[0] < t_i:
                corrected.append(pair)
            else:
                corrected.append((pair[0]+1, pair[1]))
        alignment = corrected[:]        
    alignment.extend([(t_i, -1) for t_i, token in tokens_removed.items()])
    alignment = sorted(alignment)
                
    for t_i, lemma in lemmas_removed.items():
        corrected = []
        for pair in alignment:
            if pair[1] < t_i:
                corrected.append(pair)
            else:
                corrected.append((pair[0], pair[1]+1))
        alignment = corrected[:]
    alignment.extend([(-1, t_i) for t_i, lemma in lemmas_removed.items()])
    alignment = sorted(alignment)
        
    return alignment
    

def align_sublists_on_exact_matches(tokens, lemmas, start_token=0, end_token=None, start_lemma=0, end_lemma=None):
    """Compare subsets of two lists and match the common unique tokens"""

    local_alignment = []
    
    if end_token is None:
        end_token = len(tokens)
    if end_lemma is None:
        end_lemma = len(lemmas)

    # find the tokens that are common to both, but only occur once in each
    unique_tokens = set([t for t, c in Counter(tokens[start_token:end_token]).items() if c == 1])
    unique_lemmas = set([t for t, c in Counter(lemmas[start_lemma:end_lemma]).items() if c == 1])
    common_unique = unique_tokens.intersection(unique_lemmas)
    
    shared_token_indices = []
    for i, token in enumerate(tokens[start_token:end_token]):
        if token in common_unique:
            shared_token_indices.append((start_token+i, token))

    shared_lemma_indices = []
    for i, lemma in enumerate(lemmas[start_lemma:end_lemma]):
        if lemma in common_unique:
            shared_lemma_indices.append((start_lemma+i, lemma))

    lemma_offset = 0
    for i, pair, in enumerate(shared_token_indices):
        token_index, token = pair            
        # make sure there are still more lemmas to compare against
        if i + lemma_offset < len(shared_lemma_indices):
            lemma_index, lemma = shared_lemma_indices[i+lemma_offset]
            # if the token does not match the lemma, consider skipping
            if token != lemma:
                # if there are enough lemmas, check if the token matches the next one
                if i+lemma_offset+1 < len(shared_lemma_indices) and token == shared_lemma_indices[i+lemma_offset+1][1]:
                    # if so, increase the lemma offset and use the next lemma
                    lemma_offset += 1
                    lemma_index, lemma = shared_lemma_indices[i+lemma_offset]
                # alternatively, if there are enough tokens, check if the lemma matches the next one
                elif i+1 < len(shared_token_indices) and lemma == shared_token_indices[i+1][1] and i+lemma_offset+1 > 0:
                    # if so, skip this pair, and decrease the lemma offset
                    lemma_offset -= 1
            # if the token now matches the lemma (if we added to lemma offset), add the alignment pair
            if token == lemma:
                local_alignment.append((token_index, lemma_index))

    return local_alignment



def align_sublists_on_partial_matches(long_list, short_list, long_start=0, long_end=None, short_start=0, short_end=None):
    """
    Compare subsets of two lists and match the common unique tokens
    Assume len(long_list) >= len(short_list)
    """
    
    if long_end is None:
        long_end = len(long_list)
    if short_end is None:
        short_end = len(short_list)
    
    n_long = long_end - long_start
    n_short = short_end - short_start
    
    if n_long == n_short:
        alignment = [(long_start+offset, short_start+offset) for offset in range(n_long)]
        return alignment
    else:
        short_copy = short_list[:]

        # Add Null tokens into matching one by one until there is no more point in doing so
        for iteration in range(n_long-n_short):
            #print("Iteration", iteration)
            scores = []
            # consider adding the null token in each position in the short list
            for null_offset in range(short_end - short_start + 1):
                past_null_offset = 0                
                local_alignment = []
                for token_offset in range(long_end-long_start):
                    #print(token_offset, past_null_offset, token_offset == null_offset, short_start+token_offset-past_null_offset >= short_end, long_start+token_offset, short_start+token_offset-past_null_offset, len(short_copy))
                    if token_offset >= null_offset:
                        past_null_offset = 1
                    if token_offset == null_offset:
                        local_alignment.append((long_start+token_offset, -1))
                    elif short_start+token_offset-past_null_offset >= short_end:
                        local_alignment.append((long_start+token_offset, -1))
                    elif short_copy[short_start + token_offset - past_null_offset] is None:
                        local_alignment.append((long_start+token_offset, -1))
                    else:
                        local_alignment.append((long_start+token_offset, short_start+token_offset-past_null_offset))
                score = score_alignment(long_list, short_copy, local_alignment)

                scores.append(score)
            best_position = np.argmin(scores)
            short_copy.insert(short_start+best_position, None)
            short_end += 1            

        # convert back to original indexing of short
        corrected_alignment = []
        offset = 0
        for token_offset, long_token in enumerate(long_list[long_start:long_end]):
            short_index = short_start + token_offset
            short_token = short_copy[short_index]
            if short_token is None:
                offset += 1
                corrected_alignment.append((long_start+token_offset, -1))
            else:
                corrected_alignment.append((long_start+token_offset, short_start+token_offset-offset))

        return corrected_alignment


            
def align_lists(tokens, lemmas):
    done = False
    n_tokens = len(tokens)
    n_lemmas = len(lemmas)
    alignment = [(-1, -1), (n_tokens, n_lemmas)]
    iteration = 0
    while not done:
        new_alignments = []
        for pair_i, aligned_pair in enumerate(alignment[:-1]):
            token_index, lemma_index = aligned_pair
            next_pair = alignment[pair_i+1]
            next_token_index, next_lemma_index = next_pair        
            if next_token_index - token_index > 1 and next_lemma_index - lemma_index > 1:
                local_alignments = align_sublists_on_exact_matches(tokens, lemmas, token_index+1, next_token_index, lemma_index+1, next_lemma_index)
                new_alignments.extend(local_alignments)
        if len(new_alignments) == 0:
            done = True
        else:
            alignment = sorted(alignment + new_alignments)
            iteration += 1
        if iteration > 20:
            raise RuntimeError("Exceeded 20 iterations")
            
    # Now go through and do alignment of sublists on imperfect matches
    for pair_i, aligned_pair in enumerate(alignment[:-1]):
        token_index, lemma_index = aligned_pair
        next_pair = alignment[pair_i+1]
        next_token_index, next_lemma_index = next_pair        
        if next_token_index - token_index > 1 or next_lemma_index - lemma_index > 1:
            if next_token_index - token_index >= next_lemma_index - lemma_index:                
                local_alignments = align_sublists_on_partial_matches(tokens, lemmas, token_index+1, next_token_index, lemma_index+1, next_lemma_index)
            else:
                local_alignments = align_sublists_on_partial_matches(lemmas, tokens, lemma_index+1, next_lemma_index,  token_index+1, next_token_index)
                local_alignments = [(pair[1], pair[0]) for pair in local_alignments]

            new_alignments.extend(local_alignments)
    alignment = sorted(alignment + new_alignments)

    return alignment[1:-1]
    
    

if __name__ == '__main__':
    main()
