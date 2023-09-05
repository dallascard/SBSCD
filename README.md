## Substitution-based Semantic Change Detection

This repo accompanies the paper "Substitution-based Semantic Change Detection using Contextual Embeddings", published at ACL 2023.

It contains code to perform substitution-based semantic change detection for target terms on a generic corpus, as well as code to replicate the experiments in the paper.

Using this approach to semenatic change detection involves running a pipeline of scripts, which cover tokenization, continued masked langauge model pretraining, indexing terms, getting substitutes, and finally computing amount of change using Jensen-Shannon divergence. Intermediate products produced include the substitutes for each sampled mention, which can be used for additional analyses, as is done in the paper.

### Requirements

The python packages used (along with the versions used) for this repo include:
- tqdm
- numpy
- scipy
- pandas
- spacy
- torch
- transformers

In addition, `networkx` was used for Louvain community detection in the clustering analysis, and `matplotlib` was used for making Figure 1. An `environment.yaml` file has also been included for replication of the environment, which it should be possible to build using `conda env create -f environment.yml`.


### General usage

To replicate the experiments from the paper, see the other sections below. For more general usage, follow the instructions here.

To begin, this assumes that you have two text corpora, along with (optionally) a list of target words as a `.tsv` file. The input format for the text corpora is that everything (both corpora together) should be in a single .jsonlist file. Each document should be a json object with an `id` field (containing a unique ID), a `text` field (containing the raw text), and a `source` field (indicating which of the two corpora it is part of). The names of these in your file can be specified on the command line.

In the general usage, you can use this to do semantic change detection on words, lemmas, or part-of-speech-tagged lemmas. If using lemmas or tags, you will also need to lemmatize the text using the script provided. The pipeline for semantic change detection is as follows:

1. Tokenize the text using an appropriate model:
`python -m general.tokenize --basedir <basedir> --infile <infile.jsonlist> --model <model>`
`<basedir>` is where all the work will happen
`<infile.jsonlist>` is the jsonlist file with all of the documents as json objects
`<model>` is the name of the model to use, such as `bert-large-uncased`.
If necesary, you can also specify the names of the fields in the jsonlist file using `--id-field`, `--source-field` and `--text-field`.

2. Optionally, lemmatize the text using an appropriate spacy model:
`python -m general.lemmatize --basedir <basedir> --model <model> --spacy <spacy_model>`
where `<basedir>` and `<model>` are the same as above, and `<spacy_model>` is an appropriate (language-specific) model, such as `en_core_web_sm`

3. Export the data for continued masked language model training:
`python -m general.export_for_pretraining --basedir <basedir> --model <model>`

4. Do the continued masked language model training:
`python -m general.run_mlm --basedir <basedir> --model <model> --cache-dir <cache_dir>`
where `<cache_dir>` allows you to specify where to store cached files from huggingface

5. Index terms (both target terms, and random background terms):
`python -m general.index_terms --basedir <basedir> --model <model> --targets-file <targets_file>`
Here, `<targets_file>` (optional) is the list of target words that you definitely want to be included. Other relevant parameters include:
- `--lemmas`: set to use lemmas rather than words
- `--pos`: Use POS tags from spacy to distinguish between word forms with different tags
- `--min-count`: Exclude terms from random selection with less than this many occurrences
- `--min-count-per-corpus`: Require at least this count in each corpus
- `--max-terms`: Only take a random sample of up to this many unique terms
- `--max-tokens`: Limit the number of tokens sampled per term

6. Embed the terms and get substitutes:
`python -m general.get_substitutes --basedir <basedir> --model <model>`
Once again, use the `--lemmas` and `--pos` flags if you used them above. You can also point to a file with stopwords to be excluded from the substitutes, using `--stopwords-file`. 
Other relevant parameters include `--max-window-size` and `--top-k`.

7. Compute JSD and rescale to get final semantic change predictions:
`python -m general.compute_jsds --basedir <basedir> --model <model>`
Once again, use the `--lemmas` and `--pos` flags if you used them above. Other relevant factors include `--top-k` and `--window-factor` (see paper for details).

The output of this will be a file called `<basedir>/embeddings_<model>/jsd_scores.csv` (and an equivalent `.json` file) containing the scaled estimate for all terms, including all target terms. 

Additional scripts that may be useful include `evaluate.py` (which allows you to evaluate against gold scores) and `get_top_repalcements.py` (which gather the top replacement terms from each corpus).

### Replicating GEMS experiments

To replicate the results on GEMS, one first needs to obtain the COHA dataset, as well as the GEMS target scores. The former can be obtained (for a fee) from Mark Davies via [this website](https://www.english-corpora.org/coha/). The latter were obtained from the authors of Gulordava and Baroni (2011).

The following pipeline can be used to replicate these experiments:

1. Import the COHA data into the appropriate format:
`python -m coha.import_coha --basedir <basedir>`
where `<basedir>` is the directory contianing the raw COHA data. This will output a file named `all.jsonlist` in the `clean` subdirectory of `<basedir>`

2. Convert the individual rater judgements of semantic change (obtained from the original authors):
`python -m gems.average_gems_scores.py --gems-file <gems_file.csv>`
where `<gems_file.csv>` is the file of ratings obtained from the original GEMS authors. This will produce a file with the same name, with the additional suffix `.mean.csv`. This will be the `<targets-file>` for the pipeline below.

3. Tokenize the raw text:
`python -m coha.tokenize --basedir <basedir>`

4. Export plain text for continued masked language model training of the base model:
`python -m coha.export_for_pretraining --basedir <basedir>`

5. Run continued masked language model training of the base model:
`python -m coha.run_mlm --basedir <basedir>`

6. Index the target terms:
`python -m coha.index_targets --basedir <basedir> --targets-file <targets.tsv>`
where `<targets.tsv>` is the output of `gems.average_gems_scores.py`

7. Index random background terms:
`python -m coha.index_random_tokens --basedir <basedir> --targets-file <targets.tsv>`

8. Get substitutes for target terms:
`python -m coha.get_substitutes --basedir <basedir>`

9. Get substitutes for random terms:
`python -m coha.get_substitutes --basedir <basedir> --random-targets`

10. Compute Scaled JSD:
`python -m coha.compute_jsds --basedir <basedir>`
This will produce a .csv and .json file with the scaled JSD scores per target term. These will be located in a subdirectory of `<basedir>` named `tokenized_<model>`.

11. Finally, to score the estimates of semantic change, run:
`python -m coha.evaluate --basedir <basedir> --targets-file <targets-file>`


### Replicating SemEval experiments

Becuase the SemEval targets are provided as lemmas, with multiple langauges, the pipeline for the experiments with these datasets is slightly different, even though the overall method is the same. In paritcular, the steps are essentially the same, except that there is an additional alignment step between the lemmatized and non-lemmatized text. (This is unlike the general usage section above, which does the lemmatization using spacy).

The SemEval data can be downloaded (separately for each langauge) from here: https://www.ims.uni-stuttgart.de/en/research/resources/corpora/sem-eval-ulscd/

For each SemEval script, the language can be specified using the `--lang` argument, and the model using the `--model` arguement. The pipeline for SemEval is as follows (and must be run separately for each langauge):

1. Import the data into the proper format:
`python -m semeval.import_semeval --basedir <basedir>`

2. Tokenize the non-lemmatized text:
`python -m semeval.tokenize_semeval --basedir <basedir>`

3. Do the alignment between the lemmatized and (tokenized) non-lemmatized text:
`python -m semeval.do_alignment --basedir <basedir>`

4. Export plain text for continued masked language model training of the base model:
`python -m semeval.export_for_pretraining --basedir <basedir>`

5. Run continued masked language model training of the base model:
`python -m semeval.run_mlm --basedir <basedir>`

6. Index the target terms:
`python -m semeval.index_targets --basedir <basedir>`

7. Index random background terms:
`python -m semeval.index_random_targets --basedir <basedir>`

8. Get substitutes for target terms:
`python -m semeval.get_substitutes --basedir <basedir>`

9. Get substitutes for random terms:
`python -m semeval.get_substitutes --basedir <basedir> --random-targets`

10. Evaluate:
`python -m semeval.evaluate_semeval --basedir <basedir>`

Scripts have also been included for various other parts the analysis:

1. Getting the top replacement terms:
`python -m semeval.get_top_replacements --basedir <basedir>`

2. Doing the clustering analysis:
`python -m semeval.do_clustering --basedir <basedir>`

3. Making Figure 1 in maina paper
`python -m semeval.make_figure --basedir <basedir>`



### Citation / Reference

```
@inproceedings{card-etal-2020-little,
    title = "Substitution-based Semantic Change Detection using Contextual Embeddings",
    author = "Dallas Card"
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics",
    year = "2023",
}
```