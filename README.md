## Substitution-based Semantic Change Detection

This repo accompanies the paper "Substitution-based Semantic Change Detection using Contextual Embeddings", published at ACL 2023.

It contains code to perform substitution-based semantic change detection for target terms on a generic corpus, as well as code to replicate the experiments in the paper.

Using this approach to semenatic change detection involves running a pipeline of scripts, which cover tokenization, continued masked langauge model pretraining, indexing terms, getting substitutes, and finally computing amount of change using Jensen-Shannon divergence. Intermediate products produced include the substitutes for each sampled mention, which can be used for additional analyses, as is done in the paper.

#### Requirements

The python packages used (along with the versions used) for this repo include:
- numpy
- scipy
- pandas
- tqdm
- torch
- transformers

In addition, `networkx` was used for Louvain community detection in the clustering analysis, and `matplitlib` was used for making Figure 1


### Doing semantic change detection

The generic approach assumes that one has a collection of documents from two different time periods (or sources) in jsonlist format, where each document is a json object with fields for text, source, and unique ID. By default, the assumption is these fields are named `test`, `group`, and `id`, but these can be set using command line options. (Scripts have been provided to coerce the experimental datasets used in the paper into this format). This also assumes that one has a list of target terms of interest in a text file (<targets.tsv>), with one word per line. By default, these terms are assumed to NOT be lemmatized (i.e., different word forms will be indexed separately), and not associated with part of speech tags (i.e., different syntactic forms of the same term will be treated equivalently).

For a generic corpus in the above format, the pipeline assumes that one has a working directory (`<basedir>`). The masked language model to be used can be set using command line options (`bert-large-uncased` by default). The pipeline is as follows:

1. Tokenize the raw text:
`python -m general.tokenize --basedir <basedir> --infile <infile.jsonlist>`

2. Export plain text for continued masked language model training of the base model:
`python -m general.export_for_pretraining --basedir <basedir>`

3. Run continued masked language model training of the base model:
`python -m general.run_mlm --basedir <basedir>`

4. Index the target terms:
`python -m general.index_targets --basedir <basedir> --targets-file <targets.tsv>`

5. Index random background terms:
`python -m general.index_random_tokens --basedir <basedir> --targets-file <targets.tsv>`

6. Get substitutes for target terms:
`python -m general.get_substitutes --basedir <basedir>`

7. Get substitutes for random terms:
`python -m general.get_substitutes --basedir <basedir> --random-targets`

8. Compute Scaled JSD:
`python -m general.compute_jsd --basedir <basedir>`

This will produce a .csv and .json file with the scaled JSD scores per target term. These will be located in a subdirectory of `<basedir>` named `tokenized_<model>`.

### Replicating GEMS experiments

To replicate the results on GEMS, one first needs to obtain the COHA dataset, as well as the GEMS target scores. The former can be obtained (for a fee) from Mark Davies. The latter were obtained from the authors of ...

The first step is to import the COHA data into the appropriate format. To do so, run:
`python -m coha.import_coha --basedir <basedir>`
where `<basedir>` is the directory contianing the raw COHA data. This will output a file named `all.jsonlist` in the `clean` subdirectory of `<basedir>`

To convert the individual rater judgements of semantic change (obtained from the original authors), run:
`python -m gems.average_gems_scores.py --gems-file <gems_file.csv>`
where `<gems_file.csv>` is the file of ratings obtained from the original GEMS authors. This will produce a file with the same name, with the additional suffix `.mean.csv`. This will be the `<targets-file>` for the pipeline above.

Then, run the above pipeline, pointing to the appropriate `<basedir>` and `all.jsonlist` as the `infile` arguement, and the `gems_file.csv.mean.csv` file as the `targets-file` argument.

Finally, to score the estimates of semantic change, run:
`python -m coha.evaluate --basedir <basedir> --targets-file <targets-file>`

### Replicating SemEval experiments

Becuase the SemEval targets are provided as lemmas, with multiple langauges, the pipeline for the experiments with these datasets is slightly different, even though the overall method is the same. In paritcular, the steps are essentially the same, except that there is an additional alignment step between the lemmatized and non-lemmatized text.

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

Scripts have also been included for various other parts the paper:

1. Getting the top replacement terms:
`python -m semeval.get_top_replacements --basedir <basedir>`

2. Doing the clustering analysis:
`python -m semeval.do_clustering --basedir <basedir>`

3. Making Figure 1 in maina pper
`python -m semeval.make_figure --basedir <basedir>`

