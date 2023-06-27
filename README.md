## Substitution-based Semantic Change Detection

This repo accompanies the paper "Substitution-based Semantic Change Detection using Contextual Embeddings", published at ACL 2023.

It contains code to perform substitution-based semantic change detection for target terms on a generic corpus, as well as code to replicate the experiments in the paper.

Using this approach to semenatic change detection involves running a pipeline of scripts, which cover tokenization, continued masked langauge model pretraining, indexing terms, getting substitutes, and finally computing amount of change using Jensen-Shannon divergence. Intermediate products produced include the substitutes for each sampled mention, which can be used for additional analyses, as is done in the paper.

### Requirements

The python packages used (along with the versions used) for this repo include:
- numpy
- scipy
- pandas
- tqdm
- torch
- transformers

In addition, `networkx` was used for Louvain community detection in the clustering analysis, and `matplitlib` was used for making Figure 1


### Replicating GEMS experiments

To replicate the results on GEMS, one first needs to obtain the COHA dataset, as well as the GEMS target scores. The former can be obtained (for a fee) from Mark Davies. The latter were obtained from the authors of ...

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