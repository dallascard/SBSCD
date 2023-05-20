Once for each language:
- import_semeval.py: convert raw data to cleaned format, and index targets in lemmas

For a given language (e.g., eng) and model (e.g., bert-large-uncased):
- tokenize_semeval.py: tokenize the cleaned data with the model
- do_alignment.py: align the lemams to the tokenized data
- export_for_pretraining.py: export data for continued langauge model pretraining
- run_mlm.py: do continued language model pretraining (to adapt to corpus)
- index_targets.py: index targets in tokenized data
- index_random_targets.py: index random background terms
- get_substitutes.py: get the top-k subs for each target (run for both targets and random)
- evaluate_semeval.py: do the final evaluation

Analysis and plotting:
- get_top_replacements.py 
- do_clustering.py
