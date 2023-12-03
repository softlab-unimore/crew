# CREW: Clusters of RElated Words to explain Entity Matching

CREW is a cluster-based explainer for Entity Matching, that generates faithful and easily interpretable explanations to matching decisions. 
Given a pair of entity descriptions, classified as "matching" or "not matching" by any model, CREW:
- Clusters the words according to the relatedness heuristics detailed in the paper;
- Measures the importance of each cluster on the model behavior with a local, post-hoc explainer.

An explanation consists of each word cluster pairs with its importance coefficient. 

For a detailed description of the work please read [our paper](https://example.com/). Please cite the paper if you use the code from this repository in your work.

```
@article{}
```

## Usage

```python
crew.py [-h] [--semantic {True,False}] [--importance {True,False}] [--schema {True,False}] 
        [--wordpieced {True,False}] [--max_seq_length MAX] 
        [--lime_n_word_samples WS] [--lime_n_group_samples GS] 
        [--lime_n_word_features WF] 
        [--gpu {True,False}] [-seed SEED]
        data_dir model_dir output_dir
```

Positional arguments:
-  `data_dir` : The directory with the input data.
-  `model_dir` : The directory with the model
-  `output_dir` : The directory in which the explanations and other additional outputs will be saved

Optional arguments:
-  `--semantic` : Include semantic relatedness
-  `--importance` : Include importance relatedness
-  `--schema` : Include schema relatedness
-  `--wordpieced` : if `True`, tokenize the output. If `False`, preserve original words
-  `--max_seq_length` : The maximum total length of the tokenized input sequence. Sequences longer than this will be truncated, and sequences shorter will be padded.
-  `--lime_n_word_samples` : number of perturbed samples to learn a word-level explanation [LIME]
-  `--lime_n_group_samples` : number of perturbed samples to learn a group-level explanation [LIME]
-  `--lime_n_word_features` : maximum number of features present in explanation [LIME]
-  `--gpu` : if `True`, run on GPU. If `False`, run on CPU
-  `-seed`, `--seed`
