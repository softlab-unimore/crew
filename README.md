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
-  `model_dir` : The directory with the model.
-  `expls_dir` : The directory in which the explanations and other additional outputs will be saved.

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


## Degradation score

To evaluate the degradation score on a particular experiment, run:

```python
degra.py [-h] [--degra_step STEP] [--gpu {True,False}] exp_path model_path
```

Positional arguments:
-  `expls_dir` : The directory with the explanation and additional outputs.
-  `model_dir` : The directory with the model.

Optional arguments:
-  `--degra_step` : The percentage step between two degradation levels. 100% divided by this step should be an integer number, that is the number of degradation levels. By default is `0.10` (10%).
-  `--gpu` : if `True`, run on GPU. If `False`, run on CPU. By default is `True`.

This command creates a binary file, `degrad_score`, in the explanation path. This file contains a dictionary with the degradation score alongside the information to plot the LERF and MORF curves. To do so, run the following cell in a notebook:

```python
with open(f'{exp_path}/degrad_score', 'rb') as f:
    degradict = pickle.load(f)
steps   = degradict['degrad_steps']
lerf_f1 = degradict['lerf_f1']
morf_f1 = degradict['morf_f1']
steps   = [0.] + steps
lerf_f1 = [1.] + lerf_f1
morf_f1 = [1.] + morf_f1
plt.plot(steps, lerf_f1, label='lerf', marker='o')
plt.plot(steps, morf_f1, label='morf', marker='o')
plt.show()
```

## Render explanations

To render an explanation in a .html file, run: 

```python
visualize.py [-h] expls_dir data_inx visual_dir
```

Positional arguments:
-  `expls_dir` : The directory with the explanations and additional outputs.
-  `data_inx` : The indexes of the pairs of entity descriptions for which to render the explanation. If -1, render all the explanations.
-  `visual_dir` : The directory where to write the .html that renders the explanation.

Inside `visual_dir`, the command creates two folders, `wexpls` and `gexpls`, containing the word-level and group-level renderer explanations respectively.
Each rendered explanation is a .html file, such as `expl11.html`, with `11` being the index of the EM record in the original dataset.

| Word-level explanation                                                                                  | Group-level explanation                                                                                 |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ![wexpl](https://github.com/softlab-unimore/crew/assets/100861187/88ae0b53-7057-4d77-a9f2-181e36b04027) | ![gexpl](https://github.com/softlab-unimore/crew/assets/100861187/92647b7c-917d-481f-8b7c-61009a8bab84) |
