from __future__ import division, absolute_import, print_function

import re

import numpy as np
import torch

from adapters import words_seq_to_pair, wordpieces_to_words, words_pair_to_seq
from data_parser import InputExample, convert_examples_to_features


def indexes_to_prefix_words(idxs, words):
    prefix_words = []
    for ix in idxs:
        w = words[ix]
        prefix_words.append(f'{ix}_{w}')
    return prefix_words


def get_idx(prefix_word: str) -> int:
    # return int(re.match('\d_\d+_(\d+)_', prefix_word).group(1))
    return int(re.match('(\d+)_', prefix_word).group(1))


def get_word(prefix_word: str) -> str:
    # return re.sub('\d_\d+_\d+_', '', prefix_word)
    return re.sub('\d+_', '', prefix_word)


def get_words_attrs_mask(
        input_ids: torch.Tensor, segment_ids: torch.Tensor, attrs_mask: torch.Tensor, tokenizer, wordpieced: bool,
) -> (list, list, list, list, list, list):
    words = []
    for id in input_ids:
        words.append(tokenizer.ids_to_tokens[int(id)])

    words_a, words_b, attrs_mask_a, attrs_mask_b = words_seq_to_pair(segment_ids, words, attrs_mask)
    if not wordpieced:
        attrs_mask_dev = attrs_mask.device
        seg_ids_device = segment_ids.device
        words_a, words_b, attrs_mask_a, attrs_mask_b = wordpieces_to_words(
            words_a, words_b, attrs_mask_a, attrs_mask_b)
        words, attrs_mask, segment_ids = words_pair_to_seq(words_a, words_b, attrs_mask_a, attrs_mask_b)
        attrs_mask = torch.tensor(attrs_mask, device=attrs_mask_dev)
        segment_ids = torch.tensor(segment_ids, device=seg_ids_device)

    words = np.array(words)

    # prefix_words_a = indexes_to_prefix_words(range(1, 1 + len(words_a)), words)
    # prefix_words_b = indexes_to_prefix_words(range(1 + len(words_a) + 1, 1 + len(words_a) + 1 + len(words_b)), words)
    prefix_words_a = indexes_to_prefix_words(range(0, len(words_a)), words)
    prefix_words_b = indexes_to_prefix_words(range(len(words_a), len(words_a) + len(words_b)), words)

    return words, segment_ids, attrs_mask, prefix_words_a, prefix_words_b


def prefix_words_to_feature(prefix_words, segment_ids, tokenizer, wordpieced, max_sequence_length):
    words_a_dict, words_b_dict = dict(), dict()
    prefix_words_ = np.array(prefix_words)
    for pw in prefix_words_:
        ix = get_idx(pw)
        # (words_b_dict if get_segment_id(pw) else words_a_dict)[ix] = get_word(pw)
        (words_b_dict if segment_ids[ix] else words_a_dict)[ix] = get_word(pw)
    words_a = [words_a_dict[k] for k in sorted(words_a_dict)]
    words_b = [words_b_dict[k] for k in sorted(words_b_dict)]

    example = InputExample('#0', words_a=words_a, words_b=words_b, wordpieced=wordpieced)
    features = convert_examples_to_features([example], [0, 1], max_sequence_length, tokenizer, 'classification')
    f = features[0]
    return f
