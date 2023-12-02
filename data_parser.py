# from __future__ import absolute_import, division, print_function

import logging
import os
import re

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from adapters import words_to_wordpieces_pair

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, words_a=None, words_b=None, label=None, attrs_mask_a=None, attrs_mask_b=None,
                 wordpieced=False,):
        self.guid = guid
        self.words_a = words_a
        self.words_b = words_b
        self.label = label if label else 0
        self.attrs_mask_a = attrs_mask_a if attrs_mask_a else None#[]
        self.attrs_mask_b = attrs_mask_b if attrs_mask_b else None#[]
        self.wordpieced = wordpieced


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_idx_token, attrs_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_idx_token = ori_idx_token
        self.attrs_mask = [] if not attrs_mask else attrs_mask


class DataParser(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir, **kwargs):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, **kwargs):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, **kwargs):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


def get_words(row, attr_name_ids=None):
    words = []
    attrs_mask = []
    for key in list(row.dropna().keys()):  # ignore empty attributes
        val = str(row[key]).lower()
        words_ = [w.group(2) for w in re.finditer(r'(\d+_)?([a-zA-Z0-9]+)', val)]
        for w in words_:
            words.append(w)
            if attr_name_ids:
                attrs_mask.append(attr_name_ids[key])
    return words, attrs_mask if attr_name_ids else words


class EMParser(DataParser):

    def __init__(self, l_prefix='left_', r_prefix='right_', key_attr='id', label_attr='label', out_delim=' ',
                 tokenizer=None):
        self.l_prefix = l_prefix
        self.r_prefix = r_prefix
        self.key_attr = key_attr
        self.label_attr = label_attr
        self.out_delim = out_delim if out_delim.strip() else ' '
        self.tokenizer = tokenizer

    def get_train_examples(self, data_dir, **kwargs) -> list[InputExample]:
        return self._create_examples(data_dir, 'train')

    def get_dev_examples(self, data_dir, **kwargs) -> list[InputExample]:
        return self._create_examples(data_dir, "valid")

    def get_test_examples(self, data_dir, **kwargs) -> list[InputExample]:
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        return [0, 1]

    def _create_examples(self, path, set_type) -> list[InputExample]:
        label_path = os.path.join(path, f'{set_type}.csv')
        data = pd.read_csv(label_path)
        return self.create_examples_from_dataframe(data)

    def create_examples_from_dataframe(self, dataframe):
        data = dataframe
        data = data.reset_index(drop=True)  # make sure indexes pair with number of rows

        l_key_name = self.l_prefix + self.key_attr
        r_key_name = self.r_prefix + self.key_attr
        l_cols, r_cols = [], []
        col_base_names = set()
        for c in data.columns:
            if c == self.label_attr:
                continue
            if c == l_key_name or c == r_key_name:
                continue
            if self.l_prefix in c:
                l_cols.append(c)
                col_base_names.add(c.replace(self.l_prefix, ''))
            elif self.r_prefix in c:
                r_cols.append(c)
                col_base_names.add(c.replace(self.r_prefix, ''))
        col_base_name_ids = {}
        for i, n in enumerate(sorted(col_base_names)):
            col_base_name_ids[self.l_prefix + n] = i
            col_base_name_ids[self.r_prefix + n] = i

        examples = []
        for idx, row in data.iterrows():
            guid = f'{idx}'
            label = row.get(self.label_attr, 0)
            l_tokens, l_attrs_mask = get_words(row[l_cols], col_base_name_ids)
            r_tokens, r_attrs_mask = get_words(row[r_cols], col_base_name_ids)
            examples.append(InputExample(guid=guid, words_a=l_tokens, words_b=r_tokens, label=label,
                                         attrs_mask_a=l_attrs_mask, attrs_mask_b=r_attrs_mask
                                         ))
        return examples


def convert_examples_to_features(examples: list[InputExample], label_list, max_seq_length,
                                 tokenizer,
                                 output_mode='classification',
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if not example.wordpieced:  # idempotent thanks to 'wordpieced' flag
            example = words_to_wordpieces_example(tokenizer, example)
        tokens_a, tokens_attrs_mask_a = example.words_a, example.attrs_mask_a

        tokens_b, tokens_attrs_mask_b = [], []
        if example.words_b:
            tokens_b, tokens_attrs_mask_b = example.words_b, example.attrs_mask_b
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            # truncate_seq_pair(tokens_attrs_mask_a, tokens_attrs_mask_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]
                # tokens_attrs_mask_a = tokens_attrs_mask_a[:(max_seq_length - special_tokens_count)]

        ori_idx_token = {}
        for tok in tokens_a:
            tok_id = tokenizer.vocab.get(tok)  # embeddings without context?
            ori_idx_token[tok_id] = tok
        for tok in tokens_b:
            tok_id = tokenizer.vocab.get(tok)
            ori_idx_token[tok_id] = tok

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        attrs_mask = tokens_attrs_mask_a + [-1]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            attrs_mask += [-1]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            attrs_mask += tokens_attrs_mask_b + [-1]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            attrs_mask = attrs_mask + [-1]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            attrs_mask = [-1] + attrs_mask
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attrs_mask = ([-1] * padding_length) + attrs_mask
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attrs_mask = attrs_mask + ([-1] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        # assert len(attrs_mask) == max_seq_length  # todo: check if it can be fixed in lime_explain
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # if output_mode == "classification":
        #     label_id = label_map[example.label]
        # elif output_mode == "regression":
        #     label_id = float(example.label)
        # else:
        #     raise KeyError(output_mode)
        label_id = label_map[example.label]

        features.append(InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_id,
            ori_idx_token=ori_idx_token,
            attrs_mask=attrs_mask,
        ))
    return features


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    'lrds': EMParser,
}

# output_modes = {
#     # "esnli": "classification",
#     # "quora": "classification",
#     # "qqp": "classification",
#     # "mrpc": "classification",
#     "lrds": "classification",
#     # 'wcl_attr': "classification",
#     # 'wcl_class': "classification",
# }

# GLUE_TASKS_NUM_LABELS = {
#     # "esnli": 3,
#     # "mrpc": 2,
#     # "quora": 2,
#     # "qqp": 2,
#     "lrds": 2,
#     # 'wcl_attr': 2,
#     # 'wcl_class': 2,
# }


def words_to_wordpieces_example(tokenizer, example: InputExample):
    if example.wordpieced:
        return example

    example.words_a, example.words_b, example.attrs_mask_a, example.attrs_mask_b = words_to_wordpieces_pair(
        tokenizer, example.words_a, example.words_b, example.attrs_mask_a, example.attrs_mask_b)
    example.wordpieced = True
    return example


def load_and_cache_examples(args, task, tokenizer, type):
    processor = processors[task]()
    # output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))

    if not os.path.exists(cached_features_file):
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        if type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        if args.wordpieced:
            for i, e in enumerate(examples):
                examples[i] = words_to_wordpieces_example(tokenizer, e)

        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
        # logger.info("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)
    else:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    # if output_mode == "classification":
    #     all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    # elif output_mode == "regression":
    #     all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_attrs_mask = torch.tensor([f.attrs_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_attrs_mask)

    return dataset
