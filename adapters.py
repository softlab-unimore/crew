def validate_words_attrs_mask(words_a, words_b, attrs_mask_a, attrs_mask_b):
    if attrs_mask_a is not None and len(words_a) != len(attrs_mask_a):
        raise Exception(f'Mismatching lengths: words_a={len(words_a)}, attra_mask_a={len(attrs_mask_a)})')

    if words_b:
        if attrs_mask_b is not None and len(words_b) != len(attrs_mask_b):
            raise Exception(f'Mismatching lengths: words_b={len(words_b)}, attra_mask_b={len(attrs_mask_b)})')


def _words_to_wordpieces_adapt(tokenizer, tokens: list[str], attrs_mask: list):
    if not attrs_mask:
        attrs_mask = [-1] * len(tokens)
    tokens_ = []
    attrs_mask_ = []
    for w, a in zip(tokens, attrs_mask):
        for t in tokenizer.tokenize(w):
            tokens_.append(t)
            attrs_mask_.append(a)
    return tokens_, attrs_mask_


def words_to_wordpieces_pair(tokenizer, words_a, words_b, attrs_mask_a, attrs_mask_b):
    validate_words_attrs_mask(words_a, words_b, attrs_mask_a, attrs_mask_b)

    words_a_, attrs_mask_a_ = _words_to_wordpieces_adapt(tokenizer, words_a, attrs_mask_a)
    words_b_, attrs_mask_b_ = _words_to_wordpieces_adapt(tokenizer, words_b, attrs_mask_b)
    if attrs_mask_a is None and attrs_mask_b is None:
        attrs_mask_a_ = []
        attrs_mask_b_ = []


    return words_a_, words_b_, attrs_mask_a_, attrs_mask_b_


def words_to_wordpieces(tokenizer, words, attrs_mask=None):
    words_a, words_b, attrs_mask_a, attrs_mask_b = words_seq_to_pair(words, attrs_mask)

    wordpieces_a, wordpieces_b, attrs_mask_a_, attrs_mask_b_ = words_to_wordpieces_pair(
        tokenizer, words_a, words_b, attrs_mask_a, attrs_mask_b)

    words, attrs_mask, segment_ids = words_pair_to_seq(wordpieces_a, wordpieces_b, attrs_mask_a_, attrs_mask_b_)
    return words, attrs_mask, segment_ids


def _wordpieces_to_words_adapt(wps: list[str], attrs_mask: list[int]):
    words_ = []
    attrs_mask_ = []
    w = wps[0]
    a = attrs_mask[0]
    for i in range(1, len(wps)):
        if wps[i].startswith('##'):
            # append to current word
            w += wps[i].replace('##', '')
        else:
            words_.append(w)
            attrs_mask_.append(a)
            # initialize new word
            w = wps[i]
            a = attrs_mask[i]
    words_.append(w)
    attrs_mask_.append(a)
    return words_, attrs_mask_


def wordpieces_to_words(wps_a: list[str], wps_b: list[str], attrs_mask_a, attrs_mask_b) -> (list[str], list[str], list[int], list[int]):
    validate_words_attrs_mask(wps_a, wps_b, attrs_mask_a, attrs_mask_b)

    words_a_, attrs_mask_a_ = _wordpieces_to_words_adapt(wps_a, attrs_mask_a)
    words_b_, attrs_mask_b_ = _wordpieces_to_words_adapt(wps_b, attrs_mask_b)
    return words_a_, words_b_, attrs_mask_a_, attrs_mask_b_


def words_pair_to_seq(words_a, words_b, attrs_mask_a=None, attrs_mask_b=None):
    if not attrs_mask_a:
        attrs_mask_a = [-1] * len(words_a)
    if not attrs_mask_b:
        attrs_mask_b = [-1] * len(words_b)

    words = ['[CLS]'] + words_a + ['[SEP]'] + words_b + ['[SEP]']
    attrs_mask = [-1] + attrs_mask_a + [-1] + attrs_mask_b + [-1]
    segment_ids = [0] + [0] * len(words_a) + [0] + [1] * len(words_b) + [1]
    return words, attrs_mask, segment_ids


def words_seq_to_pair(words, attrs_mask=None):
    if attrs_mask is None or len(attrs_mask) == 0:
        attrs_mask = [-1] * len(words)

    words_a = []
    words_b = []
    attrs_mask_a = []
    attrs_mask_b = []

    seps = 0
    for w, a in zip(words, attrs_mask):
        if w == '[CLS]':
            continue
        elif w == '[SEP]':
            seps += 1
            if seps == 2:
                break
            else:
                continue
        else:
            (words_b if seps == 1 else words_a).append(w)
            (attrs_mask_b if seps == 1 else attrs_mask_a).append(int(a))

    return words_a, words_b, attrs_mask_a, attrs_mask_b
