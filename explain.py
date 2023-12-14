import json
import os
import pickle

import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig
from transformers import BertTokenizer

import expldf
from data_parser import load_and_cache_examples
from embeddings_cache import EMBS, BERT4SeqClf, ModelWrapper
from eval_measures import PostHocAccuracy
from groups import get_text_groups
from lime_corrclust import LimeCorrClust
from prefix_words import get_words_attrs_mask, prefix_words_to_feature
from utils import uniquify, EXEC_TIME_PROFILER, set_seed, bindump


def explain(args, model: ModelWrapper, tokenizer, dataset: TensorDataset) -> (dict, str):
    eval_output_dir = uniquify(f'{args.output_dir}/{args.model_setup}')
    os.makedirs(eval_output_dir)
    with open(f'{eval_output_dir}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    ds_sampler = SequentialSampler(dataset)
    ds_dloader = DataLoader(dataset, sampler=ds_sampler, batch_size=eval_batch_size)

    count = -1
    acc = PostHocAccuracy()
    # degrad_score = DegradationScoreF1(.10)

    wout = expldf.ExplDataFrame()
    gout = expldf.ExplDataFrame(True)

    EXEC_TIME_PROFILER.reset()

    # lime_cache = None
    lime_cache_name = f'lime_cache_{len(dataset)}_{args.lime_n_word_samples}'
    if args.lime_n_word_features > 0:
        lime_cache_name += f'_{args.lime_n_word_features}'
    lime_cache_path = f'./{args.data_dir}/{lime_cache_name}'
    if os.path.exists(lime_cache_path):
        lime_cache = pickle.load(open(lime_cache_path, 'rb'))
        lime_cached = True
    else:
        # lime_cache = [[], [], []]
        lime_cache = None
        lime_cached = False

    _lime_corrclust = LimeCorrClust(args, model, tokenizer, lime_cache)
    if lime_cache is None:
        lime_cache = [[], [], []]

    # interpret_file_ = InterpretationFile(eval_output_dir, _lime_corrclust.do_groups)

    for batch in tqdm(ds_dloader, desc='Evaluating'):
        EXEC_TIME_PROFILER.start()

        set_seed(args)
        count += 1

        batch = tuple(t.to(args.device) for t in batch)

        words, segments_ids, attrs_mask, prefix_words_a, prefix_words_b = get_words_attrs_mask(
            batch[0][0], batch[2][0], batch[4][0], tokenizer, args.wordpieced)

        _prefix_words = prefix_words_a + prefix_words_b
        prefix_words_tags = ['cls'] + prefix_words_a + ['sep'] + prefix_words_b + ['sep']

        # BERT classification
        input_ids_ = batch[0]
        attention_mask_ = batch[1]
        segments_ids_ = batch[2]

        logits = model.predict(None, input_ids_, attention_mask_, segments_ids_, args.wordpieced)
        logits = logits.detach().cpu().numpy()
        # pred = logits.argmax(axis=1)

        idxs, word_scores, groups, group_scores = _lime_corrclust.lime_corrclust(prefix_words_tags, words,
                                                                                 attrs_mask, segments_ids,
                                                                                 logits, count)

        if len(idxs) == len(_prefix_words):
            w_probs = logits
            # pass
        else:
            top_prefix_words = [prefix_words_tags[i] for i in idxs]
            f = prefix_words_to_feature(top_prefix_words, segments_ids, tokenizer, args.wordpieced,
                                        args.max_seq_length)
            input_ids_ = torch.tensor([f.input_ids], device=args.device)
            attention_mask_ = torch.tensor([f.input_mask], device=args.device)
            segment_ids_ = torch.tensor([f.segment_ids], device=args.device)
            logits_ = model.predict(None, input_ids_, attention_mask_, segment_ids_, args.wordpieced)
            w_probs = logits_.detach().cpu().numpy()
            # pass

        if not lime_cached:
            lime_cache[0].append([prefix_words_tags[i] for i in idxs])
            lime_cache[1].append(word_scores.tolist())
            lime_cache[2].append(w_probs.tolist())

        # words_rank = [0.] * len(prefix_words_tags)
        # for i, r in enumerate(idxs):
        #     words_rank[r] = float(i)
        # words_rank = np.array(words_rank)
        # ifd_group_scores, ifd_group_scores_desc_ix = infer_group_scores(groups, np.array(words_rank))
        # ifd_group_scores_desc_ix = ifd_group_scores_desc_ix[::-1]

        # expl_pred = np.argmax(w_probs)
        # degrad_predict_cls = CReWDegradPredictGroups if _lime_corrclust.do_groups else CReWDegradPredict
        # _ixs = groups if _lime_corrclust.do_groups else idxs
        # if expl_pred == 0:
        #     _ixs = np.flip(_ixs)

        # acc.append(pred[0], expl_pred)

        # if args.wordpieced:
        #     degrad_predict = degrad_predict_cls(model, args.device, _ixs, batch[0][0], batch[1][0], batch[2][0])
        #
        # else:
        #     degrad_predict = degrad_predict_cls(model, args.device, _ixs, words, None, segments_ids, False, tokenizer)

        # degrad_score.append(degrad_predict)#, pred[0])

        # _, _, attrs_mask_a, attrs_mask_b = words_seq_to_pair(segments_ids, words, attrs_mask)
        # interpret_file_.new_row() \
        #     .set_id(count).set_l_instance(prefix_words_a).set_r_instance(prefix_words_b) \
        #     .set_l_attrs_mask(attrs_mask_a).set_r_attrs_mask(attrs_mask_b) \
        #     .set_model_pred(int(pred)) \
        #     .set_words_desc_impact(idxs).set_words_scorer_pred(w_expl_pred) \
        #     .set_groups(groups).set_group_scorer_pred(g_expl_pred)
        # if ifd_group_scores_desc_ix is not None:
        #     interpret_file_.set_groups_argsort_desc_inferred_impact(ifd_group_scores_desc_ix)
        # interpret_file_.flush_row()

        wout.add_rows(count,
                      impact=word_scores,
                      word=[words[i] for i in idxs],
                      wid_word=[prefix_words_tags[i] for i in idxs],
                      column=[int(attrs_mask[i]) for i in idxs],
                      segment=[int(segments_ids[i]) for i in idxs],
                      wid=idxs
                      )

        gout.add_rows(count,
                      group=get_text_groups({i: g for i, g in enumerate(groups)}, prefix_words_tags),
                      impact=group_scores,
                      wids=groups,
                      )

        # EXEC_TIME_PROFILER.timestep('degrad_score')
        EXEC_TIME_PROFILER.timestep('stuff')

    # interpret_file_.dump_to_file()
    bindump(wout.get_df(), f'{eval_output_dir}/wexpls')
    bindump(gout.get_df(), f'{eval_output_dir}/gexpls')

    ret = {
        'Post-hoc accuracy': acc.get_score(),
        # 'Degradation score': degrad_score.get_score(),
    }

    # bindump({
    #     'degrad_steps': degrad_score.get_degrad_steps(),
    #     'lerf_f1': degrad_score.get_lerf_f1(),
    #     'morf_f1': degrad_score.get_morf_f1(),
    # }, f'{eval_output_dir}/degrad_score.pkl')
    bindump(ret, f'{eval_output_dir}/return.pkl')
    bindump((str(os.uname()), EXEC_TIME_PROFILER.get_list()), f'{eval_output_dir}/exec_time_profile.pkl')
    # bindump(_lime_corrclust.lime_wexpl, f'{eval_output_dir}/lime_wexpl.pkl')
    # bindump(_lime_corrclust.lime_gexpl, f'{eval_output_dir}/lime_gexpl.pkl')

    if not lime_cached:
        bindump(lime_cache, lime_cache_path)

    return ret, eval_output_dir


def load_model_tokenizer(model_type, pretrained_path, device, do_lower_case=False):
    config_class, model_class, tokenizer_class = BertConfig, AutoModelForSequenceClassification, BertTokenizer
    model = model_class.from_pretrained(pretrained_path)
    tokenizer = tokenizer_class.from_pretrained(pretrained_path, do_lower_case=do_lower_case)
    model.to(device)
    return model, tokenizer


def main(args):
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        torch.cuda.empty_cache()

    set_seed(args)

    # Load a trained model and vocabulary that you have fine-tuned
    model, tokenizer = load_model_tokenizer(args.model_type, args.model_dir, args.device, args.do_lower_case)
    model.eval()

    wmodel = BERT4SeqClf(model, tokenizer, args.device, model.bert)
    EMBS.work_with(wmodel)

    # Test
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'test')

    # lime_cache = None
    # lime_cache_name = f'lime_cache_{len(test_dataset)}_{args.lime_n_word_samples}'
    # if args.lime_n_word_features > 0:
    #     lime_cache_name += f'_{args.lime_n_word_features}'
    # lime_cache_path = f'./{args.data_dir}/{lime_cache_name}'
    # if os.path.exists(lime_cache_path):
    #     lime_cache = pickle.load(open(lime_cache_path, 'rb'))

    # do_groups = (args.group_scorer and args.group_scorer == 'corrclust')
    # _lime_corrclust = LimeCorrClust(args, wmodel, tokenizer, None, do_groups)

    eval_ms, _ = explain(args, wmodel, tokenizer, test_dataset,
                         # _lime_corrclust
                         )
    print('All done!')

    # if lime_cache is None:
    #     bindump(_lime_corrclust.lime_wexpl, lime_cache_path)

    for k, v in eval_ms.items():
        print(f'{k}: {v}')
