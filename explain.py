import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import BertConfig
from transformers import BertTokenizer

from adapters import words_seq_to_pair
from binarydump import bindump
from data_parser import load_and_cache_examples
from embeddings_cache import EMBS, BERT4SeqClf, ModelWrapper
from eval_measures import PostHocAccuracy, CReWDegradPredict, DegradationScoreF1, AOPC, \
    CReWDegradPredictGroups
from groups import infer_group_scores
from interpretation_file import InterpretationFile
from lime_corrclust import LimeCorrClust
from prefix_words import get_words_attrs_mask
from utils import uniquify, EXEC_TIME_PROFILER, set_seed


# from sklearn.metrics import silhouette_score


def cc_scores_distance(A, B, cc_scores_dict, range_min, range_max):
    a, b = int(A[0]), int(B[0])
    if a == b:
        p, n = range_max, 0
    elif a < b:
        p, n = cc_scores_dict[f'{a}_{b}']
    else:
        p, n = cc_scores_dict[f'{b}_{a}']
    ret = (- n - p - range_min) / (range_max - range_min)
    return ret


def explain(args, model: ModelWrapper, tokenizer, dataset: TensorDataset, _lime_corrclust=None) -> (dict, str):
    eval_output_dir = uniquify(f'{args.output_dir}/{args.model_setup}')
    os.makedirs(eval_output_dir)
    with open(f'{eval_output_dir}/args.json', 'w') as f:
        json.dump(
            args.__dict__,
            # args._asdict(),
            f, indent=2)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    ds_sampler = SequentialSampler(dataset)
    ds_dloader = DataLoader(dataset, sampler=ds_sampler,
                            # batch_size=args.eval_batch_size
                            batch_size=eval_batch_size
                            )

    count = 0
    acc = PostHocAccuracy()
    aopc = AOPC(top_u=10)
    # slh_score = []
    # if group_scorer == 'corrclust':
    if _lime_corrclust.do_groups:
        degrad_score = DegradationScoreF1(.10)
    else:
        degrad_score = DegradationScoreF1(.10)

    interpret_file_ = InterpretationFile(eval_output_dir,
                                         # True if args.group_scorer else False
                                         _lime_corrclust.do_groups
                                         )

    # model.eval()
    iterator = ds_dloader

    # lime_cache = None
    # lime_cache_name = f'lime_cache_{len(dataset)}_{args.num_words_samples}'
    # if args.num_features > 0:
    #     lime_cache_name += f'_{args.num_features}'
    # lime_cache_path = f'./{args.data_dir}/{lime_cache_name}'
    # if os.path.exists(lime_cache_path):
    #     lime_cache = pickle.load(open(lime_cache_path, 'rb'))

    # _lime_corrclust = LimeCorrClust(args, model, tokenizer, lime_cache, (group_scorer and group_scorer == 'corrclust'))
    # _wmask_gmask = WMaskGMask(args, model, (group_scorer and group_scorer == 'gmask'))

    EXEC_TIME_PROFILER.reset()

    for x in tqdm(iterator, desc='Evaluating'):
        EXEC_TIME_PROFILER.start()

        set_seed(args)
        count += 1

        batch = x
        batch = tuple(t.to(args.device) for t in batch)

        words, segments_ids, attrs_mask, prefix_words_a, prefix_words_b = get_words_attrs_mask(
            batch[0][0], batch[2][0], batch[4][0], tokenizer, args.wordpieced)

        # input_words = prefix_words_a + prefix_words_b
        # interpret_file_.new_row().set_id(count).set_l_instance(prefix_words_a).set_r_instance(prefix_words_b)

        _prefix_words = prefix_words_a + prefix_words_b
        prefix_words_tags = ['cls'] + prefix_words_a + ['sep'] + prefix_words_b + ['sep']

        # BERT classification
        input_ids_ = batch[0]
        attention_mask_ = batch[1]
        segments_ids_ = batch[2]

        logits = model.predict(None, input_ids_, attention_mask_, segments_ids_, args.wordpieced)
        _, pred = logits.max(dim=1)

        # EXEC_TIME_PROFILER.timestep('model_pred')

        always_drop_idxs = []

        # lime_words_explainer, lime_group_explainer = None, None

        # if words_scorer == 'wmask' and args.wordpieced:
        #     w_expl_pred, idxs, g_expl_pred, groups = _wmask_gmask.wmask_gmask(
        #         _prefix_words, input_ids_, attention_mask_, segments_ids_, pred)
        #
        # elif words_scorer == 'lime':
        w_expl_pred, idxs, g_expl_pred, groups = _lime_corrclust.lime_corrclust(prefix_words_tags, words,
                                                                                attrs_mask, segments_ids,
                                                                                logits.detach().cpu().numpy(),
                                                                                count)
        words_rank = [0.] * len(prefix_words_tags)
        for i, r in enumerate(idxs):
            words_rank[r] = float(i)
        words_rank = np.array(words_rank)  # [np.array([w not in ['[CLS]', '[SEP]'] for w in words])]
        ifd_group_scores, ifd_group_scores_desc_ix = infer_group_scores(groups, np.array(words_rank))
        ifd_group_scores_desc_ix = ifd_group_scores_desc_ix[::-1]

        # else: raise Exception()

        # expl_pred = g_expl_pred if group_scorer else w_expl_pred
        # degrad_predict_cls = CReWDegradPredictGroups if group_scorer == 'corrclust' else CReWDegradPredict
        # _ixs = groups if group_scorer == 'corrclust' else idxs
        expl_pred = g_expl_pred if _lime_corrclust.do_groups else w_expl_pred
        degrad_predict_cls = CReWDegradPredictGroups if _lime_corrclust.do_groups else CReWDegradPredict
        _ixs = groups if _lime_corrclust.do_groups else idxs
        if expl_pred == 0:
            _ixs = np.flip(_ixs)

        acc.append(pred[0], expl_pred)

        if args.wordpieced:
            # # if group_scorer == 'corrclust':
            # if _lime_corrclust.do_groups:
            #     if expl_pred == 0:
            #         _ixs = np.flip(_ixs)
            # else:
            #     # if words_scorer == 'lime' and expl_pred == 0:
            #     if expl_pred == 0:
            #         _ixs = np.flip(_ixs)
            degrad_predict = degrad_predict_cls(model, args.device, _ixs, batch[0][0], batch[1][0], batch[2][0],
                                                always_drop_idxs=always_drop_idxs
                                                )

        else:
            # # if group_scorer == 'corrclust':
            # if _lime_corrclust.do_groups:
            #     if expl_pred == 0:
            #         _ixs = np.flip(_ixs)
            # else:
            #     # if words_scorer == 'lime' and expl_pred == 0:
            #     if expl_pred == 0:
            #         _ixs = np.flip(_ixs)
            degrad_predict = degrad_predict_cls(model, args.device, _ixs, words, None, segments_ids, False, tokenizer,
                                                always_drop_idxs=always_drop_idxs
                                                )

        # aopc.append(degrad_predict, expl_pred, logits[0])
        degrad_score.append(degrad_predict, pred[0])
        # EXEC_TIME_PROFILER.timestep('degrad_msr')

        _, _, attrs_mask_a, attrs_mask_b = words_seq_to_pair(words, attrs_mask)
        interpret_file_.new_row() \
            .set_id(count).set_l_instance(prefix_words_a).set_r_instance(prefix_words_b) \
            .set_l_attrs_mask(attrs_mask_a).set_r_attrs_mask(attrs_mask_b) \
            .set_model_pred(int(pred)) \
            .set_words_desc_impact(idxs).set_words_scorer_pred(w_expl_pred) \
            .set_groups(groups).set_group_scorer_pred(g_expl_pred)
        # interpret_file_.set_groups_entropy(groups_entropy)
        # interpret_file_.set_groups_intra_emb_sim(groups_intra_emb_sim)
        # interpret_file_.set_groups_intra_impacts_stdev(groups_intra_impacts_stdev)
        if ifd_group_scores_desc_ix is not None:
            interpret_file_.set_groups_argsort_desc_inferred_impact(ifd_group_scores_desc_ix)
        interpret_file_.flush_row()

        EXEC_TIME_PROFILER.timestep('degrad_score')

    interpret_file_.dump_to_file()

    ret = {
        'Post-hoc accuracy': acc.get_score(),
        # 'AOPC': aopc.get_score(),
        'Degradation score': degrad_score.get_score(),
        # 'Silhouette score': -2 if len(slh_score) == 0 else np.array(slh_score).mean()
    }

    # bindump({
    #     'degrad_steps': aopc.get_degrad_steps(),
    #     'y_prob_avg': aopc.get_y_probs_avg(),
    #     'y_degrad_probs_avgs': aopc.get_y_degrad_probs_avgs()
    # }, f'{eval_output_dir}/aopc.pkl')
    bindump({
        'degrad_steps': degrad_score.get_degrad_steps(),
        'lerf_f1': degrad_score.get_lerf_f1(),
        'morf_f1': degrad_score.get_morf_f1(),
    }, f'{eval_output_dir}/degrad_score.pkl')
    bindump(ret, f'{eval_output_dir}/return.pkl')
    bindump((str(os.uname()), EXEC_TIME_PROFILER.get_list()), f'{eval_output_dir}/exec_time_profile.pkl')
    # if words_scorer == 'lime':
    bindump(_lime_corrclust.lime_wexpl, f'{eval_output_dir}/lime_wexpl.pkl')
    # elif words_scorer == 'wmask':
    #     bindump(_wmask_gmask.wmask_expls, f'{eval_output_dir}/wmask_expls.pkl')
    # if group_scorer == 'corrclust':
    bindump(_lime_corrclust.lime_gexpl, f'{eval_output_dir}/lime_gexpl.pkl')
    # elif group_scorer == 'gmask':
    #     bindump(_wmask_gmask.gmask_expls, f'{eval_output_dir}/gmask_expls.pkl')

    return ret, eval_output_dir


def load_model_tokenizer(model_type, pretrained_path, device, do_lower_case=False):
    config_class, model_class, tokenizer_class = BertConfig, AutoModelForSequenceClassification, BertTokenizer
    model = model_class.from_pretrained(pretrained_path)
    tokenizer = tokenizer_class.from_pretrained(pretrained_path, do_lower_case=do_lower_case)
    model.to(device)
    return model, tokenizer


def main(args):

    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' #args.gpu_id
        torch.cuda.empty_cache()

    # Set seed
    set_seed(args)

    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Load a trained model and vocabulary that you have fine-tuned
    model, tokenizer = load_model_tokenizer(args.model_type, args.model_dir, args.device, args.do_lower_case)
    model.eval()

    # EMBS.work_with(model.bert, tokenizer, args.device)
    wmodel = BERT4SeqClf(model, tokenizer, args.device, model.bert)
    EMBS.work_with(wmodel)

    # Test
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, 'test')

    # lime_cache = None
    # lime_cache_name = f'lime_cache_{len(test_dataset)}_{args.num_words_samples}'
    # if args.num_features > 0:
    #     lime_cache_name += f'_{args.num_features}'
    # lime_cache_path = f'./{args.data_dir}/{lime_cache_name}'
    # if os.path.exists(lime_cache_path):
    #     lime_cache = pickle.load(open(lime_cache_path, 'rb'))
    lime_cache = None
    lime_cache_name = f'lime_cache_{len(test_dataset)}_{args.lime_n_word_samples}'
    if args.lime_n_word_features > 0:
        lime_cache_name += f'_{args.lime_n_word_features}'
    lime_cache_path = f'./{args.data_dir}/{lime_cache_name}'
    if os.path.exists(lime_cache_path):
        lime_cache = pickle.load(open(lime_cache_path, 'rb'))

    do_groups = (args.group_scorer and args.group_scorer == 'corrclust')
    _lime_corrclust = LimeCorrClust(args, wmodel, tokenizer, lime_cache, do_groups)

    eval_ms, _ = explain(args, wmodel, tokenizer, test_dataset, _lime_corrclust)
    print('All done!')

    if lime_cache is None:
        bindump(_lime_corrclust.lime_wexpl, lime_cache_path)

    for k, v in eval_ms.items():
        print(f'{k}: {v}')
