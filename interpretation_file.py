import pandas as pd

from binarydump import bindump


class InterpretationFile:

    def __init__(self, parent_path: str, has_group_scorer=False):
        # self.path = f'{parent_path}/interpret.csv'
        self.path = f'{parent_path}/interpret.pkl'
        self.has_group_scorer = has_group_scorer
        self.data = []
        self.row = self._empty_row()
        self.flushed = True

    def _empty_row(self):
        row = {
            'id': -1,
            'l_instance': [],
            'r_instance': [],
            'l_attrs_mask': [],
            'r_attrs_mask': [],
            'model_pred': -1,
            'words_desc_impact': [],
            'words_scorer_pred': -1,
        }
        if self.has_group_scorer:
            row.update({
                'groups': [],
                # 'groups_argsort_desc_impact': [],
                'groups_argsort_desc_inferred_impact': [],
                'group_scorer_pred': -1,
                # 'groups_entropy': [],
                # 'groups_intra_emb_sim': [],
                # 'groups_intra_impacts_stdev': [],
            })
        return row

    def new_row(self):
        self.flush_row()
        self.row = self._empty_row()
        return self

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if self.row.__contains__(k):
                self.row[k] = v
                self.flushed = False
        return self

    def set_id(self, id):
        return self.set(id=id)

    def set_l_instance(self, l_instance):
        return self.set(l_instance=l_instance)

    def set_r_instance(self, r_instance):
        return self.set(r_instance=r_instance)

    def set_l_attrs_mask(self, l_attrs_mask):
        return self.set(l_attrs_mask=l_attrs_mask)

    def set_r_attrs_mask(self, r_attrs_mask):
        return self.set(r_attrs_mask=r_attrs_mask)

    def set_model_pred(self, model_pred):
        return self.set(model_pred=model_pred)

    def set_words_desc_impact(self, words_desc_impact):
        return self.set(words_desc_impact=words_desc_impact)

    def set_words_scorer_pred(self, words_scorer_pred):
        return self.set(words_scorer_pred=words_scorer_pred)

    # def set_groups_desc_impact(self, x):
    #     return self.set(groups_desc_impact=x)

    def set_groups(self, groups):
        return self.set(groups=groups)

    # def set_groups_argsort_desc_impact(self, groups_argsort_desc_impact):
    #     return self.set(groups_argsort_desc_impact=groups_argsort_desc_impact)

    def set_group_scorer_pred(self, group_scorer_pred):
        return self.set(group_scorer_pred=group_scorer_pred)

    # def set_groups_entropy(self, x):
    #     return self.set(groups_entropy=x)
    #
    # def set_groups_intra_emb_sim(self, x):
    #     return self.set(groups_intra_emb_sim=x)
    #
    # def set_groups_intra_impacts_stdev(self, x):
    #     return self.set(groups_intra_impacts_stdev=x)
    #
    def set_groups_argsort_desc_inferred_impact(self, groups_argsort_desc_inferred_impact):
        return self.set(groups_argsort_desc_inferred_impact=groups_argsort_desc_inferred_impact)

    def flush_row(self):  # idempotent
        if not self.flushed:
            self.data.append(self.row)
            self.flushed = True
        return self

    def dump_to_file(self):
        bindump(pd.DataFrame.from_records(self.data).set_index('id'), self.path)
        return self
