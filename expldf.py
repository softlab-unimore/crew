import pandas as pd


class ExplDataFrame:

    def __init__(self, do_groups=False):
        self.do_groups = do_groups
        if self.do_groups:
            self.cols = [
                'data_inx',
                'wids',
                'group',
                'impact'
            ]
        else:
            self.cols = [
                'data_inx',
                'wid',
                'word',
                'wid_word',
                'column',
                'segment',
                'impact'
            ]

        self.data = []
        self.row = self._empty_row()
        self.flushed = True

    def _empty_row(self):
        return {c: None for c in self.cols}

    def new_row(self):
        self.flush_row()
        self.row = self._empty_row()
        return self

    def flush_row(self):  # idempotent
        if not self.flushed:
            self.data.append(self.row)
            self.flushed = True
        return self

    def add_rows(self, data_inx: int, *cols, **kwcols):
        if len(cols) > 0:  # ignore kwcols
            for k, col in zip(self.cols[1:], cols):
                kwcols[k] = col

        rows = []
        for row in zip(*[kwcols[k] for k in self.cols[1:]]):
            r = self._empty_row()
            r['data_inx'] = data_inx
            for k, v in zip(self.cols[1:], row):
                r[k] = v
            rows.append(r)
        self.data.extend(sorted(rows, key=lambda x: abs(x['impact']), reverse=True))

    def get_df(self):
        return pd.DataFrame.from_records(self.data)
