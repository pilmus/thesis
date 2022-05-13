import os.path
from itertools import chain

import pandas as pd
from tqdm import tqdm

from app.utils.src import utils


class IOHandler:
    """Interface between the provided training data and other modules. """

    def __init__(self,
                 fsequence,
                 fquery):
        """

        :param fsequence: training query sequence (e.g. training-sequence.tsv)
        :param fquery: training queries (e.g. fair-TREC-training-sample.json)
        :param fgroup: grouping file mapping a certain entity to a group. This can be author-to-group as in
        fair-TREC-sample-author-groups.csv or docid-to-group as in TREC-Fair-Ranking-eval-sample-groups.csv (
        generated with eval_sample_annotated.py)
        """

        self.fsequence = fsequence
        self.fquery = fquery
        queries = utils.read_jsonlines(fquery, handler=self.__unnest_query)
        queries = list(chain.from_iterable(queries))
        self.seq = self.__read_sequence(fsequence)
        self.queries = pd.DataFrame(queries)

        self.seq.qid = self.seq.qid.astype('str')
        self.queries.qid = self.queries.qid.astype('str')

    def __read_sequence(self, fsequence):
        df = pd.read_csv(fsequence, names=["sid", "q_num", "qid"], sep='^|\.|,', engine='python')
        if df.sid.isnull().all():
            df['sid'] = 0
        return df.reset_index(drop=True)

    def get_queries(self):
        return self.queries.drop_duplicates()

    def get_query_seq(self):
        seq = pd.merge(self.seq, self.queries, on="qid", how='left')
        return seq.drop_duplicates()  # dups can only happen with malformed query sequence file?

    def __unnest_query(self, query):
        ret = []
        for rank, doc in enumerate(query.get("documents"), start=1):
            ret.append({
                "doc_id": doc.get("doc_id"),
                "rank": rank,
                "relevance": doc.get("relevance"),
                "frequency": query.get("frequency"),
                "qid": query.get("qid"),
                "query": query.get("query")
            })
        return ret

    def __str__(self):
        return f"ioh_{os.path.splitext(os.path.basename(self.fsequence))[0]}_{os.path.splitext(os.path.basename(self.fquery))[0]}"

class IOHandlerKR(IOHandler):
    def __init__(self, fsequence,
                 fquery):
        super(IOHandlerKR, self).__init__(fsequence, fquery)
        self.seq = self.__read_sequence(fsequence)

    def __read_sequence(self, fsequence):
        df = pd.read_csv(fsequence, names=["sid", "q_num", "qid"], sep='^|\.|,', engine='python')
        if df.sid.isnull().all():
            df = pd.concat([df] * 150).sort_values(by='q_num')
            df['sid'] = df['q_num']
            df = df.groupby('qid', as_index=False).apply(lambda df: df.reset_index(drop=True).reset_index())
            df = df.drop('q_num', axis=1)
            df = df.rename({'index': 'q_num'}, axis=1)
            df = df.astype({'sid': int})
            df = df[['sid', 'q_num', 'qid']].sort_values(by=['sid', 'q_num'])
        return df.reset_index(drop=True)
