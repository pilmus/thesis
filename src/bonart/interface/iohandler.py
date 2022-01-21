import sys
from itertools import chain

import pandas as pd

import src.bonart.utils.io as io


class InputOutputHandler:
    """Interface between the provided training data and other modules. 
    When initialized author information for each doc is fetched from the database via the
    provided Corpus object."""

    def __init__(self,
                 corpus,
                 fsequence,
                 fquery):
        """

        :param corpus:
        :param fsequence: training query sequence (e.g. training-sequence.tsv)
        :param fquery: training queries (e.g. fair-TREC-training-sample.json)
        :param fgroup: grouping file mapping a certain entity to a group. This can be author-to-group as in
        fair-TREC-sample-author-groups.csv or docid-to-group as in TREC-Fair-Ranking-eval-sample-groups.csv (
        generated with eval_sample_annotated.py)
        """

        self.corpus = corpus

        queries = io.read_jsonlines(fquery, handler=self.__unnest_query)
        queries = list(chain.from_iterable(queries))

        sequence_df = pd.read_csv(fsequence, names=['sid_q_num','qid'], dtype={'sid_q_num':'str'}, sep=',', engine='python')
        if sequence_df.sid_q_num.str.contains('.', regex=False).any():
            sequence_df[['sid','q_num']] = sequence_df.sid_q_num.str.split('.',expand = True)
            sequence_df = sequence_df.drop('sid_q_num',axis = 1)
        else:
            sequence_df['sid'] = 0
            sequence_df = sequence_df.rename(columns={'sid_q_num':'q_num'})

        sequence_df = sequence_df[['sid','q_num','qid']]

        self.seq = sequence_df
        self.queries = pd.DataFrame(queries)
        # self.groups = pd.read_csv(fgroup, dtype={"author_id": str, "gid": str})
        # self.authors = None

    def get_queries(self):
        return self.queries.drop_duplicates()

    def get_query_seq(self):
        seq = pd.merge(self.seq, self.queries, on="qid", how='left')
        return seq
        # return seq.drop_duplicates() # no dropping duplicates, double queries in a sequence are ok?


    # def get_authors(self):
    #     if self.authors is None:
    #         ids = self.queries.doc_id.drop_duplicates().tolist()
    #         self.authors = self.corpus.get_authors(ids)
    #         self.authors = pd.merge(self.authors, self.groups, on="author_id", how='left')
    #     return self.authors

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

    def write_submission(self, model, outfile):
        """
        accepts a model and writes a jsonlines submission file.
        """
        model.predictions.sort_values(['sid', 'q_num', 'rank'], axis=0, inplace=True)
        submission = model.predictions.groupby(['sid', 'q_num', 'qid']).apply(
            lambda df: pd.Series({'ranking': df['doc_id']}))
        submission = submission.reset_index()
        submission.q_num = submission.sid.astype(str) + '.' + submission.q_num.astype(str)
        # q_num = [str(submission['sid'][i]) + "." + str(submission['q_num'][i]) for i in range(len(submission))]
        # submission['q_num'] = q_num
        submission = submission[['q_num','qid','ranking']]
        # submission.drop('sid', axis=1, inplace=True)
        submission.to_json(outfile, orient='records', lines=True)
