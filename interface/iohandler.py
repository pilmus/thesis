from itertools import chain

import pandas as pd
from tqdm import tqdm

import utils.io as io


class InputOutputHandler:
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

        queries = io.read_jsonlines(fquery, handler=self.__unnest_query)
        queries = list(chain.from_iterable(queries))
        self.seq = self.__read_sequence(fsequence)
        self.queries = pd.DataFrame(queries)

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

    def write_submission(self, model, outfile):
        """
        accepts a model and writes a jsonlines submission file.
        """
        print("Writing submission...")
        model.predictions.sort_values(['sid', 'q_num', 'rank'], axis=0, inplace=True)
        tqdm.pandas()
        submission = model.predictions.groupby(['sid', 'q_num', 'qid']).progress_apply(
            lambda df: pd.Series({'ranking': df['doc_id']}))
        submission = submission.reset_index()
        q_num = [str(submission['sid'][i]) + "." + str(submission['q_num'][i]) for i in range(len(submission))]
        submission['q_num'] = q_num
        submission.drop('sid', axis=1, inplace=True)
        submission.to_json(outfile, orient='records', lines=True)
