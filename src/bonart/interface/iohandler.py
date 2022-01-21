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
        :param fgroup: author groups (e.g. fair-TREC-sample-author-groups.csv)
        """

        self.corpus = corpus

        queries = io.read_jsonlines(fquery, handler=self.__unnest_query)
        queries = list(chain.from_iterable(queries))
        self.seq = pd.read_csv(fsequence, names=["sid", "q_num", "qid"], sep='\.|,', engine='python')
        self.queries = pd.DataFrame(queries)

    def get_queries(self):
        return self.queries.drop_duplicates()

    def get_query_seq(self):
        seq = pd.merge(self.seq, self.queries, on="qid", how='left')
        seq = seq.dropna() #todo: good solution?
        return seq.drop_duplicates()

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
        submission.reset_index(inplace=True)
        q_num = [str(submission['sid'][i]) + "." + str(submission['q_num'][i]) for i in range(len(submission))]
        submission['q_num'] = q_num
        submission.drop('sid', axis=1, inplace=True)
        submission.to_json(outfile, orient='records', lines=True)
