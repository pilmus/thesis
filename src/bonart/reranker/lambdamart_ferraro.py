import pickle

import pyltr
import pandas as pd
import src.bonart.reranker.model as model
from bonart.reranker.lambdamart import LambdaMart


class LambdaMartFerraro(LambdaMart):
    """
    Extends Bonart's LambdaMart wrapper with a predict function for reproducing the results in
    Ferraro, Porcaro, and Serra, ‘Balancing Exposure and Relevance in Academic Search’.
    """

    def __randomize_apply(self,df):
        df
        return df

    def _predict(self, inputhandler):
        x, y, qids, tmp1, tmp2, tmp3 = self._prepare_data(inputhandler, frac=1)
        pred = self.lambdamart.predict(x)

        qids = qids.assign(pred=pred)
        qids.groupby(['q_num']).apply(self.__randomize_apply)

        qids.loc[:, 'rank'] = qids.groupby('q_num')['pred'].apply(pd.Series.rank, ascending=False, method='first')
        qids.drop('pred', inplace=True, axis=1)
        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], qids,
                        how='left', on=['sid', 'q_num', 'doc_id'])
        return pred

    # def save(self, path):
    #     print(f"Saving models...")
    #
    #     with open(path, 'wb') as fp:
    #         pickle.dump(self.lambdamart, fp)
    #     return True
    #
    # def load(self, path):
    #     with open(path, "rb") as fp:
    #         self.lambdamart = pickle.load(fp)
