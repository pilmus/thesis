import random
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm


class RankerInterface(ABC):
    """
    This abstract class must be implemented by all training models
    """

    def __init__(self, featureengineer):
        self.fe = featureengineer
        self.predictions = pd.DataFrame(columns=['q_num', 'qid', 'doc_id', 'rank'])
        super().__init__()

    @abstractmethod
    def train(self, inputhandler,save):
        """
        uses the labelled queries from the inputhandler to estimate a ranking model
        """
        pass

    # @abstractmethod #todo necessary?
    # def __grouping_apply(self, df, grouping_file):
    #     pass

    def predict(self, inputhandler):
        """
        uses the query sequences from the inputhandler to rerank the documents according to the trained model. 
        must return a dataframe with columns [sid, q_num, qid, doc_id, rank]
        """
        self.predictions = self._predict(inputhandler)[['sid', 'q_num', 'qid', 'doc_id', 'rank']]
        return self.predictions

    @abstractmethod
    def _predict(self, inputhandler):
        pass


class RandomRanker(RankerInterface):

    def __shuffle_group(self, group):
        group.loc[:, 'doc_id'] = random.sample(group['doc_id'].to_list(), len(group['doc_id']))
        group.loc[:, 'rank'] = range(1, len(group['doc_id']) + 1)
        return group

    def _predict(self, inputhandler):
        tqdm.pandas()
        pred = inputhandler.get_query_seq().groupby(['sid', 'q_num', 'qid']).progress_apply(self.__shuffle_group)
        return pred

    def train(self, inputhandler):
        pass