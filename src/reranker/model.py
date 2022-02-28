from abc import ABC, abstractmethod

import pandas as pd


class RankerInterface(ABC):
    """
    This abstract class must be implemented by all training models
    """

    def __init__(self, featureengineer):
        self.fe = featureengineer
        self.predictions = pd.DataFrame(columns=['q_num', 'qid', 'doc_id', 'rank'])
        super().__init__()

    @abstractmethod
    def train(self, inputhandler, random_state=None, missing_value_strategy=None):
        """
        uses the labelled queries from the inputhandler to estimate a ranking model
        """
        pass

    def predict(self, inputhandler, missing_value_strategy=None):
        """
        uses the query sequences from the inputhandler to rerank the documents according to the trained model.
        must return a dataframe with columns [sid, q_num, qid, doc_id, rank]
        """
        self.predictions = self._predict(inputhandler, missing_value_strategy=missing_value_strategy)[
            ['sid', 'q_num', 'qid', 'doc_id', 'rank']]
        return self.predictions

    @abstractmethod
    def _predict(self, inputhandler, missing_value_strategy=None):
        pass
