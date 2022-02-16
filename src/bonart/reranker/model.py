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
    def train(self, inputhandler):
        """
        uses the labelled queries from the inputhandler to estimate a ranking model
        """
        pass

    def predict(self, inputhandler, prepped_data=None):
        """
        uses the query sequences from the inputhandler to rerank the documents according to the trained model.
        must return a dataframe with columns [sid, q_num, qid, doc_id, rank]
        """
        self.predictions = self._predict(inputhandler,prepped_data=prepped_data)[['sid', 'q_num', 'qid', 'doc_id', 'rank']]
        return self.predictions

    @abstractmethod
    def save(self, path):
        """
        Save the trained model.
        """
        pass

    @abstractmethod
    def load(self, path):
        """load trained model"""
        pass

    @abstractmethod
    def _predict(self, inputhandler,prepped_data=None):
        pass


