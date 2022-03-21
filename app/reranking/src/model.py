from abc import ABC, abstractmethod

import pandas as pd

from app.pre_processing.pre_processor import get_preprocessor


class RankerInterface(ABC):
    """
    This abstract class must be implemented by all training models
    """

    def __init__(self):
        self.fe = get_preprocessor().fe
        self.predictions = pd.DataFrame(columns=['q_num', 'qid', 'doc_id', 'rank'])
        super().__init__()

    @abstractmethod
    def train(self, inputhandler):
        """
        uses the labelled queries from the inputhandler to estimate a ranking model
        """
        pass

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
