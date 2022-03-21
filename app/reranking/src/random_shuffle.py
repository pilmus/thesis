import random

from tqdm import tqdm

from app.reranking.src import model


class RandomRanker(model.RankerInterface):

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

    def load(self, path):
        pass

    def save(self, path):
        pass