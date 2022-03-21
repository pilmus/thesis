# inputs:
# - application instance
#       - ranker name
#       - ranker arguments
# - train and eval dataframes
import inspect
from enum import IntEnum

from app.preprocessing.preprocessor import get_preprocessor
from reranker.deltr import Deltr
from reranker.lambdamart import LambdaMartYear, LambdaMartRandomization
from reranker.random_ranker import RandomRanker


class Reranker(IntEnum):
    RANDOM_SHUFFLE = 1
    LAMBDAMART = 2
    LAMBDAMART_R = 3
    DELTR = 4
    P_CONTROLLER = 5
    COORD_ASCENT = 6


def rerank(app_entry):
    reranker_num = app_entry.get_ranker()

    rerankers = {1: RandomRanker,
                 2: LambdaMartYear,
                 3: LambdaMartRandomization,
                 4: Deltr,
                 # 5: PController,
                 # 6: CoordinateAscent,
                 }

    reranker_class = rerankers[reranker_num]

    reranker_args = inspect.signature(reranker_class.__init__)

    paramd = {}
    for paramk in reranker_args.parameters.keys():
        paramd[paramk] = app_entry.get_argument(paramk)

    del paramd['self']
    print(paramd)

    reranker = reranker_class(**paramd)

    preprocessor = get_preprocessor()

    reranker.train(preprocessor)
    reranker.predict(preprocessor)