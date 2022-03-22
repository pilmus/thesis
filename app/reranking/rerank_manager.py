import inspect
from enum import IntEnum

from app.pre_processing.pre_processor import get_preprocessor
from app.reranking.src.deltr import Deltr
from app.reranking.src.lambdamart import LambdaMartYear, LambdaMartRandomization
from app.reranking.src.random_shuffle import RandomRanker


class Reranker(IntEnum):
    RANDOM_SHUFFLE = 1
    LAMBDAMART = 2
    LAMBDAMART_R = 3
    DELTR = 4
    P_CONTROLLER = 5
    COORD_ASCENT = 6


def rerank(app_entry):
    reranker_num = app_entry.ranker_num

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
        # print(paramk, paramd[paramk])

    del paramd['self']
    print(paramd)

    reranker = reranker_class(**paramd)

    preprocessor = get_preprocessor()

    reranker.train(preprocessor.ioht)
    predictions = reranker.predict(preprocessor.iohe)
    return predictions
    # do a yield here? and then in the single run you just have a generator with only 1 return
    # in multi-run, you can keep yielding until you're donee