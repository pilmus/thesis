import inspect
from enum import IntEnum

from app.pre_processing.pre_processor import get_preprocessor
from app.reranking.src.deltr import Deltr
from app.reranking.src.lambdamart import LambdaMartYear, LambdaMartRandomization, LambdaMartMRFR
from app.reranking.src.mrfr import MRFR
from app.reranking.src.p_controller import PController
from app.reranking.src.random_shuffle import RandomRanker


class Reranker(IntEnum):
    RANDOM_SHUFFLE = 1
    LAMBDAMART = 2
    LAMBDAMART2020 = 3
    P_CONTROLLER = 4
    MRFR = 5
    LAMBDAMART_MRFR = 6
    # LAMBDAMART_R = 3
    # DELTR = 4
    # COORD_ASCENT = 6


def rerank(app_entry):
    reranker_num = app_entry.ranker_num

    rerankers = {1: RandomRanker,
                 2: LambdaMartYear,
                 3: LambdaMartYear,
                 4: PController,
                 5: MRFR,
                 6: LambdaMartMRFR
                 # 3: LambdaMartRandomization,
                 # 4: Deltr,
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
