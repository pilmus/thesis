from app.reranking.src import model


class MRFR(model.RankerInterface):

    def __init__(self, relevance_probabilities, grouping, K, beta, lambd, k):
