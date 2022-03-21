from app.reranking.rerank_manager import rerank, Reranker


paramd = {}

class AppEntry:
    def __init__(self):
        pass

    def get_ranker(self):
        return Reranker.LAMBDAMART

    def get_argument(self, paramk):
        paramv = paramd.get(paramk,None)
        return paramv


def main():
    app_entry = AppEntry()
    rerank(app_entry)

if __name__ == '__main__':
    main()