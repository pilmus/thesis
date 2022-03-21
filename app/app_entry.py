import json

from app.pre_processing.pre_processor import get_preprocessor

from app.post_processing.post_processor import get_postprocessor
from app.reranking.rerank_manager import rerank, Reranker


class AppEntry:
    _paramd = None
    _reranker_name = None
    _config = None

    def __init__(self):
        pass

    def load_config(self, filename):
        with open(filename, 'r') as fp:
            self._paramd = json.load(fp)

    @property
    def reranker_name(self):
        return self._reranker_name

    @reranker_name.setter
    def reranker_name(self, value):
        self._reranker_name = value

    @property
    def config(self):
        return self._config

    def get_ranker(self):
        return Reranker.LAMBDAMART

    def get_argument(self, paramk):
        paramv = self._paramd.get(self._reranker_name, None).get(self._config, None).get(paramk, None)
        # print(paramk, paramv)
        if paramv is None:
            paramv = self._paramd.get(self._reranker_name, None).get('default', None).get(paramk, None)
        # print(paramk, paramv)
        return paramv

    def run(self):
        print("Choose a ranker:")
        self.load_config('config/appconfig.json')

        for reranker in Reranker:
            print(f"{reranker.value}: {reranker.name}")

        reranker_num = input("$ ")
        self.reranker_name = Reranker(int(reranker_num)).name.lower()

        print("Choose a configuration:")
        config_list = list(self._paramd.get(self._reranker_name, None).keys())
        for i, config in enumerate(config_list):
            print(f"{i}: {config}")

        config_idx = int(input("$ "))
        config_name = config_list[config_idx]
        self._config = config_name

        get_preprocessor().init(self)

        predictions = rerank(self)

        get_postprocessor().init(self)
        get_postprocessor().write_submission(predictions)


def main():
    app_entry = AppEntry()
    app_entry.run()


if __name__ == '__main__':
    main()
