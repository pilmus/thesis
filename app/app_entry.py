import itertools
import json
import sys

from app.evaluation.evaluator import evaluate
from app.pre_processing.pre_processor import get_preprocessor

from app.post_processing.post_processor import get_postprocessor
from app.reranking.rerank_manager import rerank, Reranker


def dict_product(dicts):
    """
    https://stackoverflow.com/a/40623158/5128654
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))


class AppEntry:
    _paramd = None
    _reranker_name = None
    _config_name = None
    _incrementables = None
    _incrstate = None
    _rankers = None

    def __init__(self):
        pass

    def load_config(self, filename):
        with open(filename, 'r') as fp:
            self._paramd = json.load(fp)

        self.incrementables = {}
        for incrementable in self._paramd.get("incrementables", []):
            self.incrementables[incrementable] = None

        self.rankers = self._paramd.get('rankers', [])
        for ranker, configs in self.rankers.items():
            for config, params in configs.items():
                for incrementable in self.incrementables:
                    incr_text = params.get(incrementable)
                    if incr_text is not None:
                        etext = f"params[incrementable] = {incr_text}"
                        exec(etext)

    def init_incrementables(self):
        """
        Set each incrementable to the list of the values it can take.
        :return:
        """
        for incrementable in self.incrementables:
            self.incrementables[incrementable] = self.config.get(incrementable)

    @property
    def reranker_name(self):
        return self._reranker_name

    @reranker_name.setter
    def reranker_name(self, value):
        self._reranker_name = value

    @property
    def configs(self):
        return self.rankers.get(self.reranker_name, None)

    @property
    def config_name(self):
        return self._config_name

    @property
    def config_incr_name(self):
        base = self.config_name
        for k, v in self.incrstate.items():
            base = f"{base}_{k}={v}"

        return base

    @property
    def config(self):
        return self.rankers.get(self.reranker_name, None).get(self.config_name, None)

    @property
    def incrementables(self):
        return self._incrementables

    @incrementables.setter
    def incrementables(self, value):
        self._incrementables = value

    @property
    def incrstate(self):
        return self._incrstate

    @incrstate.setter
    def incrstate(self, value):
        self._incrstate = value

    @property
    def rankers(self):
        return self._rankers

    @rankers.setter
    def rankers(self, value):
        self._rankers = value

    @property
    def ranker_num(self):
        return Reranker[self._reranker_name.upper()].value
        # return Reranker.LAMBDAMART

    def get_argument(self, paramk):
        if paramk in self.incrementables:
            print(paramk)
            return self.incrstate[paramk]
        paramv = self.config.get(paramk, None)
        if paramv is None:
            paramv = self.rankers.get(self._reranker_name, None).get('default', None).get(paramk, None)
        return paramv

    def run(self):

        self.load_config('config/appconfig.json')

        print("Choose a ranker:")
        for reranker in Reranker:
            print(f"{reranker.value}: {reranker.name}")

        reranker_num = int(input("$ ") or 2)
        self.reranker_name = Reranker(reranker_num).name.lower()

        print("Choose a configuration:")
        config_list = list(self._paramd["rankers"].get(self._reranker_name, None).keys())
        for i, config in enumerate(config_list):
            print(f"{i}: {config}")

        config_idx = int(input("$ ") or 0)
        config_name = config_list[config_idx]
        self._config_name = config_name

        self.init_incrementables()

        get_preprocessor().init(self)

        # from here
        for incrcombo in dict_product(self.incrementables):
            self.incrstate = incrcombo

            predictions = rerank(self)

            get_postprocessor().init(self)
            get_postprocessor().write_submission(predictions)

            evaluate(self)
        # to here
        # has to be repeated on a multirun
        # preprocessor is not repeated because we do multi-runs with e.g. different seeds, different params
        # if you want to do multi runs with different pre-processing steps you should make a different configuration
        # is doable b/c you might want to run over a thousand parameters but not over a thousand input sequences in this case


def main():
    app_entry = AppEntry()
    app_entry.run()


if __name__ == '__main__':
    main()
