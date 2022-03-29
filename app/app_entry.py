import itertools
import json
import sys

from app.evaluation.evaluator import evaluate, summarize, compare_means
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
    _basename = None

    _incrementables = None
    _incrstate = None

    _preproc_config = None
    _preproc_config_name = None

    _rankers = None

    def __init__(self):
        pass

    def load_config(self, filename):
        with open(filename, 'r') as fp:
            self._paramd = json.load(fp)

        self.rankers = self._paramd.get('rankers', [])

    def init_incrementables(self):
        """
        Set each incrementable to the list of the values it can take.
        :return:
        """

        self.incrementables = {}

        for incrementable in self.ranker.get("incrementables", []):
            self.incrementables[incrementable] = None

        for incrementable in self.incrementables:
            incr_text = self.config.get(incrementable)
            if incr_text is None:
                incr_text = self.configs['default'][incrementable]
            etext = f"self.incrementables[incrementable] = {incr_text}"
            exec(etext)

    @property
    def reranker_name(self):
        return self._reranker_name

    @reranker_name.setter
    def reranker_name(self, value):
        self._reranker_name = value

    @property
    def configs(self):
        return self.ranker.get("rerank_configs")

    @property
    def ranker(self):
        return self.rankers.get(self.reranker_name, None)

    @property
    def config_name(self):
        return self._config_name

    @property
    def preproc_config_name(self):
        return self._preproc_config_name

    @config_name.setter
    def config_name(self, value):
        self._config_name = value

    @property
    def config_incr_name(self):
        base = self.basename
        for k, v in self.incrstate.items():
            base = f"{base}_{k}={v}"

        return base

    @property
    def preproc_config(self):
        return self._preproc_config

    @property
    def config(self):
        return self.configs.get(self.config_name, None)

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

    @property
    def basename(self):
        return self._basename

    @basename.setter
    def basename(self, value):
        self._basename = value

    def get_argument(self, paramk):
        if paramk in self.incrementables:
            print(paramk)
            return self.incrstate[paramk]
        paramv = self.config.get(paramk, None)
        if paramv is None:
            paramv = self.configs.get('default', None).get(paramk, None)
        return paramv

    def entry(self):
        while True:
            print("What do you want to do?")
            print("1: Run")
            print("2: Analyse")
            print("3: Quit")
            choice = int(input("$ ") or 1)

            if choice == 1:
                self.run()
            elif choice == 2:
                self.analyze()
            elif choice == 3:
                sys.exit(0)
            else:
                raise ValueError(f"Invalid choice: {choice}.")

    def common_logic(self):
        self.load_config('config/appconfig.json')
        print("Choose a ranker:")
        for reranker in Reranker:
            print(f"{reranker.value}: {reranker.name}")
        reranker_num = int(input("$ ") or 2)
        self.reranker_name = Reranker(reranker_num).name.lower()

        pre_configs = self.ranker.get('preproc_config')
        prprp_keys = list(pre_configs.keys())
        if len(prprp_keys) > 1:
            print("Choose a preprocessing configuration:")
            for i, pre_config in enumerate(prprp_keys):
                print(f"{i + 1}: {pre_config}")

            preproc_choice = int((input("$ ") or 1))
            prpr_key = prprp_keys[preproc_choice - 1]
            self._preproc_config = pre_configs[prpr_key]
        else:
            print("Using default preprocessing configuration.")
            prpr_key = prprp_keys[0]
            self._preproc_config = next(iter(pre_configs.values()))
        self._preproc_config_name = prpr_key

        config_list = list(self.configs.keys())

        if len(config_list) > 1:
            print("Choose a configuration:")
            for i, config in enumerate(config_list):
                print(f"{i + 1}: {config}")
            config_idx = int(input("$ ") or 1)
            config_name = config_list[config_idx - 1]
            self.config_name = config_name
        else:
            print("Using default ranker configuration.")
            self.config_name = config_list[0]

        self.basename = f"{self.preproc_config_name}_{self.config_name}"

    def run(self):

        self.common_logic()

        self.init_incrementables()

        get_preprocessor().init(self)

        # from here
        print(self.incrementables)
        for incrcombo in dict_product(self.incrementables):
            self.incrstate = incrcombo

            predictions = rerank(self)

            get_postprocessor().init(self)
            get_postprocessor().write_submission(predictions)

            evaluate(self)
        self.analyze_logic()
        # to here
        # has to be repeated on a multirun
        # preprocessor is not repeated because we do multi-runs with e.g. different seeds, different params
        # if you want to do multi runs with different pre-processing steps you should make a different configuration
        # is doable b/c you might want to run over a thousand parameters but not over a thousand input sequences in this case

    def analyze(self):
        self.common_logic()
        self.init_incrementables()

        self.analyze_logic()

    def analyze_logic(self):
        # todo add evaluate
        # print("Evaluate?")
        # i = input("[y/n] ")
        # if i == 'y':
        #

        print("Compare means?")
        i = input("[y/n] ")
        if i == 'y':
            compare_means(self)
        # kendall_tau(self)
        print("Summarize?")
        i = input("[y/n] ")
        if i == 'y':
            summarize(self)


def main():
    app_entry = AppEntry()
    app_entry.entry()


if __name__ == '__main__':
    main()
