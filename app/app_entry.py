import itertools
import json
import os.path
import sys
import random

import pandas as pd
from sklearn.datasets import dump_svmlight_file

from app.evaluation.evaluator import evaluate, summarize, compare_means
from app.evaluation.src.y2020.eval.trec.json2qrels import json_to_group_qrels, json_to_base_qrels
from app.pre_pre_processing.src.merged_annotations_to_groups import annotations_to_groups, MappingMode, Grouping
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
        paths = [("Run", self.run), ("Prepare", self.prepare), ("Analyse", self.analyze), ("Quit", self.quit)]
        while True:
            print("What do you want to do?")
            for i, path in enumerate(paths):
                print(f"{i + 1}: {path[0]}")
            choice = int(input("$ ") or 1)

            choice = choice - 1

            if choice > len(paths):
                print(f"Invalid choice: {choice + 1}.")

            path_method = paths[choice][1]
            path_method()
            #
            # if choice == 1:
            #     self.run()
            # elif choice == 2:
            #     self.analyze()
            # elif choice == 3:
            #     sys.exit(0)
            # else:

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
        self.init_incrementables()

    def run(self):

        self.common_logic()

        get_preprocessor().init(self)

        # from here
        print(self.incrementables)
        for incrcombo in dict_product(self.incrementables):
            self.incrstate = incrcombo

            predictions = rerank(self)

            get_postprocessor().init(self)
            get_postprocessor().write_submission(predictions)

            evaluate(self)
        # self.analyze_logic()
        # to here
        # has to be repeated on a multirun
        # preprocessor is not repeated because we do multi-runs with e.g. different seeds, different params
        # if you want to do multi runs with different pre-processing steps you should make a different configuration
        # is doable b/c you might want to run over a thousand parameters but not over a thousand input sequences in this case

    def prepare(self):
        print("What do you want to prepare?")
        print(f"1: Group mapping file")
        print(f"2: Qrels")
        print(f"3: Feature file (ES)")
        print(f"4: Feature file (SVM)")
        print(f"5: Augmented training sample")
        choice = int(input("$ (default: 2)") or 2)

        if choice == 1:
            valid_training_sample = False
            while not valid_training_sample:
                print("Enter the path to the training sample")
                training_sample = str(input(
                    "$ (default: pre_processing/resources/training/2020/TREC-Fair-Ranking-training-sample.json)") or "pre_processing/resources/training/2020/TREC-Fair-Ranking-training-sample.json")
                valid_training_sample = os.path.exists(training_sample)

            valid_eval_sample = False
            while not valid_eval_sample:
                print("Enter the path to the evaluation sample")
                eval_sample = str(input(
                    "$ (default: pre_processing/resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json)") or "pre_processing/resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json")
                valid_eval_sample = os.path.exists(eval_sample)

            valid_doc_annotations = False
            while not valid_doc_annotations:
                print("Enter the path to the document annotations")
                doc_annotations = str(input(
                    "$ (default: pre_pre_processing/resources/doc-annotations.csv)") or "pre_pre_processing/resources/doc-annotations.csv")
                valid_doc_annotations = os.path.exists(doc_annotations)

            valid_gm = False
            while not valid_gm:
                print("Choose a grouping")
                for grouping_mode in Grouping:
                    print(f"{grouping_mode.value}: {grouping_mode.name}")
                gm_val = int(input("$ ") or 1)
                gm = Grouping(gm_val).name
                valid_gm = Grouping.has_value(gm_val)

            valid_mm = False
            while not valid_mm:
                for mapping_mode in MappingMode:
                    print("Choose a mapping mode")
                    print(f"{mapping_mode.value}: {mapping_mode.name}")
                mm_val = int(input("$ ") or 1)
                mm = MappingMode(mm_val).name.lower()
                valid_mm = MappingMode.has_value(mm_val)

            grouping = annotations_to_groups(training_sample, eval_sample, doc_annotations, gm, mm)

            outfile = f"full-annotations-{gm}-{mm}.csv"

            valid_outdir = False
            while not valid_outdir:
                print("Enter the save location of the grouping file")
                outdir = str(
                    input("$ (default: evaluation/resources/2020/groupings)") or "evaluation/resources/2020/groupings")
                valid_outdir = os.path.exists(outdir)

            outfile = os.path.join(outdir, outfile)

            grouping.to_csv(outfile, index=False)
        elif choice == 2:
            valid_sample = False
            while not valid_sample:
                print("Enter the path to the ground truth file")
                sample = str(input(
                    "$ (default: pre_processing/resources/training/2020/TREC-Fair-Ranking-training-sample.json)") or "pre_processing/resources/training/2020/TREC-Fair-Ranking-training-sample.json")
                valid_sample = os.path.exists(sample)

            print("Group qrels or base qrels?")
            print("1: Group")
            print("2: Base")
            choice1 = int(input("$ (default: 1)") or 1)

            if choice1 == 1:

                valid_grouping = False
                while not valid_grouping:
                    print("Enter the path to grouping file")
                    grouping = str(input(
                        "$ (default: evaluation/resources/2020/groupings/full-annotations-mixed_group.csv)") or "evaluation/resources/2020/groupings/full-annotations-mixed_group.csv")
                    valid_grouping = os.path.exists(grouping)

                valid_outdir = False
                while not valid_outdir:
                    print("Enter the path to the outdir")
                    outdir = str(input(
                        "$ (default: evaluation/resources/2020/qrels)") or "evaluation/resources/2020/qrels")
                    valid_outdir = os.path.exists(outdir)

                outfile = f"{os.path.splitext(grouping)[0]}-qrels.tsv"

                json_to_group_qrels(sample, grouping, outfile, complete=True, not_verbose=False)
            elif choice1 == 2:
                valid_outdir = False
                while not valid_outdir:
                    print("Enter the path to the outdir")
                    outdir = str(input(
                        "$ (default: evaluation/resources/2020/qrels)") or "evaluation/resources/2020/qrels")
                    valid_outdir = os.path.exists(outdir)

                print("Enter the desired file name")
                outfile = str(input(
                    "$ (default: qrels.qrel)") or "qrels.qrel")

                json_to_base_qrels(sample, os.path.join(outdir, outfile))
        elif choice == 3:
            self.common_logic()

            pr = get_preprocessor()
            pr.init(self)
            # todo: move this logic to preprocessor


            tf = pr.fe.get_feature_mat(pr.ioht)
            ef = pr.fe.get_feature_mat(pr.iohe) #todo: make it so that you can extract features from ioht or iohe seperately

            ff = pd.concat([tf, ef]).drop_duplicates()
            pr.save_feature_mat(ff)
        elif choice == 4:
            print("Convert training or test file?")
            print(f"1: Training")
            print(f"2: Test")
            choice1 = int(input("$ (default: 1)") or 1)
            self.common_logic()

            pr = get_preprocessor()
            pr.init(self)
            if choice1 == 1:
                ioh = pr.ioht
            elif choice1 == 2:
                ioh = pr.iohe
            else:
                return  # todo: ?¿?

            fm = pr.fe.get_feature_mat(ioh)
            fm = pd.merge(fm, ioh.get_query_seq()[['qid', 'doc_id', 'relevance']].drop_duplicates(),
                          on=['qid', 'doc_id'])
            fm = fm.sort_values(by='qid')

            qids = fm['qid'].to_list()
            y = fm['relevance'].to_list()
            X = fm.drop(['qid', 'relevance', 'doc_id'], axis=1)
            docids = fm['doc_id']

            print("Sparse or dense?")
            print(f"1: Sparse")
            print(f"2: Dense")
            choice_sparsedense = bool(int(input("$ (default: 1)") or 1) - 1)

            print("Zero- or one indexed?")
            print(f"1: Zero")
            print(f"2: One")
            choice_indexing = bool(int(input("$ (default: 1)") or 1) - 1)

            pr.dump_svm(X, y, qids, docids, choice_sparsedense, choice_indexing)
        elif choice == 5:

            valid_training_sample = False
            while not valid_training_sample:
                print("Enter the path to the training sample")
                training_sample = str(input(
                    "$ (default: pre_processing/resources/training/2020/TREC-Fair-Ranking-training-sample.json)") or "pre_processing/resources/training/2020/TREC-Fair-Ranking-training-sample.json")
                valid_training_sample = os.path.exists(training_sample)

            print("What percentage of documents should be sampled?")
            frac = float(input("$ "))

            dfs = [pd.read_json(training_sample, lines=True)]
            for i in range(5): #todo: make range param configurable
                df = pd.read_json(training_sample, lines=True)
                df[['qid','documents']] = df.apply(lambda row: pd.Series({"qid":f"{row.qid}_{i+1}","documents":random.sample(row.documents, int(frac * len(row.documents)))}), axis=1)
                dfs.append(df)

            aug_df = pd.concat(dfs)

            outpath = f"{os.path.splitext(training_sample)[0]}_aug{frac}.json"
            aug_df.to_json(outpath,orient='records',lines=True)

            aug_df.qid.drop_duplicates().reset_index(drop=True).to_csv(os.path.join(os.path.dirname(training_sample),f"training-sequence-full_aug{frac}.tsv"))


            # load training file
            # for each qid, sample
            pass
        else:
            print("Invalid choice: ", choice)

    def analyze(self):
        self.common_logic()

        self.analyze_logic()

    def quit(self):
        sys.exit(0)

    def analyze_logic(self):
        # todo add evaluate
        print("Evaluate?")
        i = input("[y/n] ")
        if i == 'y':
            get_preprocessor().init(self)
            for incrcombo in dict_product(self.incrementables):
                self.incrstate = incrcombo
                get_postprocessor().init(self)
                evaluate(self)

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
