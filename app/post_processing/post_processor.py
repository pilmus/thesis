import os

import pandas as pd
from tqdm import tqdm

POSTPROCESSOR = None


def get_postprocessor():
    global POSTPROCESSOR
    if not POSTPROCESSOR:
        POSTPROCESSOR = PostProcessor()
    return POSTPROCESSOR


class PostProcessor():
    _app_entry = None
    _outfile = None

    def init(self, app_entry):
        self._app_entry = app_entry
        outdir = app_entry.get_argument('outdir')
        self._outfile = os.path.join(outdir, f"{self._app_entry.reranker_name}_{self._app_entry.config}.json")

    def write_submission(self, predictions):
        """
        accepts a model and writes a jsonlines submission file.
        """
        print("Writing submission...")
        predictions.sort_values(['sid', 'q_num', 'rank'], axis=0, inplace=True)
        tqdm.pandas()
        submission = predictions.groupby(['sid', 'q_num', 'qid']).progress_apply(
            lambda df: pd.Series({'ranking': df['doc_id']}))
        submission = submission.reset_index()
        q_num = [str(submission['sid'][i]) + "." + str(submission['q_num'][i]) for i in range(len(submission))]
        submission['q_num'] = q_num
        submission.drop('sid', axis=1, inplace=True)
        submission.to_json(self._outfile, orient='records', lines=True)
