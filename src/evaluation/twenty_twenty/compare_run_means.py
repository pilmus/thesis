import glob
import os

import pandas as pd

basename = 'resources/evaluation/2020/eval_output'

eval_files = glob.glob(os.path.join(basename, 'trec_run.*-mixed_group-unnormalized.tsv'))


print(os.path.join(basename, 'trec_run.*.tsv'))
for eval in eval_files:
    df = pd.read_csv(eval, sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    print(f"{os.path.basename(eval)}:\t{round(df.difference.mean(),3)}")
