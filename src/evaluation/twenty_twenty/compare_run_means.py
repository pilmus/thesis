import glob
import os

import pandas as pd

basename = 'resources/evaluation/2020/eval_output'

eval_files = glob.glob(os.path.join(basename, 'trec_run.*-mixed_group-unnormalized.tsv'))


print(os.path.join(basename, 'trec_run.*.tsv'))

eval_files = [os.path.join(basename,'deltr_gammas_std_True-full-iter-5-relweight-0.5-unnormalized-mixed_group.tsv'),
              os.path.join(basename,'deltr_gammas_std_True-full-iter-5-relweight-0.5-unnormalized-sqrt-mixed_group.tsv'),
              os.path.join(basename, 'trec_run','trec_run.Deltr-gammas-mixed_group-unnormalized.tsv'),
              os.path.join(basename, 'param_investigation','Deltr-gammas-ferraro-mix_up-unnormalized-sqrt.tsv'),]

print(eval_files)
for eval in eval_files:
    df = pd.read_csv(eval, sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    print(f"{os.path.basename(eval)}:\t{round(df.difference.mean(),3)}")
