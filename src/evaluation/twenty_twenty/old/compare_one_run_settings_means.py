import os

import pandas as pd
"""This script was used to discover which mix of parameters is used in the official evaluation."""

basename = 'resources/evaluation/2020/eval_output'
runfiles = ['Deltr-gammas-ferraro-basic.tsv',
            'Deltr-gammas-ferraro-basic-sqrt.tsv',
            'Deltr-gammas-ferraro-basic-unnormalized.tsv',
            'Deltr-gammas-ferraro-basic-unnormalized-sqrt.tsv',
            'Deltr-gammas-ferraro-mix_down.tsv',
            'Deltr-gammas-ferraro-mix_down-sqrt.tsv',
            'Deltr-gammas-ferraro-mix_down-unnormalized.tsv',
            'Deltr-gammas-ferraro-mix_down-unnormalized-sqrt.tsv',
            'Deltr-gammas-ferraro-mix_up.tsv',
            'Deltr-gammas-ferraro-mix_up-sqrt.tsv',
            'Deltr-gammas-ferraro-mix_up-unnormalized.tsv',
            'Deltr-gammas-ferraro-mix_up-unnormalized-sqrt.tsv',
            'Deltr-gammas-ferraro-mixed_group.tsv',
            'Deltr-gammas-ferraro-mixed_group-sqrt.tsv',
            'Deltr-gammas-ferraro-mixed_group-unnormalized.tsv',
            'Deltr-gammas-ferraro-mixed_group-unnormalized-sqrt.tsv',
            'Deltr-gammas-ferraro-nomixed.tsv',
            'Deltr-gammas-ferraro-nomixed-sqrt.tsv',
            'Deltr-gammas-ferraro-nomixed-unnormalized.tsv',
            'Deltr-gammas-ferraro-nomixed-unnormalized-sqrt.tsv', ]

for runfile in runfiles:
    df = pd.read_csv(os.path.join(basename, runfile), sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    print(f"{runfile}:\t{df.difference.mean()}")
