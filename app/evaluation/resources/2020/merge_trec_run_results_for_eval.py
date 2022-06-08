import glob
import os

import pandas as pd
from tqdm import tqdm

eval_results = '/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/trec_run'
eel = glob.glob(os.path.join(eval_results, '*EEL.tsv'))
eeli = glob.glob(os.path.join(eval_results, '*EEL_ind.tsv'))

print(eel)

columns = ['difference', 'disparity', 'relevance', 'difference_ind', 'disparity_ind', 'relevance_ind', 'qid', 'source']


def pivot_table(table_path):
    df = pd.read_csv(table_path, sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    df = df.reset_index()
    return df


concat = []
for e, ei in tqdm(zip(eel, eeli), total=len(eel)):
    edf = pivot_table(e)
    eidf = pivot_table(ei).rename(
        {'difference': 'difference_ind', 'disparity': 'disparity_ind', 'relevance': 'relevance_ind'}, axis=1)

    merge = pd.merge(edf, eidf, on='qid')
    merge = merge.reindex(columns=columns)
    merge.source = os.path.splitext(os.path.basename(e))[0]

    print(os.path.splitext(os.path.basename(e))[0])

    concat.append(merge)

outdf = pd.concat(concat)
outdf.to_csv('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/all_experiments_eval_tr.csv', index=False)

