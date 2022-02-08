import glob
import os

import pandas as pd

"""This script was used to confirm that the first four columns of each eval run are summary statistics over all submitted runs."""

rundir = 'resources/evaluation/2020/trec_runs/reranking'

runfiles = glob.glob(os.path.join(rundir, '*.csv'))

df = pd.read_csv(runfiles[0], sep='\t', dtype={'qid': str})[['qid']]

for runfile in runfiles:
    runname = runfile.split('.')[1]
    rundf = pd.read_csv(runfile, sep='\t', dtype={'qid': str})[['qid', 'run']]
    rundf = rundf.rename({'run': f'run-{runname}'}, axis=1)
    df = pd.merge(df, rundf, on='qid', how='left')


def compute_stats(row):
    statrow = row.drop('qid')
    row['min'] = round(statrow.min(), 7)
    row['max'] = round(statrow.max(), 7)
    row['mean'] = round(statrow.mean(), 7)
    row['median'] = round(statrow.median(), 7)
    # row[['min','max','mean','median']] = [rowmin,rowmax,rowmean,rowmedian]
    return row[['qid', 'min', 'max', 'mean', 'median']]


statsdf = df.apply(lambda row: compute_stats(row), axis=1)

ogdf = pd.read_csv(runfiles[0], sep='\t', dtype={'qid': str})[['qid', 'min', 'max', 'mean', 'median']]

comparison = statsdf.compare(ogdf)
comparison
