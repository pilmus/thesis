import glob
import os
import sys

import pandas as pd
from tqdm import tqdm

eval_results = '/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/'

eel = glob.glob('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/*EEL.tsv')
eeli = glob.glob('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/*ERR.tsv')
util = glob.glob('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/*util.tsv')

# eel = glob.glob('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/ac*EEL.tsv')
# eeli = glob.glob('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/ac*ERR.tsv')
# util = glob.glob('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/ac*util.tsv')

columns = ['difference', 'disparity', 'relevance', 'difference_ind', 'disparity_ind', 'relevance_ind', 'util', 'qid',
           'ranker', 'source', 'group', 'subgroup', 'theta', 'hfunc', 'augmentation', 'val_metric',
           'feature_method', 'num_features', 'balancing_factor']


def pivot_table(table_path):
    df = pd.read_csv(table_path, sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    df = df.reset_index()
    return df


concat = []
for e, ei, u in tqdm(zip(eel, eeli, util), total=len(eel)):
    edf = pivot_table(e)
    eidf = pivot_table(ei).rename(
        {'difference': 'difference_ind', 'disparity': 'disparity_ind', 'relevance': 'relevance_ind'}, axis=1)
    udf = pivot_table(u)
    merge = pd.merge(pd.merge(edf, eidf, on='qid'), udf, on='qid')
    merge = merge.reindex(columns=columns)

    fname = os.path.splitext(os.path.basename(e))[0]
    components = fname.split('_')
    if fname.startswith('ac'):
        if len(components) == 12:
            # ['ac', 'controller', 'train10', 'meta', '99', 'linear', 'doc', 'train', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
            source = f'{components[3].upper()}_train'
            theta = components[4]
            hfunc = components[5]
            group = components[6]
            subgroup = '-'

            augmentation = '-'
            val_metric = '-'
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'

        elif len(components) == 13:
            # ['ac', 'controller', 'train10', 'meta', '99', 'linear', 'author', 'ind', 'train', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
            source = f'{components[3].upper()}_train'
            theta = components[4]
            hfunc = components[5]
            group = components[6]
            subgroup = components[7]

            augmentation = '-'
            val_metric = '-'
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'


        elif len(components) == 15:
            # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'err', 'aug0.5', 'nofeat', 'doc', '99', 'linear', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
            source = f'lambdamart_{components[6]}_{components[5]}_{components[7]}'
            theta = components[9]
            hfunc = components[10]
            group = components[8]
            subgroup = '-'

            augmentation = components[6]
            val_metric = components[5]
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'

        elif len(components) == 16:
            # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'err', 'aug0.5', 'nofeat', 'author', 'ind', '99', 'linear', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
            source = f'lambdamart_{components[6]}_{components[5]}_{components[7]}'
            theta = components[10]
            hfunc = components[11]
            group = components[8]
            subgroup = components[9]

            augmentation = components[6]
            val_metric = components[5]
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'

        elif len(components) == 17:
            # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'err', 'aug0.5', 'mpt', '10', '0.1', 'doc', '99', 'linear', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
            source = f'lambdamart_{components[6]}_{components[5]}_{components[7]}_{components[8]}_{components[9]}'
            theta = components[11]
            hfunc = components[12]
            group = components[10]
            subgroup = '-'

            augmentation = components[6]
            val_metric = components[5]
            feature_method = components[7]
            num_features = components[8]
            balancing_factor = components[9]

        elif len(components) == 18:
            # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'err', 'aug0.5', 'mpt', '10', '0.1', 'author', 'ind', '99', 'linear', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
            source = f'lambdamart_{components[6]}_{components[5]}_{components[7]}_{components[8]}_{components[9]}'
            theta = components[12]
            hfunc = components[13]
            group = components[10]
            subgroup = components[11]

            augmentation = components[6]
            val_metric = components[5]
            feature_method = components[7]
            num_features = components[8]
            balancing_factor = components[9]


        merge.ranker = 'ac'
        merge.source = source
        merge.theta = theta
        merge.hfunc = hfunc
        merge.group = group
        merge.subgroup = subgroup

        merge.augmentation = augmentation
        merge.val_metric = val_metric
        merge.feature_method = feature_method
        merge.num_features = num_features
        merge.balancing_factor = balancing_factor

    elif fname.startswith('mrfr'):

        if len(components) == 12:
            source = f'{components[5].upper()}_train'
            group = components[7]
            subgroup = '-'
            augmentation = '-'
            val_metric = '-'
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'

        elif len(components) == 13:
            source = f'{components[5].upper()}_train'
            group = components[7]
            subgroup = components[8]
            augmentation = '-'
            val_metric = '-'
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'


        elif len(components) == 15:

            source = f'lambdamart_{components[8]}_{components[7]}_{components[9]}'
            group = components[10]
            subgroup = '-'

            augmentation = components[8]
            val_metric = components[7]
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'
        # ['mrfr', 'train', '90', '10', 'split', 'lm', 'mrfr', 'err', 'aug0.5', 'nofeat', 'doc', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
        # ['mrfr', 'train', '90', '10', 'split', 'lm', 'mrfr', 'err', 'aug0.5', 'nofeat', 'author', 'ind', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']

        elif len(components) == 16:
            source = f'lambdamart_{components[8]}_{components[7]}_{components[9]}'
            group = components[10]
            subgroup = components[11]

            augmentation = components[8]
            val_metric = components[7]
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'

        elif len(components) == 17:
            source = f'lambdamart_{components[8]}_{components[7]}_{components[9]}_{components[10]}_{components[11]}'
            group = components[12]
            subgroup = '-'

            augmentation = components[8]
            val_metric = components[7]
            feature_method = components[9]
            num_features = components[10]
            balancing_factor = components[11]

        elif len(components) == 18:
            source = f'lambdamart_{components[8]}_{components[7]}_{components[9]}_{components[10]}_{components[11]}'
            group = components[12]
            subgroup = components[13]

            augmentation = components[8]
            val_metric = components[7]
            feature_method = components[9]
            num_features = components[10]
            balancing_factor = components[11]

        merge.ranker = 'rfr'
        merge.source = source
        merge.group = group
        merge.subgroup = subgroup
        merge.augmentation = augmentation
        merge.val_metric = val_metric
        merge.feature_method = feature_method
        merge.num_features = num_features
        merge.balancing_factor = balancing_factor

    elif fname.startswith('lambdamart'):
        if len(components) == 15:
            metric = components[7]
            augmentation = '-'
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'
        if len(components) == 16:
            metric = components[8]
            augmentation = components[6]
            feature_method = '-'
            num_features = '-'
            balancing_factor = '-'
        if len(components) == 17:
            metric = components[7]
            augmentation = '-'
            feature_method = components[8]
            num_features = components[9]
            balancing_factor = components[10]
        if len(components) == 18:
            metric = components[8]
            augmentation = components[6]
            feature_method = components[9]
            num_features = components[10]
            balancing_factor = components[11]
        merge.ranker = 'lambdamart'
        merge.val_metric = metric
        merge.augmentation = augmentation
        merge.feature_method = feature_method
        merge.num_features = num_features
        merge.balancing_factor = balancing_factor

    elif fname.startswith('relevance'):
        merge.ranker = 'relevance_ranker'
        merge.source = '_'.join(components[6:7])

    concat.append(merge)

outdf = pd.concat(concat)
# sys.exit()
outdf.to_csv('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/all_experiments.csv', index=False)
