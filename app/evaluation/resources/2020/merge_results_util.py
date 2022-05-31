import glob
import os

import pandas as pd
from tqdm import tqdm

eel = glob.glob('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/*util.tsv')

factor_names = ['ranker', 'source', 'group', 'subgroup', 'theta', 'hfunc', 'augmentation', 'val_metric',
                'feature_method', 'num_features', 'balancing_factor']

dicts = []
for f in tqdm(eel):

    df = pd.read_csv(f, sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    adddict = {'util_mean': df.util.mean()}
    for factor in factor_names:
        adddict[factor] = None
    fname = os.path.basename(f)
    fname = fname.replace('ac_controller_train10', 'ac')
    fname = fname.replace(
        '_TREC-Fair-Ranking-training-sample_-10-full-annotations-DocHLevel-mixed_group-qrels_util.tsv',
        '')
    adddict['ranker_config'] = fname
    if fname.startswith('ac'):
        adddict['ranker'] = 'ac'
        fname = fname.replace('_train', '')
        components = fname.split('_')
        if '_text_' in fname or '_meta_' in fname:
            # if '_train' in fname:
            adddict['source'] = f"{components[1].upper()}_train"
            # elif '_test' in fname:
            #     adddict['source'] = f"{components[1].upper()}_eval"
            adddict['theta'] = components[2]
            adddict['hfunc'] = components[3]
            adddict['group'] = components[4]
            if 'author' in fname:
                adddict['subgroup'] = components[-1]

        else:
            adddict['hfunc'] = components[-1]
            adddict['theta'] = components[-2]
            if '_doc_' in fname:
                adddict['group'] = components[-3]
                adddict['source'] = '_'.join(components[1:-3])
            else:
                adddict['subgroup'] = components[-3]
                adddict['group'] = components[-4]
                adddict['source'] = '_'.join(components[1:-4])
    elif fname.startswith('lambdamart'):
        adddict['ranker'] = 'lambdamart'
        fname = fname.replace('lambdamart_mrfr_train_90_10_split', 'lambdamart')
        fname = fname.replace('_random_state=0', '')
        fname = fname.replace('_lm', '')
        components = fname.split('_')
        if 'aug' in fname:
            adddict['augmentation'] = components[1]
            adddict['val_metric'] = components[2]
        else:
            adddict['val_metric'] = components[1]
        if 'msd' in fname or 'mpt' in fname:
            adddict['feature_method'] = components[-3]
            adddict['num_features'] = components[-2]
            adddict['balancing_factor'] = components[-1]
    elif fname.startswith('mrfr'):
        adddict['ranker'] = 'mrfr'
        fname = fname.replace('_train_90_10_split', '')
        # fname = fname.replace('_train','')
        components = fname.split('_')

        if 'lm_mrfr' in fname:
            if '_doc' in fname:
                adddict['group'] = components[-1]
                adddict['source'] = '_'.join(components[1:-1])
            else:
                adddict['group'] = components[-2]
                adddict['subgroup'] = components[-1]
                adddict['source'] = '_'.join(components[1:-2])
        else:
            adddict['source'] = '_'.join(components[1:3])

            if '_doc' in fname:
                adddict['group'] = components[-1]
            else:
                adddict['group'] = components[-2]
                adddict['subgroup'] = components[-1]
    elif fname.startswith('relevance'):
        fname = fname.replace("_train_90_10_split", '')
        adddict['ranker'] = 'relevance_ranker'
        components = fname.split('_')
        adddict['source'] = '_'.join(components[2:])
    dicts.append(adddict)


df = pd.DataFrame(dicts)
df.to_csv('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/all_experiments_util.csv', index=False)
