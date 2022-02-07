import glob
import os

import pandas as pd

basename = 'resources/evaluation/2020/eval_output'
#
# reference_files = ['Deltr-gammas-ferraro-mixed_group-unnormalized.tsv', 'Deltr-gammas-ferraro-mixed_group-unnormalized-sqrt.tsv']
# reference_files = [os.path.join(basename, 'param_investigtation', ref_file) for ref_file in reference_files]

ref_nosqrt = os.path.join(basename, 'param_investigation', 'Deltr-gammas-ferraro-mixed_group-unnormalized.tsv')
ref_sqrt = os.path.join(basename, 'param_investigation', 'Deltr-gammas-ferraro-mixed_group-unnormalized-sqrt.tsv')

eval_files = glob.glob(os.path.join(basename, 'deltr_gammas', '*'))

#
# eval_files = [os.path.join(basename,'deltr_gammas_std_True-full-iter-5-relweight-0.5-unnormalized-mixed_group.tsv'),
#               os.path.join(basename,'deltr_gammas_std_True-full-iter-5-relweight-0.5-unnormalized-sqrt-mixed_group.tsv'),
#               os.path.join(basename, 'trec_run','trec_run.Deltr-gammas-mixed_group-unnormalized.tsv'),
#               os.path.join(basename, 'param_investigation','Deltr-gammas-ferraro-mix_up-unnormalized-sqrt.tsv'),]

df_ref = pd.read_csv(ref_nosqrt, sep='\t', names=['key', 'qid', 'value'])
df_ref = df_ref.pivot(index='qid', columns='key', values='value')
ref_mean = df_ref.difference.mean()

df_ref_s = pd.read_csv(ref_sqrt, sep='\t', names=['key', 'qid', 'value'])
df_ref_s = df_ref_s.pivot(index='qid', columns='key', values='value')
ref_mean_s = df_ref_s.difference.mean()

comp_dicts = []
comp_dicts_s = []
comp_dicts.append(
    {'mean': round(ref_mean, 3),
     'mean-refmean': round(ref_mean - ref_mean, 3),
     'refmean': round(ref_mean, 3),
     'alpha':None,
     'corpus': None,
     'group': None,
     'file': os.path.basename(ref_nosqrt)})
comp_dicts_s.append(
    {'mean': round(ref_mean_s, 5),
     'mean-refmean_s': round(ref_mean_s - ref_mean_s, 5),
     'refmean_s': round(ref_mean_s, 5),
     'alpha':None,
     'corpus': None,
     'group': None,
     'file': os.path.basename(ref_sqrt)})

for eval in eval_files:
    df = pd.read_csv(eval, sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    df_mean = round(df.difference.mean(), 3)
    # print(f"{round(df.difference.mean(),3)}\t")
    # print(f"{df_mean}\t{df_mean - ref_mean}\t{df_mean - ref_mean_s}\t{ref_mean}\t{ref_mean_s}\t{eval}")
    splits = eval.split('-')
    alpha = float(splits[2])
    corp = splits[4]
    group = splits[6].split('.')[0]
    if 'squared' in eval:
        comp_dicts_s.append(
            {'mean': round(df_mean, 5),
             'mean-refmean_s': round(df_mean - ref_mean_s, 5),
             'refmean_s': round(ref_mean_s, 5),
             'alpha': alpha,
             'corpus': corp,
             'group': group,
             'file': os.path.basename(eval)})

    else:
        comp_dicts.append(
            {'mean': round(df_mean, 3),
             'mean-refmean': round(df_mean - ref_mean, 3),
             'refmean': round(ref_mean, 3),
             'alpha': alpha,
             'corpus': corp,
             'group': group,
             'file': os.path.basename(eval)})

df_comp = pd.DataFrame(comp_dicts)
df_comp_s = pd.DataFrame(comp_dicts_s)
df_comp.to_csv('mean_comp_deltr_gammas.csv')
df_comp_s.to_csv('mean_comp_sq_deltr_gammas.csv')
