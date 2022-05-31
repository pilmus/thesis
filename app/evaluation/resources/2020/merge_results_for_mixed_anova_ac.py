import glob
import os

import pandas as pd
from tqdm import tqdm

eval_results = '/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/eval_results/'
eel_ac_rfr = glob.glob(os.path.join(eval_results, 'ac*EEL.tsv'))
eeli_ac_rfr = glob.glob(os.path.join(eval_results, 'ac*ERR.tsv'))
util_ac_rfr = glob.glob(os.path.join(eval_results, 'ac*util.tsv'))

eel_pre_source = glob.glob(os.path.join(eval_results, 'lambdamart*EEL.tsv')) + glob.glob(
    os.path.join(eval_results, 'relevance*EEL.tsv'))
eeli_pre_source = glob.glob(os.path.join(eval_results, 'lambdamart*ERR.tsv')) + glob.glob(
    os.path.join(eval_results, 'relevance*ERR.tsv'))
util_pre_source = glob.glob(os.path.join(eval_results, 'lambdamart*util.tsv')) + glob.glob(
    os.path.join(eval_results, 'relevance*util.tsv'))

# 1344it [05:29,  4.08it/s]

dicts = []


def pivot_table(table_path):
    df = pd.read_csv(table_path, sep='\t', names=['key', 'qid', 'value'])
    df = df.pivot(index='qid', columns='key', values='value')
    df = df.reset_index()
    return df


concat = []
for e, ei, u in tqdm(zip(eel_ac_rfr, eeli_ac_rfr, util_ac_rfr)):

    post_source = os.path.splitext(os.path.basename(e))[0]
    components = post_source.split('_')
    if len(components) == 12:
        pre_source = f'{components[3].upper()}_train'
        theta = components[4]
        hscore = components[5]
        group = components[6]
        subgroup = '-'
        pre_rerank_e = next((epre_source for epre_source in eel_pre_source if pre_source in epre_source), None)
        pre_rerank_ei = next((eipre_source for eipre_source in eeli_pre_source if pre_source in eipre_source), None)
        pre_rerank_u = next((upre_source for upre_source in util_pre_source if pre_source in upre_source), None)
        # ['ac', 'controller', 'train10', 'meta', '99', 'linear', 'doc', 'train', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']

    elif len(components) == 13:
        pre_source = f'{components[3].upper()}_train'
        theta = components[4]
        hscore = components[5]
        group = components[6]
        subgroup = components[7]
        pre_rerank_e = next((epre_source for epre_source in eel_pre_source if pre_source in epre_source), None)
        pre_rerank_ei = next((eipre_source for eipre_source in eeli_pre_source if pre_source in eipre_source), None)
        pre_rerank_u = next((upre_source for upre_source in util_pre_source if pre_source in upre_source), None)
        # ['ac', 'controller', 'train10', 'meta', '99', 'linear', 'author', 'ind', 'train', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']

    elif len(components) == 15:
        # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'err', 'aug0.5', 'nofeat', 'doc', '99', 'linear', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
        pre_source = f'lambdamart_mrfr_train_90_10_split_{components[6]}_lm_{components[5]}_{components[7]}'
        theta = components[9]
        hscore = components[10]
        group = components[8]
        subgroup = '-'
        pre_rerank_e = glob.glob(os.path.join(eval_results, pre_source + '*EEL.tsv'))[0]
        pre_rerank_ei = glob.glob(os.path.join(eval_results, pre_source + '*ERR.tsv'))[0]
        pre_rerank_u = glob.glob(os.path.join(eval_results, pre_source + '*util.tsv'))[0]

    elif len(components) == 16:
        # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'err', 'aug0.5', 'nofeat', 'author', 'ind', '99', 'linear', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
        pre_source = f'lambdamart_mrfr_train_90_10_split_{components[6]}_lm_{components[5]}_{components[7]}'
        theta = components[10]
        hscore = components[11]
        group = components[8]
        subgroup = components[9]
        pre_rerank_e = glob.glob(os.path.join(eval_results, pre_source + '*EEL.tsv'))[0]
        pre_rerank_ei = glob.glob(os.path.join(eval_results, pre_source + '*ERR.tsv'))[0]
        pre_rerank_u = glob.glob(os.path.join(eval_results, pre_source + '*util.tsv'))[0]

    elif len(components) == 17:
        # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'ndcg', 'noaug', 'msd', '20', '0.9', 'doc', '9', 'min', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
        pre_source = f'lambdamart_mrfr_train_90_10_split_lm_{components[5]}_{components[7]}_{components[8]}_{components[9]}'
        theta = components[11]
        hscore = components[12]
        group = components[10]
        subgroup = '-'
        pre_rerank_e = glob.glob(os.path.join(eval_results, pre_source + '*EEL.tsv'))[0]
        pre_rerank_ei = glob.glob(os.path.join(eval_results, pre_source + '*ERR.tsv'))[0]
        pre_rerank_u = glob.glob(os.path.join(eval_results, pre_source + '*util.tsv'))[0]

    elif len(components) == 18:
        # ['ac', 'controller', 'train10', 'lm', 'mrfr', 'ndcg', 'noaug', 'msd', '20', '0.9', 'author', 'one', '9', 'min', 'TREC-Fair-Ranking-training-sample', '-10-full-annotations-DocHLevel-mixed', 'group-qrels', 'EEL']
        pre_source = f'lambdamart_mrfr_train_90_10_split_lm_{components[5]}_{components[7]}_{components[8]}_{components[9]}'
        theta = components[12]
        hscore = components[13]
        group = components[10]
        subgroup = components[11]
        pre_rerank_e = glob.glob(os.path.join(eval_results, pre_source + '*EEL.tsv'))[0]
        pre_rerank_ei = glob.glob(os.path.join(eval_results, pre_source + '*ERR.tsv'))[0]
        pre_rerank_u = glob.glob(os.path.join(eval_results, pre_source + '*util.tsv'))[0]

    edf = pivot_table(e)
    eidf = pivot_table(ei).rename(
        {'difference': 'difference_ind', 'disparity': 'disparity_ind', 'relevance': 'relevance_ind'}, axis=1)
    udf = pivot_table(u)

    predf = pivot_table(pre_rerank_e).rename(
        {'difference': 'difference_pre', 'disparity': 'disparity_pre', 'relevance': 'relevance_pre'}, axis=1)
    preidf = pivot_table(pre_rerank_ei).rename(
        {'difference': 'difference_ind_pre', 'disparity': 'disparity_ind_pre', 'relevance': 'relevance_ind_pre'},
        axis=1)
    prudf = pivot_table(pre_rerank_u).rename({'util': 'util_pre'}, axis=1)

    merge = pd.merge(
        pd.merge(pd.merge(pd.merge(pd.merge(edf, eidf, on='qid'), udf, on='qid'), predf, on='qid'), preidf, on='qid'),
        prudf, on='qid')

    merge['pre_source'] = pre_source
    merge['post_source'] = post_source
    merge['theta'] =theta
    merge['hscore'] =hscore
    merge['group'] =group
    merge['subgroup'] =subgroup
    merge['ranker'] = 'ac'
    concat.append(merge)

outdf = pd.concat(concat)
outdf.to_csv('/mnt/c/Users/maaik/Documents/thesis/app/evaluation/resources/2020/ac_for_mixed_anova.csv', index=False)
