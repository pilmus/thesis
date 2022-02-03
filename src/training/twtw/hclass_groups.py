from collections import Counter

import pandas as pd
from tqdm import tqdm

paper_authors_file = 'resources/evaluation/2020/corpus-subset-for-queries.paper_authors.csv'
authors_file = 'resources/evaluation/2020/corpus-subset-for-queries.authors.csv'

paper_authors = pd.read_csv(paper_authors_file, dtype={'paper_sha': str, 'corpus_author_id': str, 'position': int})
authors = pd.read_csv(authors_file, dtype={'corpus_author_id': str, 'h_class': str})

paper_author_hclass = pd.merge(paper_authors, authors, how='left', on='corpus_author_id')


def merge_hclass(df, mode):
    h_classes = df.h_class.to_list()
    c = Counter(h_classes)
    if mode == 'all_low':
        if c['L'] > 0:
            df['group'] = '0'
        else:
            df['group'] = '1'
    elif mode == 'all_high':
        if c['H'] > 0:
            df['group'] = '1'
        else:
            df['group'] = '0'
    elif mode == 'mixed':
        if c['L'] > 0 and c['H'] > 0:
            df['group'] = '2'
        elif c['L'] > 0:
            df['group'] = '0'
        else:
            df['group'] = '1'
    elif mode == 'majorityL':
        if c['L'] >= c['H']:
            df['group'] = '0'
        else:
            df['group'] = '1'
    elif mode == 'majorityH':
        if c['L'] > c['H']:
            df['group'] = '0'
        else:
            df['group'] = '1'
    else:
        raise ValueError(f'Invalid mode {mode}.')
    return df

modes = ['all_low','all_high','mixed','majorityL','majorityH']

for mode in tqdm(modes):
    paper_author_group = paper_author_hclass.groupby('paper_sha').apply(merge_hclass, mode)
    paper_author_group = paper_author_group[['paper_sha', 'group']]
    paper_author_group = paper_author_group.drop_duplicates()
    paper_author_group = paper_author_group.rename({'paper_sha': 'doc_id'}, axis=1)
    paper_author_group.to_csv(f'resources/training/2020/doc-annotations-hclass-groups-{mode}.csv', index=False)
