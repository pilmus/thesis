"""
| level                 | feature                                    |
| --------------------- | ------------------------------------------ |
| query-document (BM25) | title                                      |
|                       | abstract                                   |
|                       | entities                                   |
|                       | venue                                      |
|                       | journal                                    |
|                       | fields of study                            |
|                       | author's names                             |
| document              | min h-index of paper's authors             |
|                       | avg h-index of paper's authors             |
|                       | max h-index of paper's authors             |
|                       | min number of papers of paper's authors    |
|                       | avg number of papers of paper's authors    |
|                       | max number of papers of paper's authors    |
|                       | min i10-index of paper's authors           |
|                       | avg i10-index of paper's authors           |
|                       | max i10-index of paper's authors           |
|                       | min number of citations of paper's authors |
|                       | avg number of citations of paper's authors |
|                       | max number of citations of paper's authors |
|                       | number of in-citations to paper            |
|                       | number of out-citations from the paper     |
|                       | number of authors with class L             |
|                       | number of authors with class H             |


"""
from collections import Counter

import pandas as pd
from tqdm import tqdm

from ferraro import load_data

authors_file = "resources/evaluation/2020/corpus-subset-for-queries.authors.csv"
papers_file = "resources/evaluation/2020/corpus-subset-for-queries.papers.csv"
authors_papers_file = "resources/evaluation/2020/corpus-subset-for-queries.paper_authors.csv"



def extract_metadata(df):
    df['min_hindex'] = df.h_index.min()
    df['avg_hindex'] = df.h_index.mean()
    df['max_hindex'] = df.h_index.max()
    df['min_authorpapers'] = df.num_papers.min()
    df['avg_authorpapers'] = df.num_papers.mean()
    df['max_authorpapers'] = df.num_papers.max()
    df['min_i10index'] = df.i10.min()
    df['avg_i10index'] = df.i10.mean()
    df['max_i10index'] = df.i10.max()
    df['min_numcitations'] = df.num_citations.min()
    df['avg_numcitations'] = df.num_citations.mean()
    df['max_numcitations'] = df.num_citations.max()

    hclasses = Counter(df.h_class.to_list())
    df['num_hclass_L'] = hclasses['L']
    df['num_hclass_H'] = hclasses['H']

    return df[['paper_sha',
               'min_hindex',
               'avg_hindex',
               'max_hindex',
               'min_authorpapers',
               'avg_authorpapers',
               'max_authorpapers',
               'min_i10index',
               'avg_i10index',
               'max_i10index',
               'min_numcitations',
               'avg_numcitations',
               'max_numcitations',
               'num_hclass_L',
               'num_hclass_H']]


p = pd.read_csv(papers_file)
ap = pd.read_csv(authors_papers_file, dtype={'corpus_author_id':str})
a = pd.read_csv(authors_file, dtype={'corpus_author_id':str})
m = pd.merge(p,pd.merge(ap,a,how='left'),how='left')



tqdm.pandas()
meta = m.groupby('paper_sha').progress_apply(extract_metadata)
meta = meta.drop_duplicates()
meta.to_csv('resources/features/metadata_sayed.csv',index=False)