import pandas as pd

train_file = 'resources/training/2020/TREC-Fair-Ranking-training-sample.json'
paper_authors_file = 'resources/evaluation/2020/corpus-subset-for-queries.paper_authors.csv'
authors_file = 'resources/evaluation/2020/corpus-subset-for-queries.authors.csv'

train = pd.read_json(train_file, lines=True)
paper_authors = pd.read_csv(paper_authors_file, dtype={'paper_sha': str, 'corpus_author_id': str, 'position': int})
authors = pd.read_csv(authors_file, dtype={'corpus_author_id': str, 'h_class': str})

authors = authors[['corpus_author_id', 'h_class']]

paper_author_hclass = pd.merge(paper_authors, authors, how='left', on='corpus_author_id')


def merge_hclass(df):
    h_classes = df.h_class.to_list()
    if 'L' in h_classes:
        df['merged_h_class'] = 'L'
    else:
        df['merged_h_class'] = 'H'
    return df


paper_author_hclass = paper_author_hclass.groupby('paper_sha').apply(merge_hclass)
paper_author_hclass = paper_author_hclass[['paper_sha', 'merged_h_class']]
paper_author_hclass = paper_author_hclass.drop_duplicates()

train = train.explode('documents')
train['paper_sha'] = train.documents.apply(lambda row: row['doc_id'])

merged = pd.merge(train, paper_author_hclass, how='left', on='paper_sha')
merged = merged.groupby('qid').filter(lambda df: not df.merged_h_class.isnull().values.any())

paper_sha_to_hclass = merged[['paper_sha', 'merged_h_class']]
paper_sha_to_hclass = paper_sha_to_hclass.rename({'paper_sha': 'doc_id', 'merged_h_class': 'h_class'}, axis=1)

paper_sha_to_hclass.to_csv('resources/training/2020/doc-hclass.csv', index=False)

merged = merged[['qid', 'query', 'frequency', 'documents']]
merged = merged.groupby(['qid', 'query', 'frequency']).documents.apply(list).reset_index()
merged.to_json('resources/training/2020/DELTR-training-sample.json', orient='records',lines=True)

