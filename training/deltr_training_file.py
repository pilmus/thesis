import pandas as pd

train = pd.read_json('resources/training/2020/TREC-Fair-Ranking-training-sample.json', lines=True)
groups = pd.read_csv('resources/training/2020/doc-annotations-hclass-groups.csv')


train = train.explode('documents')
train['doc_id'] = train.documents.apply(lambda row: row['doc_id'])

merged = pd.merge(train, groups, how='left', on='doc_id')
merged = merged.groupby('qid').filter(lambda g: not g.group.isnull().values.any())
merged = merged[['qid', 'query', 'frequency', 'documents']]
merged = merged.groupby(['qid', 'query', 'frequency']).documents.apply(list).reset_index()
merged.to_json('resources/training/2020/DELTR-training-sample.json', orient='records',lines=True)

