import pandas as pd

eval_file = 'resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json'
doc_annotations = "resources/evaluation/2020/doc-annotations.csv"

eval = pd.read_json(eval_file, lines=True)
eval = eval.explode('documents')
eval[['doc_id','relevance']] = eval.documents.apply(pd.Series)
eval['docids'] = eval['documents'].apply(lambda row: row['doc_id'])


docs = pd.read_csv(doc_annotations)

di = {'Advanced' : '2', 'Developing' : '1', 'Mixed': '1|2'}

docs['group'] = docs.DocLevel.map(di)

merged = pd.merge(eval,docs, how='left', left_on = 'doc_id', right_on='id')


merged = merged[['doc_id', 'group']]

outfile = 'resources/evaluation/2020/doc-annotations-groups.csv'
merged.to_csv(outfile, index=False)