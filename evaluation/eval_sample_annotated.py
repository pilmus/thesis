import pandas as pd

merged_annot_file = 'resources/evaluation/2020/merged-annotations.json'
doc_annot_file = 'resources/evaluation/2020/doc-annotations.csv'
eval_file = 'resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json'

merged_annot_df = pd.read_json(merged_annot_file, lines=True)
merged_annot_df = merged_annot_df.explode('authors')
merged_annot_df['econ_group'] = merged_annot_df.authors.apply(lambda row: row.get('level'))

doc_annot_df = pd.read_csv(doc_annot_file)

eval_df = pd.read_json(eval_file, lines=True)
eval_df = eval_df.explode('documents')
eval_df['doc_id'] = eval_df.documents.apply(lambda row: row['doc_id'])

eval_docs = set(eval_df.doc_id.tolist())
annot_docs = set(merged_annot_df.id.tolist())

annotated_eval_df = pd.merge(eval_df, doc_annot_df, left_on='doc_id', right_on='id', how='left')

annotated_eval_df['group'] = annotated_eval_df.DocLevel.map({'Developing': '1', 'Mixed': '1|2', 'Advanced': '2'})

annotated_eval_df = annotated_eval_df.dropna()

annotated_eval_df = annotated_eval_df[['doc_id', 'group']]

annotated_eval_df.to_csv('resources/evaluation/2020/TREC-Fair-Ranking-eval-sample-groups.csv', index=False)

eval_df
