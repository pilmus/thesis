import pandas as pd


def load_features():
    pass


def load_queries():
    pass


def extract_qid_query_docid(file):
    df = pd.read_json(file, lines=True)
    df = df.explode('documents')
    df['doc_id'] = df.apply(lambda row: row.documents.get('doc_id'), axis=1)
    df = df[['qid', 'query', 'doc_id']]
    return df




print(extract_qid_query_docid("resources/evaluation/2019/TREC-Competition-eval-sample-with-rel.json"))
