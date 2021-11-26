import jsonlines
import pandas as pd


#
# subcorpus_ids = 'corpus2020/TREC-Fair-Ranking-subcorpus-docids.txt'
# evaltrain_ids = 'corpus2020/TREC-Fair-Ranking-docids.txt'
#
# with open(subcorpus_ids,'r') as f:
#     subcorpus_ids = set(f.read().splitlines())
#
# with open(evaltrain_ids, 'r') as f:
#     evaltrain_ids = set(f.read().splitlines())

def sample_to_df(filename):
    df = pd.read_json(filename, lines=True)
    df = df[['documents']]
    df = df.explode('documents')
    df['documents'] = df['documents'].transform(lambda x: x['doc_id'])
    return df


eval_file = 'resources/corpus2020/TREC-Fair-Ranking-eval-sample.json'
train_file = 'resources/corpus2020/TREC-Fair-Ranking-training-sample.json'
corp_file = 'resources/corpus2020/corpus-subset-for-queries.papers.csv'

corp_df = pd.read_csv(corp_file)
corp_df = corp_df[['paper_sha']]
corp_df = corp_df.rename(columns={"paper_sha": "documents"})

print(corp_df.columns)

eva_df = sample_to_df(eval_file)
tra_df = sample_to_df(train_file)
all_df = pd.concat([eva_df, tra_df])
all_df = all_df.drop_duplicates()


merger = pd.merge(all_df, corp_df, how='inner', on=['documents'])

print(len(merger))


def extract_docs(reader):
    for documents in reader.iter(type=dict, skip_invalid=True):

        yield [doc['doc_id'] for doc in documents['documents']]


def extract_docids(filename):
    with open(filename, 'r') as f:
        reader = jsonlines.Reader(f)
        docgen = extract_docs(reader)
        docids = []
        for doc in docgen:
            docids.extend(doc)
    return docids


evadocids = extract_docids(eval_file)
tradocids = extract_docids(train_file)
evatra_docids = set(evadocids + tradocids)

