import jsonlines
import gzip

subcorpus = 'corpus2020/corpus-subset-for-queries.jsonl.gz'

subcorpus_ids = []
with gzip.open(subcorpus) as f:
    reader = jsonlines.Reader(f)
    # docgen = doc_generator(reader)
    for doc in reader:
        subcorpus_ids.append(doc['id'])

with open('resources/corpus2020/TREC-Fair-Ranking-subcorpus-docids.txt', 'w') as f:
    f.write('\n'.join(subcorpus_ids))