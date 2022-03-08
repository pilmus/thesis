"""
It has come to our attention that the training query set includes document IDs that cannot be resolved in the OpenCorpus download.
We therefore recommend removing such IDs.
There is a two-step procedure you can follow to remove such documents:

    Remove all documents from query 'documents' sets that do not exist in OpenCorpus
    Drop all queries that, after this document removal, have fewer than 5 documents

This will result in dropping approximately 100 training queries.
"""
from interface.corpus import Corpus
from interface.iohandler import InputOutputHandler
from utils import io

idx_name = 'semanticscholar2020og'
fseq = "training/2020/training-sequence-10.tsv"
fq = 'training/2020/TREC-Fair-Ranking-training-sample.json'
outfile = 'training/2020/TREC-Fair-Ranking-training-sample-cleaned.json'

corpus = Corpus(idx_name)
input = InputOutputHandler(corpus, fquery=fq, fsequence=fseq)

queries = input.get_queries()

doc_ids = queries.doc_id.drop_duplicates().to_list()
ids_available = corpus.get_docs_by_ids(doc_ids).doc_id.drop_duplicates().to_list()

ids_missing = [id for id in doc_ids if id not in ids_available]

queries = queries.loc[queries.doc_id.isin(ids_available)]

result_size = queries.groupby('qid').doc_id.count()

queries_remove = result_size[result_size < 5].keys().to_list()

queries_raw = io.read_jsonlines(fq)

queries_raw = [query for query in queries_raw if query['qid'] not in queries_remove]

io.write_jsonlines(queries_raw, outfile)
