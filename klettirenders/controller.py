import math
import sys

import numpy as np
import pandas as pd
from attrdict import AttrDict
from elasticsearch import Elasticsearch
from tqdm import tqdm

from interface.iohandler import InputOutputHandler
from klettirenders.poibin.poibin import PoiBin


def mus_vs(N, gamma=0.5, u=0.7):
    mus = []
    vs = []
    for s in range(0, N):
        targexp = targexp_per_relgrade(N, s, gamma, u)
        mus = mus + [targexp[1]]
        vs = vs + [targexp[0]]
    return mus, vs


def f(x, k=0.7):
    return k * x


def targexp_per_relgrade(N, s, gamma=0.5, k=0.7):
    """
    Compute the expected exposure for relevant and nonrelevant items given the total ranking length and the number of relevant documents in the ranking.

    :param N: The total ranking length.
    :param s: The number of relevant items in the ranking.
    :param gamma: The patience parameter.
    :param k: The utility function. This is set to a single value because it is the value of k*x, where k=0.7 and x is
    the relevance grade. Since we are working with binary relevance, this value will be either 0.7 or 0. In all
    situations where it would be 0, the expression it is in reduces to "1" in a multiplication, so it cancels out.

    :return: a dict mapping the relevance levels (1 and 0) to the expected exposure for a document with that relevance
    grade
    """
    targetExposurePerRelevanceLevel = {}
    for rel in [1, 0]:

        if rel == 1:
            if s == 0:
                targexp = 0
            else:
                targexp = (1 - gamma ** s * (1 - k) ** s) / (s(1 - gamma * (1 - k)))
        elif rel == 0:
            if s == N:
                targexp = 0
            else:
                targexp = ((1 - k) ** s * (gamma ** s - gamma ** N)) / ((N - s) * (1 - gamma))
        else:
            raise ValueError(f'Only binary relevance supported.')
        targetExposurePerRelevanceLevel[rel] = targexp
    return targetExposurePerRelevanceLevel


def actexp_document(docid, rankdf, rhos, gamma=0.5, k=0.7):
    rankpos = rankdf[rankdf.document == docid].iloc[0]['rank']

    actexp = gamma ** (rankpos - 1)
    for i in range(1, rankpos + 1):
        doc_at_rank = rankdf[rankdf.rank == i].iloc[0]['document']
        rho_at_rank = rhos[doc_at_rank]
        actexp *= (1 - f(rho_at_rank, k=k))

    return actexp


def targexp_document(document, rhos, mus, vs, N):  # rhos do not have to be sorted
    rho = rhos[document]

    poibin_params = [v for k, v in rhos.items() if not k == document]
    pb = PoiBin(poibin_params)
    targexp = 0
    for s in range(0, N):
        targexp += pb.pmf(s) * (rho * mus[s + 1] + (1 - rho) * vs[s])
    return targexp


#
# def load_author_list(pa_file):
#     pa = pd.read_csv(pa_file)
#     authors = pa.corpus_author_id.drop_duplicates().to_list()
#     return authors
#
#
# def concat_sample_files():
#     ts = pd.read_json('training/2020/TREC-Fair-Ranking-training-sample.json', lines=True).explode('documents')
#     ts['doc_id'] = ts.documents.apply(lambda row: row.get('doc_id'))
#     es = pd.read_json('evaluation/2020/TREC-Fair-Ranking-eval-sample.json', lines=True).explode('documents')
#     es['doc_id'] = es.documents.apply(lambda row: row.get('doc_id'))
#     ests = pd.concat([es, ts])
#     return ests
#
#
# def sample_files_doc_ids():
#     ests = concat_sample_files()
#     return ests.doc_id.drop_duplicates().to_list()
#
#
# def mappings_from_res(res):
#     res = AttrDict(res)
#     hits = res.hits.hits
#     author_to_doc_mapping = {}
#     doc_to_author_mapping = {}
#     for hit in hits:
#         source = hit['_source']
#         docid = hit.get('_id')
#         authors = source.get('author_ids')
#         doc_to_author_mapping[docid] = authors
#         for author in authors:
#             if author not in author_to_doc_mapping:
#                 author_to_doc_mapping[author] = []
#             print(author_to_doc_mapping)
#             author_to_doc_mapping[author] = author_to_doc_mapping[author] + [docid]
#     return author_to_doc_mapping, doc_to_author_mapping
#
#
# def sample_files_author_doc_mappings(ids):
#     es = Elasticsearch(timeout=500)
#     res = es.search(index='semanticscholar2020og', body={'size': len(ids), 'query': {'ids': {'values': ids}}})
#     return mappings_from_res(res)

def advantage_mean(authors, author_advantages):
    return sum([author_advantages[author] for author in authors]) / len(authors)


def producer_advantage(actexp, targexp):
    expdiff = actexp - targexp
    return (expdiff ** 2) * math.copysign(1, expdiff)


def naive_controller(rhos, doc_to_producer_mapping, producer_to_doc_mapping,
                     theta=0.9, gamma=0.5, k=0.7):
    producers = list(producer_to_doc_mapping.keys())
    documents = list(doc_to_producer_mapping.keys())

    N = len(documents)
    mus, vs = mus_vs(N)

    producer_advantages = {producer: [] for producer in producers}
    document_advantages = {document: [] for document in documents}

    producer_actual_expected_exposure = {producer: [] for producer in producers}
    producer_target_expected_exposure = {producer: [] for producer in producers}

    for producer in producers:
        producer_advantages[producer][0] = 0
        producer_actual_expected_exposure[producer][0] = 0
        producer_target_expected_exposure[producer][0] = 0

    sequence_df = pd.DataFrame(columns=['q_num', 'document', 'score', 'rank'])

    for t in range(1, 151):
        for document in documents:
            doc_producers = doc_to_producer_mapping[document]
            document_advantages[document] = advantage_mean(doc_producers, producer_advantages)

        hscores = {}
        for document in documents:
            hscores[document] = theta * rhos[document] - (1 - theta) * document_advantages[document]

        hscore_df = pd.DataFrame({'q_num': t - 1, 'document': list(hscores.keys()), 'score': list(hscores.values())})
        tqdm.pandas()
        hscore_df['rank'] = hscore_df.score.progress_apply(pd.Series.rank, ascending=False,
                                                           method='first')

        sequence_df = sequence_df.append(hscore_df)
        for producer in producers:
            prod_docs = producer_to_doc_mapping[producer]
            overlaplist = [doc for doc in prod_docs if doc in documents]
            new_actual_exp = 0
            new_target_exp = 0
            for doc in overlaplist:
                new_actual_exp += actexp_document(doc, hscore_df, rhos, gamma, k)
                new_target_exp += targexp_document(doc, rhos, mus, vs, N)

            producer_actual_expected_exposure[producer][t] = producer_actual_expected_exposure[t - 1] + new_actual_exp
            producer_target_expected_exposure[producer][t] = producer_target_expected_exposure[t - 1] + new_target_exp

            producer_advantages[producer][t + 1] = producer_advantages[producer][t] + [
                producer_advantage(producer_actual_expected_exposure[producer][t],
                                   producer_target_expected_exposure[producer][t])]
    return sequence_df


if __name__ == '__main__':
    # queries = concat_sample_files()
    seq = 'training/2020/training-sequence-full.tsv'
    q = 'training/2020/TREC-Fair-Ranking-training-sample.json'
    ioh = InputOutputHandler(seq, q)

    qseq = ioh.get_query_seq()

    naive_controller(qseq[qseq.sid == 0])
