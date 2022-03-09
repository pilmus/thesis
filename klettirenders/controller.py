import json
import math
import sys

import numpy as np
import pandas as pd
from attrdict import AttrDict
from elasticsearch import Elasticsearch
from tqdm import tqdm

from interface.iohandler import InputOutputHandler, IOHandlerKR
from klettirenders.poibin.poibin import PoiBin


def mus_vs(N, gamma=0.5, u=0.7):
    mus = []
    vs = []
    # loop runs until N inclusively because we want values for if 0 - N documents are relevant,
    # so for example: 1,2,3,4,5 --> 0 rel, 1 rel, 2 rel, 3 rel, 4 rel, 5 rel == 6 mus
    for s in range(0, N + 1):
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
                targexp = (1 - gamma ** s * (1 - k) ** s) / (s * (1 - gamma * (1 - k)))
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

    actexp = gamma ** (rankpos)
    for i in range(0, rankpos):
        doc_at_rank = rankdf[rankdf['rank'] == i + 1].iloc[0]['document']
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

def advantage_mean(authors, author_advantages, t):
    return sum([author_advantages[author][t] for author in authors]) / len(authors)


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


    targexp_documents = {}
    for doc in tqdm(documents):
        targexp_documents[doc] = targexp_document(doc,rhos,mus,vs,N)


    for producer in producers:
        producer_advantages[producer] = [0] * 150
        producer_actual_expected_exposure[producer] = [0] * 150
        producer_target_expected_exposure[producer] = [0] * 150

    sequence_df = pd.DataFrame(columns=['q_num', 'document', 'score', 'rank'])

    for t in tqdm(range(0, 150)):
        for document in documents:
            doc_producers = doc_to_producer_mapping[document]
            document_advantages[document] = advantage_mean(doc_producers, producer_advantages, t)

        hscores = {}
        for document in documents:
            hscores[document] = theta * rhos[document] - (1 - theta) * document_advantages[document]

        hscore_df = pd.DataFrame({'q_num': t, 'document': list(hscores.keys()), 'score': list(hscores.values())})
        tqdm.pandas()
        hscore_df['rank'] = hscore_df.score.rank(method='first', ascending=False)
        hscore_df = hscore_df.astype({'rank': int})

        sequence_df = sequence_df.append(hscore_df)
        for producer in producers:
            prod_docs = producer_to_doc_mapping[producer]
            # overlaplist = [doc for doc in prod_docs if doc in documents]
            new_actual_exp = 0
            new_target_exp = 0
            for doc in prod_docs:
                new_actual_exp += actexp_document(doc, hscore_df, rhos, gamma, k)
                new_target_exp += targexp_documents[doc]

            producer_actual_expected_exposure[producer][t] = new_actual_exp
            producer_target_expected_exposure[producer][t] = new_target_exp

            producer_advantages[producer][t + 1] = producer_advantages[producer][t] + producer_advantage(
                producer_actual_expected_exposure[producer][t], producer_target_expected_exposure[producer][t])
    return sequence_df


def res_to_doc_to_author_mapping(res):
    hits = res['hits']['hits']
    doc_to_author_mapping = {hit['_id']: hit['_source']['author_ids'] for hit in hits}
    return doc_to_author_mapping


def doc_to_author_mapping_from_es(docids, outfile):
    es = Elasticsearch(timeout=500)
    res = es.search(index='semanticscholar2020og', body={'size': len(docids), 'query': {'ids': {'values': docids}}})
    mapping = res_to_doc_to_author_mapping(res)
    with open(outfile, 'w') as fp:
        json.dump(mapping, fp)
    return mapping


def get_doc_to_author_mapping(docids, mapping_file):
    with open(mapping_file) as fp:
        mapping = json.load(fp)
    filtered_mapping = {k: v for k, v in mapping.items() if k in docids}
    return filtered_mapping


def author_doc_mapping(doc_author_mapping):
    author_doc_mapping = {}
    for docid, authorlist in doc_author_mapping.items():
        for author in authorlist:
            if author not in author_doc_mapping:
                author_doc_mapping[author] = []
            author_doc_mapping[author] = author_doc_mapping[author] + [docid]

    return author_doc_mapping


if __name__ == '__main__':
    # queries = concat_sample_files()
    seq = 'training/2020/training-sequence-full.tsv'
    q = 'training/2020/TREC-Fair-Ranking-training-sample.json'
    ioh = IOHandlerKR(seq, q)

    estimated_relevances = pd.read_csv('klettirenders/relevances/Training_rel_scores_model_A.csv')
    qseq_with_relevances = pd.merge(ioh.get_query_seq(), estimated_relevances, on=['qid', 'doc_id', 'relevance'],
                                    how='inner').sort_values(by=['sid', 'q_num']).reset_index(drop=True)

    docids = qseq_with_relevances.doc_id.drop_duplicates().to_list()
    doc_to_author_mapping = get_doc_to_author_mapping(docids, 'klettirenders/mappings/training_doc_to_author.json')

    # doc_to_author_mapping_from_es(docids,'klettirenders/mappings/training_doc_to_author.json')

    subdf = qseq_with_relevances[qseq_with_relevances.sid == 0]
    subdocids = subdf.doc_id.to_list()
    filtered_d2a = {k: v for k, v in doc_to_author_mapping.items() if k in subdocids}
    author_to_doc_mapping = author_doc_mapping(filtered_d2a)

    rhos_df = subdf[['doc_id', 'est_relevance']].drop_duplicates()
    rhos = dict(zip(rhos_df.doc_id, rhos_df.est_relevance))

    naive_controller(rhos, filtered_d2a, author_to_doc_mapping)
