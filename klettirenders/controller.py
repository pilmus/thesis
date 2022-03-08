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


def targexp_per_relgrade(N, s, gamma=0.5, u=0.7):
    """
    Compute the expected exposure for relevant and nonrelevant items given the total ranking length and the number of relevant documents in the ranking.

    :param N: The total ranking length.
    :param s: The number of relevant items in the ranking.
    :param gamma: The patience parameter.
    :param u: The utility function. This is set to a single value because it is the value of k*x, where k=0.7 and x is
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
                targexp = (1 - gamma ** s * (1 - u) ** s) / (s(1 - gamma * (1 - u)))
        elif rel == 0:
            if s == N:
                targexp = 0
            else:
                targexp = ((1 - u) ** s * (gamma ** s - gamma ** N)) / ((N - s) * (1 - gamma))
        else:
            raise ValueError(f'Only binary relevance supported.')
        targetExposurePerRelevanceLevel[rel] = targexp
    return targetExposurePerRelevanceLevel


def targexp_document(document, rhos, mus, vs, N): #rhos do not have to be sorted
    rho = rhos[document]

    poibin_params = [v for k, v in rhos.items() if not k == document]
    pb = PoiBin(poibin_params)
    targexp = 0
    for s in range(0, N):
        targexp += pb.pmf(s) * (rho * mus[s + 1] + (1 - rho) * vs[s])
    return targexp


def actexp_document(docid, ranking, rhos):
    pass


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


def controller(sequence, rhos, theta):
    qid = sequence.iloc[0].qid
    sid = sequence.iloc[0].sid
    docs = queries[queries.qid == qid]
    #
    # authors = load_author_list()
    # documents = load_doclist()
    docids = docs.doc_id.drop_duplicates().to_list()
    author_to_doc_mapping, doc_to_author_mapping = sample_files_author_doc_mappings(docids)
    authors = list(author_to_doc_mapping.keys())

    # initialize advantage dict for all producers (id to advantage mapping)
    author_advantages = {author_id: 0 for author_id in authors}

    # initialize target expected exposure dict for all producers
    author_targ_exps = {author_id: 0 for author_id in authors}

    # initialize actual expected exposure dict for all producers
    author_real_exps = {author_id: 0 for author_id in authors}

    # initialize advantage dict for all documents (size = # docs x # iterations in sequence)
    doc_advantages = np.zeros((len(sequence), len(docids)))

    theta_mat = np.matrix([[theta], [1 - theta]])

    for t in range(0, len(sequence)):
        # for each document get the producers, take mean of the producer advantages (def advantage_mean(producer_advs))
        for doc in docids:
            doc_advantages[t][doc] = advantage_mean(doc_to_author_mapping[doc], author_advantages)

        # compute the controller scores (h) --> don't have to save these, are re-computed each iteration (h[i,t] = theta * rho[i] +(1-theta)*Adv[i,t], i is the doc num)
        cscores_term1 = np.vstack((rhos, -doc_advantages[t])).T
        controller_scores = np.dot(cscores_term1, theta_mat)

        # sort documents by computed h-scores, save ranking in later output df

        for author in authors:
            author_targ_exps[author][t] +=
        # foreach producer
        # # update actual expected exposure (AEE[p,t] = AEE[p,t-1] + )
        # # update target expected exposure
        # # update producer advantages ()
        continue

    pass


def producer_advantage(actexp, targexp):
    expdiff = actexp - targexp
    return (expdiff ** 2) * math.copysign(1, expdiff)



def naive_controller(sequence_id, producers, documents, doc_to_producer_mapping, producer_to_document_mapping, rhos,
                     theta):
    producer_advantages = {producer: [] for producer in producers}

    document_advantages = {document: [] for document in documents}

    producer_actual_expected_exposure = {producer: [] for producer in producers}
    producer_target_expected_exposure = {producer: [] for producer in producers}

    for producer in producers:
        producer_advantages[producer][0] = 0
        producer_actual_expected_exposure[producer][0] = 0
        producer_target_expected_exposure[producer][0] = 0

    sequence_pd = pd.DataFrame(columns=['q_num', 'document', 'score', 'rank'])

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

        sequence_pd = sequence_pd.append(hscore_df)
        for producer in producers:
            prod_docs = producer_to_document_mapping[producer]
            overlaplist = [doc for doc in prod_docs if doc in documents]
            new_actual_exp = 0
            new_target_exp = 0
            for doc in overlaplist:
                new_actual_exp += actualexp_document()
                new_target_exp += targexp_document()

            producer_actual_expected_exposure[producer][t] = producer_actual_expected_exposure[t - 1] + new_actual_exp
            producer_target_expected_exposure[producer][t] = producer_target_expected_exposure[t - 1] + new_target_exp

            producer_advantages[producer][t + 1] = producer_advantages[producer][t] + [
                producer_advantage(producer_actual_expected_exposure[producer][t],
                                   producer_target_expected_exposure[producer][t])]


if __name__ == '__main__':
    # queries = concat_sample_files()
    seq = 'training/2020/training-sequence-full.tsv'
    q = 'training/2020/TREC-Fair-Ranking-training-sample.json'
    ioh = InputOutputHandler(seq, q)

    qseq = ioh.get_query_seq()

    controller(qseq[qseq.sid == 0])
