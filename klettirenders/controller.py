import json
import math
import sys
import time
from functools import reduce

import numpy as np
import pandas as pd
from attrdict import AttrDict
from elasticsearch import Elasticsearch
from tqdm import tqdm

from interface.iohandler import InputOutputHandler, IOHandlerKR
from klettirenders.poibin.poibin import PoiBin


# def mus_vs(N, gamma=0.5, u=0.7):
#     mus = []
#     vs = []
#     # loop runs until N inclusively because we want values for if 0 - N documents are relevant,
#     # so for example: 1,2,3,4,5 --> 0 rel, 1 rel, 2 rel, 3 rel, 4 rel, 5 rel == 6 mus
#     for s in range(0, N + 1):
#         targexp_rel, targexp_irrel = targexp_per_relgrade(N, s, gamma, u)
#         mus = mus + [targexp_rel]
#         vs = vs + [targexp_irrel]
#     return mus, vs


def mus_vs_matrix(Nmin, Nmax, gamma=0.5, u=0.7):
    mus = np.zeros(Nmax+1)
    vs = np.zeros((Nmax+1, Nmax+1))
    for N in range(Nmin, Nmax+1):
        for s in range(0, N + 1):
            # targexp_rel, targexp_irrel = targexp_per_relgrade(N, s, gamma, u)
            # mus[N][s] = targexp_rel
            vs[N][s] = targexp_irrel(N,s,gamma,u)
    for s in range(0,Nmax + 1):
        mus[s] = targexp_rel(s,gamma,u)
    return mus, vs


def f(x, k=0.7):
    return k * x


def targexp_rel(s,gamma=0.5,k=0.7):
    val = 0
    if s != 0:
        val = (1 - gamma ** s * (1 - k) ** s) / (s * (1 - gamma * (1 - k)))
    return val

def targexp_irrel(N,s,gamma=0.5,k=0.7):
    val = 0
    if s != N:
        val = ((1 - k) ** s * (gamma ** s - gamma ** N)) / ((N - s) * (1 - gamma))
    return val


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


    return targexp_rel(s,gamma,k), targexp_irrel(N,s,gamma,k)




def actexp_document(docid, rankdf, rhos, gamma=0.5, k=0.7):
    rankpos = rankdf[rankdf.document == docid].iloc[0]['rank']

    actexp = gamma ** (rankpos)
    for i in range(0, rankpos):
        doc_at_rank = rankdf[rankdf['rank'] == i + 1].iloc[0]['document']
        rho_at_rank = rhos[doc_at_rank]
        actexp *= (1 - f(rho_at_rank, k=k))

    return actexp


def actexp_documents(rankdf, rhos, gamma=0.5, k=0.7):
    rankdf = rankdf.sort_values(by='rank', ascending=True)
    mult_factors = [0] * len(rankdf)
    actexps = {}
    for i in range(len(rankdf)):
        row = rankdf.iloc[i]
        if i == 0:
            mult_factors[i] = 1
        else:
            doc_at_rank = row.document
            rho_at_rank = rhos[doc_at_rank]
            factor = (1 - f(rho_at_rank, k=k))
            mult_factors[i] = factor
        actexps[row.document] = gamma ** i * reduce(lambda x, y: x * y, mult_factors[0:i + 1])

    return actexps


def targexp_document(document, rhos, mus, vs, N):  # rhos do not have to be sorted
    rho = rhos[document]

    poibin_params = [v for k, v in rhos.items() if not k == document]
    pb = PoiBin(poibin_params)
    targexp = 0
    for s in range(0, N):
        targexp += pb.pmf(s) * (rho * mus[s + 1] + (1 - rho) * vs[s])
    return targexp


import sys, os


# Disable
def block_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__


def advantage_mean(producers, producer_advantages, t):
    if len(producers) == 0:
        return 0
    return sum([producer_advantages[producer][t] for producer in producers]) / len(producers)


def producer_advantage(actexp, targexp):
    expdiff = actexp - targexp
    return (expdiff ** 2) * math.copysign(1, expdiff)


def naive_controller(rhos, doc_to_producer_mapping, producer_to_doc_mapping, mus_all, vs_all,
                     theta=0.9, gamma=0.5, k=0.7, verbose=True):
    if not verbose:
        block_print()
    producers = list(producer_to_doc_mapping.keys())
    documents = list(doc_to_producer_mapping.keys())

    N = len(documents)
    mus = mus_all
    vs = vs_all[N][:N + 1]

    producer_advantages = {producer: [0] * 151 for producer in producers}

    producer_actual_expected_exposure = {producer: [0] * 150 for producer in producers}
    producer_target_expected_exposure = {} # target expected exposure doesn't depend on the actual ranking

    print("Initializing target expected experience per document...")
    targexp_documents = {}
    # for doc in tqdm(documents):
    for doc in documents:
        targexp_documents[doc] = targexp_document(doc, rhos, mus, vs, N)

    print("Initializing producer advantage, actual expected exposure, target_expected_exposure...")
    # for producer in tqdm(producers):
    for producer in producers:
        producer_doclist = producer_to_doc_mapping[producer]
        producer_target_expected_exposure[producer] = sum([targexp_documents[doc] for doc in producer_doclist])

    sequence_df = pd.DataFrame(columns=['q_num', 'document', 'score', 'rank'])

    # for t in tqdm(range(0, 150)):
    for t in range(0, 150):
        print("Computing doc advantages...")
        t0 = time.time()
        document_advantages = {}
        for document in documents:
            doc_producers = doc_to_producer_mapping[document]
            document_advantages[document] = advantage_mean(doc_producers, producer_advantages, t)
        t1 = time.time()
        print(f"Doc advantages took {round(t1 - t0, 2)}s.")

        print("Computing hscores...")
        hscores = {}
        for document in documents:
            hscores[document] = theta * rhos[document] + (1 - theta) * document_advantages[document]
        t2 = time.time()
        print(f"Hscores took {round(t2 - t1, 2)}s.")

        hscore_df = pd.DataFrame({'q_num': t, 'document': list(hscores.keys()), 'score': list(hscores.values())})
        # tqdm.pandas()
        # according to text: ties broken randomly
        hscore_df['rank'] = hscore_df.score.rank(method='first', ascending=False)
        hscore_df = hscore_df.astype({'rank': int})

        updated_doc_actexp = actexp_documents(hscore_df, rhos, gamma, k)
        t25 = time.time()
        print(f"Updating actual expected exposures for documents took {round(t25 - t2, 2)}s.")


        print("Updating expected exposures and advantages...")
        for producer in producers:
            prod_docs = producer_to_doc_mapping[producer]
            # overlaplist = [doc for doc in prod_docs if doc in documents]
            new_actual_exp = 0
            # new_target_exp = 0
            for doc in prod_docs:
                new_actual_exp += updated_doc_actexp[doc]
                # new_target_exp += targexp_documents[doc]

            producer_actual_expected_exposure[producer][t] = new_actual_exp
            # producer_target_expected_exposure[producer][t] = new_target_exp

            producer_advantages[producer][t + 1] = producer_advantages[producer][t] + producer_advantage(
                producer_actual_expected_exposure[producer][t], producer_target_expected_exposure[producer])
        t3 = time.time()
        print(f"Expeval etc. update took {round(t3 - t2, 2)}s.")
        sequence_df = sequence_df.append(hscore_df)

    if not verbose:
        enable_print()
    return sequence_df


def res_to_doc_to_author_mapping(res):
    hits = res['hits']['hits']
    doc_to_author_mapping = {}
    for hit in tqdm(hits):
        doc_to_author_mapping[hit['_id']] = hit['_source']['author_ids']
    # doc_to_author_mapping = {hit['_id']: hit['_source']['author_ids'] for hit in hits}
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
        stored_mapping = json.load(fp)
    mapping = {doc: stored_mapping.get(doc, []) for doc in docids}
    return mapping


def author_doc_mapping(doc_author_mapping):
    author_doc_mapping = {}
    for docid, authorlist in doc_author_mapping.items():
        for author in authorlist:
            if author not in author_doc_mapping:
                author_doc_mapping[author] = []
            author_doc_mapping[author] = author_doc_mapping[author] + [docid]

    return author_doc_mapping


def main():
    # queries = concat_sample_files()
    seq = 'evaluation/2020/TREC-Fair-Ranking-eval-full-seq.tsv'
    q = 'evaluation/2020/TREC-Fair-Ranking-eval-sample.json'
    ioh = IOHandlerKR(seq, q)

    estimated_relevances = pd.read_csv('klettirenders/relevances/Evaluation_rel_scores_model_A.csv')
    qseq_with_relevances = pd.merge(ioh.get_query_seq(), estimated_relevances, on=['qid', 'doc_id', ],
                                    how='left').sort_values(by=['sid', 'q_num']).reset_index(drop=True)

    # set the est relevance of each item that doesn't have an estimated relevance to 0
    qseq_with_relevances = qseq_with_relevances.fillna(0)

    docids = qseq_with_relevances.doc_id.drop_duplicates().to_list()
    doc_to_author_mapping = get_doc_to_author_mapping(docids, 'klettirenders/mappings/evaluation_doc_to_author.json')

    assert len(doc_to_author_mapping) == len(ioh.get_query_seq().doc_id.drop_duplicates())

    outdf = pd.DataFrame(columns=['sid', 'q_num', 'document', 'score', 'rank'])

    Nmin = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().min()
    Nmax = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().max()

    mus, vs = mus_vs_matrix(Nmin, Nmax)

    sids = qseq_with_relevances.sid.drop_duplicates().to_list()
    for sid in tqdm(sids):
        subdf = qseq_with_relevances[qseq_with_relevances.sid == sid]

        subdocids = subdf.doc_id.drop_duplicates().to_list()
        sub_doc_to_author_mapping = {k: v for k, v in doc_to_author_mapping.items() if k in subdocids}
        sub_author_to_doc_mapping = author_doc_mapping(sub_doc_to_author_mapping)

        rhos_df = subdf[['doc_id', 'est_relevance']].drop_duplicates()
        rhos = dict(zip(rhos_df.doc_id, rhos_df.est_relevance))

        seq_df = naive_controller(rhos, sub_doc_to_author_mapping, sub_author_to_doc_mapping, mus, vs, verbose=False)
        seq_df['sid'] = sid

        outdf = outdf.append(seq_df[['sid', 'q_num', 'document', 'score', 'rank']])

    outdf = outdf[['sid', 'q_num', 'document', 'score', 'rank']]
    qseq = ioh.get_query_seq()[['sid', 'q_num', 'qid']]
    submission = pd.merge(outdf, qseq, on=['sid', 'q_num'], how='inner').drop_duplicates()
    submission = submission.groupby(['sid', 'q_num', 'qid']).apply(lambda df: pd.Series({'ranking': df['document']}))
    submission = submission.reset_index()
    q_num = [str(submission['sid'][i]) + "." + str(submission['q_num'][i]) for i in range(len(submission))]
    submission['q_num'] = q_num
    submission = submission.drop('sid', axis=1)
    submission.to_json("nle_meta_9.json", orient='records', lines=True)


if __name__ == '__main__':
    main()
