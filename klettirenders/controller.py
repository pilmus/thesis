import sys

import numpy as np
import pandas as pd

#
# function: expected
#
# compute the expected exposure for a set of permutations according to
# different user models.
#
# RBP [1]
#
# exposure_i = p^i
#
# where i is the base 0 rank of the document
#
# GERR [2; section 7.2]
#
# exposure_i = p^i * \prod_{j<i} (1-u^r_j)
#            = p^i * (1-u)^(|{j<i : r_j=1}|)
#
# where
#
# r_j = 0 if document at rank j has relevance 0
#       1 otherwise
#
# [1]    Alistair Moffat and Justin Zobel. Rank-biased precision for
# measurement of retrieval effectiveness. ACM Trans. Inf. Syst.,
# 27(1):2:1--2:27, December 2008.
# [2]	Olivier Chapelle, Donald Metzler, Ya Zhang, and Pierre Grinspan.
# Expected reciprocal rank for graded relevance. In Proceedings of the 18th acm
# conference on information and knowledge management, CIKM '09, 621--630, New
# York, NY, USA, 2009. , ACM.
#
from attrdict import AttrDict
from elasticsearch import Elasticsearch

from interface.iohandler import InputOutputHandler


def expected(permutations, qrels, umType, p, u):
    """

    :param permutations: permutations as returned by the system we're evaluating
    :param qrels:
    :param umType:
    :param p:
    :param u:
    :return: the exposure that's on each document across all permutations
    """
    numSamples = len(permutations.keys())
    exposures = {}
    for itr, permutationObj in permutations.items():
        permutation = permutationObj.value()
        relret = 0
        for i in range(len(permutation)):
            did = permutation[i]
            if not (did in exposures):
                exposures[did] = 0.0

            if (umType == "rbp"):
                e_i = p ** (i)
            elif (umType == "gerr"):
                e_i = p ** (i) * (1.0 - u) ** (relret)
            else:
                raise ValueError(f"Invalid umType specified:{umType}.")

            exposures[did] += (e_i / numSamples)
            if (did in qrels) and (qrels[did] > 0):
                relret = relret + 1
    return exposures


def f(x, k=0.7):
    return k * x


def expected_exposure(N, s, gamma=0.5, u=0.7):
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
    relevance_levels = [[s, 1], [(N - s), 0]]

    targetExposurePerRelevanceLevel = {}
    for i in range(len(relevance_levels)):
        rel = relevance_levels[i][0]  # rellevel
        num_rel = relevance_levels[i][1]  # num of docs of this level

        if rel == 1:
            targexp = (1 - gamma ** s * (1 - u) ** s) / (s(1 - gamma * (1 - u)))
        elif rel == 0:
            targexp = ((1 - u) ** s * (gamma ** s - gamma ** N)) / ((N - s) * (1 - gamma))
        else:
            raise ValueError(f'Only binary relevance supported.')
        targetExposurePerRelevanceLevel[rel] = targexp


def load_author_list(pa_file):
    pa = pd.read_csv(pa_file)
    authors = pa.corpus_author_id.drop_duplicates().to_list()
    return authors


def concat_sample_files():
    ts = pd.read_json('training/2020/TREC-Fair-Ranking-training-sample.json', lines=True).explode('documents')
    ts['doc_id'] = ts.documents.apply(lambda row: row.get('doc_id'))
    es = pd.read_json('evaluation/2020/TREC-Fair-Ranking-eval-sample.json', lines=True).explode('documents')
    es['doc_id'] = es.documents.apply(lambda row: row.get('doc_id'))
    ests = pd.concat([es, ts])
    return ests


def sample_files_doc_ids():
    ests = concat_sample_files()
    return ests.doc_id.drop_duplicates().to_list()


def mappings_from_res(res):
    res = AttrDict(res)
    hits = res.hits.hits
    author_to_doc_mapping = {}
    doc_to_author_mapping = {}
    for hit in hits:
        source = hit['_source']
        docid = hit.get('_id')
        authors = source.get('author_ids')
        doc_to_author_mapping[docid] = authors
        for author in authors:
            if author not in author_to_doc_mapping:
                author_to_doc_mapping[author] = []
            print(author_to_doc_mapping)
            author_to_doc_mapping[author] = author_to_doc_mapping[author] + [docid]
    return author_to_doc_mapping, doc_to_author_mapping


def sample_files_author_doc_mappings(ids):
    es = Elasticsearch(timeout=500)
    res = es.search(index='semanticscholar2020og', body={'size': len(ids), 'query': {'ids': {'values': ids}}})
    return mappings_from_res(res)


def controller(sequence):
    print('a')
    sys.exit()
    qid = sequence.iloc[0].qid
    docs = queries[queries.qid == qid]
    #
    # authors = load_author_list()
    # documents = load_doclist()
    docids = docs.doc_id.drop_duplicates().to_list()
    author_to_doc_mapping, doc_to_author_mapping = sample_files_author_doc_mappings(docids)
    authors = list(author_to_doc_mapping.keys())

    # initialize advantage dict for all producers (id to advantage mapping)
    advantages = {author_id: 0 for author_id in authors}

    # initialize target expected exposure dict for all producers
    targ_exps = {author_id: 0 for author_id in authors}

    # initialize actual expected exposure dict for all producers
    real_exps = {author_id: 0 for author_id in authors}

    # initialize advantage dict for all documents (size = # docs x # iterations in sequence)
    doc_advantages = np.zeros((len(sequence), len(docids)))

    for t in range(0, len(sequence)):
        # for each document get the producers, take mean of the producer advantages (def advantage_mean(producer_advs))

        # compute the controller scores (h) --> don't have to save these, are re-computed each iteration (h[i,t] = theta * rho[i] +(1-theta)*Adv[i,t], i is the doc num)
        # sort documents by computed h-scores, save ranking in later output df

        # foreach producer
        # # update actual expected exposure (AEE[p,t] = AEE[p,t-1] + )
        # # update target expected exposure
        # # update producer advantages ()
        continue

    pass


if __name__ == '__main__':
    # queries = concat_sample_files()
    seq = 'training/2020/training-sequence-full.tsv'
    q = 'training/2020/TREC-Fair-Ranking-training-sample.json'
    ioh = InputOutputHandler(seq, q)

    qseq = ioh.get_query_seq()

    controller(qseq[qseq.sid == 0])
