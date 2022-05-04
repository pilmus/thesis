import itertools
import math

import pandas as pd

from app.reranking.src import model
from app.reranking.src.p_controller import actexp_documents, targexp_document


def util_ranking(rankdf, rhos, gamma=0.5, k=0.7):
    """Compute the utility for this ranking as given in eq 8 of Biega2021 todo:change name"""
    actexps = actexp_documents(rankdf, rhos, gamma, k)
    for doc, actexp in actexps.items():
        actexps[doc] = actexp * rhos[doc]

    return sum(actexps.values())


def exposure_disparity(E, E_star):
    """TODO: Also implement the other option for exposure disparity"""
    return E - E_star


def small_delta(document_authors, document_author_actual_expected_exposures, document_author_target_expected_exposures,
                t):
    """Compute the small delta value. Computed as the sum of E_p - E_p* for all producers of this document i"""
    total = 0
    for author in document_authors:
        total += exposure_disparity(document_author_actual_expected_exposures[author][t],
                                    document_author_target_expected_exposures[author][t])

    return total


def big_delta():
    raise NotImplementedError


# TODO: implement option for singleton document grouping

class MRFR(model.RankerInterface):

    def __init__(self, relevance_probabilities, grouping, K, beta, lambd):
        super().__init__()
        self._relevance_probabilities = relevance_probabilities
        self._grouping = grouping
        self._K = K
        self._beta = beta
        self._lambda = lambd

    def rerank_loop(self, rhos, doc_to_producer_mapping, producer_to_doc_mapping, mus_all, vs_all, K, beta, lambd):
        producers = list(producer_to_doc_mapping.keys())
        documents = list(doc_to_producer_mapping.keys())

        #  1: procedure MRFR_adapted
        #  2:   for each producer set the exposures at time 0 to 0
        #  3:   set the utilities at time 0 to 0
        #  3:   for t=1 to t=n:
        #  4:     for each document i small_delta(i,t) = sum(act_exp(p,t-1) - targ_exp(p,t-1)), p produces i
        #  5:     for each document i phi(i,t) = rho(i) - beta * small_delta(i, t)
        #  6:     sort the documents by phi(i,t)
        #  7:     take the top K documents and create all permutations of the top
        #  8:     create candidate rankings pi(c) by appending the permutations to the remains
        #  9:     for each candidate ranking compute:
        # 10:       - U as (u(this ranking) + sum(u all prev rankings)) / t
        # 11:       - big_delta as given in eq 3 biega 2021, where E(t) is E(t-1) + E(candidate ranking)
        # 12:     and combine into psi as psi(cand) = U - lambda * big_delta
        # 13:     pick cand with highest psi and return

        N = len(documents)
        mus = mus_all
        vs = vs_all[N][:N + 1]

        producer_accumulated_exposure_difference = {producer: [0] * 151 for producer in producers}

        producer_actual_expected_exposure = {producer: [0] * 151 for producer in producers}
        targexp_documents = {}
        for doc in documents:
            targexp_documents[doc] = targexp_document(doc, rhos, mus, vs,
                                                      N)  # todo: precomputable for all docs in all rankings?

        producer_target_expected_exposure = {}
        for producer in producers:
            docs_for_producer = producer_to_doc_mapping[producer]
            producer_target_expected_exposure[producer] = sum([targexp_documents[doc] for doc in docs_for_producer])

        producer_actual_expected_exposure = {producer: [0] * 151 for producer in producers}

        utilites = [0] * 151



        sequence_df = pd.DataFrame(columns=['q_num', 'doc_id', 'score', 'rank'])

        for t in range(1, 151):
            phis = {doc: 0 for doc in documents}
            for document in documents:
                doc_producers = doc_to_producer_mapping[document]
                phis[document] = rhos[document] - beta * small_delta(doc_producers, producer_actual_expected_exposure,
                                                                     producer_target_expected_exposure, t)

            phis = dict(sorted(phis.items(), key=lambda phi: phi[1]))
            topKdocs = list(phis.keys())[:K]
            restdocs = list(phis.keys())[K:]

            topKperms = itertools.permutations(topKdocs)

            candidate_rankings = {}

            for i, perm in enumerate(topKperms):
                candidate_rankings[i] = list(perm) + restdocs

            candidate_utils = {}
            candidate_actexps_documents = {}
            for i, c_ranking in candidate_rankings.items():
                c_rankdf = pd.DataFrame({'rank': list(range(1, len(c_ranking) + 1)), 'doc_id': c_ranking})
                candidate_utils[i] = (util_ranking(c_rankdf, rhos) + utilites[t - 1]) / t
                candidate_actexps_documents[i] = actexp_documents(c_rankdf, rhos)

            candidate_actexps_authors = {}

            for i, doc_actexps in candidate_actexps_documents.items():
                candidate_actexps_authors[i] = {}
                for producer in producers:
                    prod_docs = producer_to_doc_mapping[producer]
                    candidate_actexp_author = 0
                    for doc in prod_docs:
                        candidate_actexp_author += doc_actexps[doc]
                    candidate_actexps_authors[i][producer] = candidate_actexp_author

            big_deltas = {}
            for i in candidate_rankings:
                bigdelta = 0
                for producer in producers:
                    bigdelta += (candidate_actexps_authors[i][producers] + producer_actual_expected_exposure[producer][
                        t - 1] - t * producer_target_expected_exposure[producer]) ** 2
                big_deltas[i] = math.sqrt(bigdelta)

            psis = {}

            for i in candidate_rankings:
                psis[i] = candidate_utils[i] - lambd * big_deltas[i]

            best_candidate = max(psis, key=psis.get)

            candidate_rankings[i] #todo: convert into ranking


        def rerank(self, ioh):
            rel_probs = pd.read_csv(self._relevance_probabilities)
            qseq_with_relevances = pd.merge(ioh.get_query_seq(), rel_probs, on=['qid', 'doc_id'],
                                            how='left').sort_values(
                by=['sid', 'q_num']).reset_index(drop=True)
            qseq_with_relevances = qseq_with_relevances.fillna(0)
            docids = qseq_with_relevances.doc_id.drop_duplicates().to_list()

            grouping = self.get_grouping(docids, self._grouping)
