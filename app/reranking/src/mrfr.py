import itertools
import json
import math
import time
from functools import reduce

import pandas as pd
from tqdm import tqdm

from app.reranking.src import model
from app.reranking.src.p_controller import targexp_document, mus_vs_matrix, \
    invert_key_to_list_mapping, f, block_print, enable_print


def util_ranking(actexps, rhos, gamma=0.5, k=0.7):
    """Compute the utility for this ranking as given in eq 8 of Biega2021 todo:change name
    gamma and k are set as in kletti and renders
    """
    # confirmed correct until here
    # actexps = actexp_documents(rankdf, rhos, gamma, k)
    for doc, actexp in actexps.items():
        actexps[doc] = actexp * f(rhos[doc])

    return sum(actexps.values())


def exposure_disparity(E, E_star):
    """TODO: Also implement the other option for exposure disparity"""
    return E - E_star


def small_delta(document_authors, document_author_actual_expected_exposures, document_author_target_expected_exposures,
                t):
    """Compute the small delta value. Computed as the sum of E_p - E_p* for all producers of this document i"""
    total = 0
    for author in document_authors:
        total += exposure_disparity(document_author_actual_expected_exposures[author][t - 1],
                                    (t - 1) * document_author_target_expected_exposures[author])

    return total


# TODO: implement option for singleton document grouping

def get_doc_to_group_mapping(docids, _grouping, missing_group_strategy):
    with open(_grouping) as fp:
        mapping = json.load(fp)

    if missing_group_strategy == 'ignore':
        mapping = {doc: mapping.get(doc, []) for doc in docids}
    else:
        raise ValueError('Invalid missing group strategy: ', missing_group_strategy)
    return mapping


def actexp_documents(ranking, rhos, gamma=0.5, k=0.7):
    mult_factors = [0] * len(ranking)
    actexps = {}
    for i in range(len(ranking)):

        doc_at_rank = ranking[i]
        rho_at_rank = rhos[doc_at_rank]
        factor = (1 - f(rho_at_rank, k=k))
        mult_factors[i] = factor
        if i == 0:
            actexps[doc_at_rank] = 1
        else:
            prob_user_already_satisfied = reduce(lambda x, y: x * y, mult_factors[0:i])
            prob_user_gives_up = gamma ** i
            actexps[doc_at_rank] = prob_user_gives_up * prob_user_already_satisfied

    return actexps

class MRFR(model.RankerInterface):

    def __init__(self, relevance_probabilities, grouping, K, beta, lambd):
        super().__init__()
        self._relevance_probabilities = relevance_probabilities
        self._grouping = grouping
        self._K = K
        self._beta = beta
        self._lambda = lambd

    def rerank_loop(self, rhos, doc_to_producer_mapping, producer_to_doc_mapping, mus_all, vs_all, K, beta, lambd):
        block_print()
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

        targexp_documents = {}
        for doc in documents:
            targexp_documents[doc] = targexp_document(doc, rhos, mus, vs, N)

        producer_target_expected_exposure = {}
        for producer in producers:
            docs_for_producer = producer_to_doc_mapping[producer]
            producer_target_expected_exposure[producer] = sum([targexp_documents[doc] for doc in docs_for_producer])

        producer_actual_expected_exposure = {producer: [0] * 151 for producer in producers}

        utilities = [0] * 151

        sequence_df = pd.DataFrame(columns=['q_num', 'doc_id', 'rank'])

        for t in range(1, 151):
            phis = {doc: 0 for doc in documents}
            t0 = time.time()
            for document in documents:
                doc_producers = doc_to_producer_mapping[document]
                phis[document] = rhos[document] - beta * small_delta(doc_producers, producer_actual_expected_exposure,
                                                                     producer_target_expected_exposure, t)

            t1 = time.time()
            print(f"Doc advantages took {round(t1 - t0, 4)}s.")

            phis = dict(sorted(phis.items(), key=lambda phi: phi[1], reverse=True))
            topKdocs = list(phis.keys())[:K]
            restdocs = list(phis.keys())[K:]

            topKperms = itertools.permutations(topKdocs)

            candidate_rankings = {}

            for i, perm in enumerate(topKperms):
                candidate_rankings[i] = list(perm) + restdocs

            t2 = time.time()
            print(f"Computing permutations took {round(t2 - t1, 4)}s.")

            candidate_utils = {}
            candidate_actexps_documents = {}
            for i, c_ranking in candidate_rankings.items():


                # c_rankdf = pd.DataFrame({'rank': list(range(1, len(c_ranking) + 1)), 'doc_id': c_ranking})
                t21 = time.time()
                # print(f"Making cand ranking took {round(t21 - t2, 4)}s.")

                candidate_actexps_documents[i] = actexp_documents(c_ranking, rhos)
                t22 = time.time()
                print(f"Actexps took {round(t22 - t21, 4)}s.")

                candidate_utils[i] = (util_ranking(candidate_actexps_documents[i], rhos) + utilities[t - 1]) / t
                t23 = time.time()
                print(f"Cand utils took {round(t23 - t22, 4)}s.")

            t3 = time.time()
            print(f"Candidate doc actexp and utils took {round(t3 - t2, 4)}s.")

            candidate_actexps_authors = {}

            for i, doc_actexps in candidate_actexps_documents.items():
                candidate_actexps_authors[i] = {}
                for producer in producers:
                    prod_docs = producer_to_doc_mapping[producer]
                    candidate_actexp_author = 0
                    for doc in prod_docs:
                        candidate_actexp_author += doc_actexps[doc]
                    candidate_actexps_authors[i][producer] = candidate_actexp_author

            t4 = time.time()
            print(f"Canididate author actexps took  {round(t4 - t3, 4)}s.")

            big_deltas = {}
            for i in candidate_rankings:
                bigdelta = 0
                for producer in producers:
                    bigdelta += (candidate_actexps_authors[i][producer] + producer_actual_expected_exposure[producer][
                        t - 1] - t * producer_target_expected_exposure[producer]) ** 2
                big_deltas[i] = math.sqrt(bigdelta)

            t5 = time.time()
            print(f"Big deltas took {round(t5 - t4, 4)}s.")

            psis = {}


            for i in candidate_rankings:
                psis[i] = candidate_utils[i] - lambd * big_deltas[i]

            t6 = time.time()
            print(f"Psis took {round(t6 - t5, 4)}s.")

            best_candidate = max(psis, key=psis.get)

            for producer in producers:
                producer_actual_expected_exposure[producer][t] = producer_actual_expected_exposure[producer][t - 1] + \
                                                                 candidate_actexps_authors[best_candidate][producer]

            utilities[t] = utilities[t - 1] + candidate_utils[best_candidate]

            t7 = time.time()
            print(f"Actexps and utils took {round(t7 - t6, 4)}s.")

            next_ranking = candidate_rankings[best_candidate]
            rankdf = pd.DataFrame({'q_num': [t - 1] * N, 'doc_id': next_ranking, 'rank': list(range(1, N + 1))})
            sequence_df = sequence_df.append(rankdf)
            print()
        enable_print()
        return sequence_df

    def rerank(self, ioh):
        rel_probs = pd.read_csv(self._relevance_probabilities)
        qseq_with_relevances = pd.merge(ioh.get_query_seq(), rel_probs, on=['qid', 'doc_id'],
                                        how='left').sort_values(
            by=['sid', 'q_num']).reset_index(drop=True)
        qseq_with_relevances = qseq_with_relevances.fillna(0)
        docids = qseq_with_relevances.doc_id.drop_duplicates().to_list()

        doc_to_group_mapping = get_doc_to_group_mapping(docids, self._grouping, missing_group_strategy='ignore')

        outdf = pd.DataFrame(columns=['sid', 'q_num', 'doc_id', 'rank'])
        Nmin = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().min()
        Nmax = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().max()
        mus, vs = mus_vs_matrix(Nmin, Nmax)
        sids = qseq_with_relevances.sid.drop_duplicates().to_list()
        for sid in tqdm(sids):
            subdf = qseq_with_relevances[qseq_with_relevances.sid == sid]

            subdocids = subdf.doc_id.drop_duplicates().to_list()
            sub_doc_to_group_mapping = {k: v for k, v in doc_to_group_mapping.items() if k in subdocids}
            sub_group_to_doc_mapping = invert_key_to_list_mapping(sub_doc_to_group_mapping)

            rhos_df = subdf[['doc_id', 'est_relevance']].drop_duplicates()
            rhos = dict(zip(rhos_df.doc_id, rhos_df.est_relevance))

            seq_df = self.rerank_loop(rhos, sub_doc_to_group_mapping, sub_group_to_doc_mapping, mus, vs, self._K,
                                      self._beta, self._lambda)

            seq_df['sid'] = sid

            outdf = outdf.append(seq_df[['sid', 'q_num', 'doc_id', 'rank']])
        outdf = outdf[['sid', 'q_num', 'doc_id', 'rank']]

        return outdf

    def train(self, inputhandler):
        pass

    def _predict(self, inputhandler):
        rerankings = self.rerank(inputhandler)

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']], rerankings, how='left',
                        on=['sid', 'q_num', 'doc_id'])

        return pred
