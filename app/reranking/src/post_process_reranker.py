import itertools
import json
import math
import time
from functools import reduce
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm import tqdm
from app.reranking.poibin.poibin import PoiBin
from app.reranking.src import model

from app.utils.src.utils import invert_key_to_list_mapping, block_print, enable_print, write_json


# todo: move methods below to appropriate location; something with making mappings
def res_to_doc_to_author_mapping(res):
    hits = res['hits']['hits']
    doc_to_author_mapping = {}
    for hit in tqdm(hits):
        doc_to_author_mapping[hit['_id']] = hit['_source']['author_ids']
    return doc_to_author_mapping


def doc_to_author_mapping_from_es(docids, outfile):
    es = Elasticsearch(timeout=500)
    res = es.search(index='semanticscholar2020og', body={'size': len(docids), 'query': {'ids': {'values': docids}}})
    mapping = res_to_doc_to_author_mapping(res)
    write_json(mapping, outfile)
    return mapping


def doc_singleton_grouping(docids, outfile):
    mapping = {doc: [doc] for doc in docids}
    write_json(mapping, outfile)


class PostProcessReranker(model.RankerInterface):
    def __init__(self, estimated_relevance, grouping, missing_group_strategy):
        super().__init__()
        self._erels = estimated_relevance
        self._grouping = grouping
        self._missing_group_strategy = missing_group_strategy

    def get_doc_to_group_mapping(self, docids, mapping_file, missing_group_strategy):
        with open(mapping_file) as fp:
            mapping = json.load(fp)

        if missing_group_strategy == 'ignore':
            mapping = {doc: mapping.get(doc, []) for doc in docids}
        elif missing_group_strategy == 'single_dummy':
            mapping = {doc: mapping.get(doc, ['dummy']) for doc in docids}
        elif missing_group_strategy == 'docid_dummy':
            mapping = {doc: mapping.get(doc, [doc]) for doc in docids}
        else:
            raise ValueError('Invalid missing group strategy: ', missing_group_strategy)
        return mapping

    def mus_vs_matrix(self, Nmin, Nmax, gamma=0.5, u=0.7):
        mus = np.zeros(Nmax + 1)
        vs = np.zeros((Nmax + 1, Nmax + 1))
        for N in range(Nmin, Nmax + 1):
            for s in range(0, N + 1):
                vs[N][s] = self.targexp_irrel(N, s, gamma, u)

        for s in range(0, Nmax + 1):
            mus[s] = self.targexp_rel(s, gamma, u)
        return mus, vs

    def f(self, x, k=0.7):
        return k * x

    def targexp_rel(self, s, gamma=0.5, k=0.7):
        val = 0
        if s != 0:
            val = (1 - gamma ** s * (1 - k) ** s) / (s * (1 - gamma * (1 - k)))
        return val

    def targexp_irrel(self, N, s, gamma=0.5, k=0.7):
        val = 0
        if s != N:
            val = ((1 - k) ** s * (gamma ** s - gamma ** N)) / ((N - s) * (1 - gamma))
        return val

    def targexp_per_relgrade(self, N, s, gamma=0.5, k=0.7):
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

        return self.targexp_rel(s, gamma, k), self.targexp_irrel(N, s, gamma, k)

    def actexp_documents(self, ranking, rhos, gamma=0.5, k=0.7):
        """Takes in a ranking and returns the actual exposure that is on each document. Ranking should be in desired order already."""
        mult_factors = [0] * len(ranking)
        actexps = {}
        for i in range(len(ranking)):

            doc_at_rank = ranking[i]
            rho_at_rank = rhos[doc_at_rank]
            factor = (1 - self.f(rho_at_rank, k=k))
            mult_factors[i] = factor
            if i == 0:
                actexps[doc_at_rank] = 1
            else:
                prob_user_already_satisfied = reduce(lambda x, y: x * y, mult_factors[0:i])
                prob_user_gives_up = gamma ** i
                actexps[doc_at_rank] = prob_user_gives_up * prob_user_already_satisfied

        return actexps

    def actexp_document(self, docid, rankdf, rhos, gamma=0.5, k=0.7):
        rankpos = rankdf[rankdf.document == docid].iloc[0]['rank']

        actexp = gamma ** (rankpos)
        for i in range(0, rankpos):
            doc_at_rank = rankdf[rankdf['rank'] == i + 1].iloc[0]['document']
            rho_at_rank = rhos[doc_at_rank]
            actexp *= (1 - self.f(rho_at_rank, k=k))

        return actexp

    def targexp_document(self, document, rhos, mus, vs, N):  # rhos do not have to be sorted
        rho = rhos[document]

        poibin_params = [v for k, v in rhos.items() if not k == document]
        pb = PoiBin(poibin_params)
        targexp = 0
        for s in range(0, N):
            targexp += pb.pmf(s) * (rho * mus[s + 1] + (1 - rho) * vs[s])
        return targexp

    def rerank(self, ioh):
        erels = pd.read_csv(self._erels)
        qseq_with_relevances = pd.merge(ioh.get_query_seq(), erels, on=['qid', 'doc_id'],
                                        how='left').sort_values(
            by=['sid', 'q_num']).reset_index(drop=True)
        qseq_with_relevances = qseq_with_relevances.fillna(0)
        docids = qseq_with_relevances.doc_id.drop_duplicates().to_list()

        doc_to_group_mapping = self.get_doc_to_group_mapping(docids, self._grouping,
                                                             missing_group_strategy=self._missing_group_strategy)

        outdf = pd.DataFrame(columns=['sid', 'q_num', 'doc_id', 'rank'])
        Nmin = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().min()
        Nmax = qseq_with_relevances.groupby(['sid', 'q_num']).doc_id.count().max()
        mus, vs = self.mus_vs_matrix(Nmin, Nmax)
        sids = qseq_with_relevances.sid.drop_duplicates().to_list()
        for sid in tqdm(sids):
            subdf = qseq_with_relevances[qseq_with_relevances.sid == sid]

            subdocids = subdf.doc_id.drop_duplicates().to_list()
            sub_doc_to_group_mapping = {k: v for k, v in doc_to_group_mapping.items() if k in subdocids}
            sub_group_to_doc_mapping = invert_key_to_list_mapping(sub_doc_to_group_mapping)

            rhos_df = subdf[['doc_id', 'est_relevance']].drop_duplicates()
            rhos = dict(zip(rhos_df.doc_id, rhos_df.est_relevance))

            seq_df = self.rerank_loop(rhos, sub_doc_to_group_mapping, sub_group_to_doc_mapping, mus, vs, gamma=0.5,
                                      k=0.7,
                                      verbose=False)
            seq_df['sid'] = sid

            outdf = outdf.append(seq_df[['sid', 'q_num', 'doc_id', 'rank']])
        outdf = outdf[['sid', 'q_num', 'doc_id', 'rank']]

        return outdf

    def rerank_loop(self, rhos, doc_to_group_mapping, group_to_doc_mapping, mus, vs, gamma, k, verbose):
        raise NotImplementedError("Override this function.")

    def train(self, inputhandler):
        pass

    def _predict(self, inputhandler):
        rerankings = self.rerank(inputhandler)

        pred = pd.merge(inputhandler.get_query_seq()[['sid', 'q_num', 'qid', 'doc_id']],
                        rerankings, how='left', on=['sid', 'q_num', 'doc_id'])

        return pred


class AdvantageController(PostProcessReranker):
    def __init__(self, est_rel_file, grouping, missing_group_strategy, theta):
        super().__init__(est_rel_file, grouping, missing_group_strategy)
        self._theta = theta

    def advantage_mean(self, groups, group_advantages, t):
        if len(groups) == 0:
            return 0
        return sum([group_advantages[group][t] for group in groups]) / len(groups)

    def group_advantage(self, actexp, targexp):
        expdiff = actexp - targexp
        new_adv = (expdiff ** 2) * math.copysign(1, expdiff)
        return new_adv

    def rerank_loop(self, rhos, doc_to_group_mapping, group_to_doc_mapping, mus_all, vs_all, gamma, k, verbose):
        if not verbose:
            block_print()
        groups = list(group_to_doc_mapping.keys())
        documents = list(doc_to_group_mapping.keys())

        N = len(documents)
        mus = mus_all
        vs = vs_all[N][:N + 1]

        group_advantages = {group: [0] * 152 for group in groups}
        group_actual_expected_exposure = {group: [0] * 151 for group in groups}
        document_advantages = {document: [0] * 151 for document in documents}

        print("Initializing target expected experience per document...")
        targexp_documents = {}
        for doc in documents:
            targexp_documents[doc] = self.targexp_document(doc, rhos, mus, vs, N)

        print("Initializing group advantage, actual expected exposure, target_expected_exposure...")
        group_target_expected_exposure = {}  # target expected exposure doesn't depend on the actual ranking
        for group in groups:
            group_doclist = group_to_doc_mapping[group]
            group_target_expected_exposure[group] = sum([targexp_documents[doc] for doc in group_doclist])

        sequence_df = pd.DataFrame(columns=['q_num', 'doc_id', 'rank'])

        for t in range(1, 151):
            print("Computing doc advantages...")
            t0 = time.time()

            for document in documents:
                doc_groups = doc_to_group_mapping[document]
                document_advantages[document][t] = self.advantage_mean(doc_groups, group_advantages, t)

            t1 = time.time()
            print(f"Doc advantages took {round(t1 - t0, 2)}s.")
            print("Computing hscores...")

            hscores = {}
            for document in documents:
                hscores[document] = self.hscore(document, document_advantages, rhos, t)

            t2 = time.time()
            print(f"Hscores took {round(t2 - t1, 2)}s.")

            hscore_df = pd.DataFrame(
                {'q_num': t - 1, 'doc_id': list(hscores.keys()), 'score': list(hscores.values())})
            hscore_df['rank'] = hscore_df.score.rank(method='first',
                                                     ascending=False)
            hscore_df = hscore_df.sort_values(by='rank')
            updated_doc_actexp = self.actexp_documents(hscore_df.doc_id.to_list(), rhos, gamma, k)

            t25 = time.time()
            print(f"Updating actual expected exposures for documents took {round(t25 - t2, 2)}s.")
            print("Updating expected exposures and advantages...")

            for group in groups:
                prod_docs = group_to_doc_mapping[group]
                new_actual_exp = 0
                for doc in prod_docs:
                    new_actual_exp += updated_doc_actexp[doc]

                group_actual_expected_exposure[group][t] = group_actual_expected_exposure[group][
                                                               t - 1] + new_actual_exp

                group_advantages[group][t + 1] = self.group_advantage(
                    group_actual_expected_exposure[group][t],
                    t * group_target_expected_exposure[group])

            t3 = time.time()
            print(f"Expeval etc. update took {round(t3 - t2, 2)}s.")

            sequence_df = sequence_df.append(hscore_df[['q_num', 'doc_id', 'rank']])

        if not verbose:
            enable_print()
        return sequence_df

    def hscore(self, document, document_advantages, rhos, t, method):
        base_hscore = self._theta * rhos[document] - (1 - self._theta) * document_advantages[document][t]
        if method == 'linear':
            return base_hscore
        elif method == 'max':
            return max(0, base_hscore)
        else:
            raise ValueError("Invalid hscore method: ", method)


class MRFR(PostProcessReranker):
    def __init__(self, estimated_relevance, grouping, missing_group_strategy, K, beta, lambd):
        super().__init__(estimated_relevance, grouping, missing_group_strategy)
        self._K = K
        self._beta = beta
        self._lambda = lambd

    def util_ranking(self, actexps, rhos):
        """Compute the utility for this ranking as given in eq 8 of Biega2021 todo:change name
        gamma and k are set as in kletti and renders
        """
        # confirmed correct until here
        # actexps = actexp_documents(rankdf, rhos, gamma, k)
        for doc, actexp in actexps.items():
            actexps[doc] = actexp * self.f(rhos[doc])

        return sum(actexps.values())

    def exposure_disparity(self, E, E_star):
        """TODO: Also implement the other option for exposure disparity"""
        return E - E_star

    def small_delta(self, document_authors, document_author_actual_expected_exposures,
                    document_author_target_expected_exposures,
                    t):
        """Compute the small delta value. Computed as the sum of E_p - E_p* for all producers of this document i"""
        total = 0
        for author in document_authors:
            total += self.exposure_disparity(document_author_actual_expected_exposures[author][t - 1],
                                             (t - 1) * document_author_target_expected_exposures[author])

        return total

    def rerank_loop(self, rhos, doc_to_producer_mapping, producer_to_doc_mapping, mus_all, vs_all, gamma, k, verbose):
        block_print()
        producers = list(producer_to_doc_mapping.keys())
        documents = list(doc_to_producer_mapping.keys())

        N = len(documents)
        mus = mus_all
        vs = vs_all[N][:N + 1]

        targexp_documents = {}
        for doc in documents:
            targexp_documents[doc] = self.targexp_document(doc, rhos, mus, vs, N)

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
                phis[document] = rhos[document] - self._beta * self.small_delta(doc_producers,
                                                                                producer_actual_expected_exposure,
                                                                                producer_target_expected_exposure, t)

            t1 = time.time()
            print(f"Doc advantages took {round(t1 - t0, 4)}s.")

            phis = dict(sorted(phis.items(), key=lambda phi: phi[1], reverse=True))
            topKdocs = list(phis.keys())[:self._K]
            restdocs = list(phis.keys())[self._K:]

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

                candidate_actexps_documents[i] = self.actexp_documents(c_ranking, rhos)
                t22 = time.time()
                print(f"Actexps took {round(t22 - t21, 4)}s.")

                candidate_utils[i] = (self.util_ranking(candidate_actexps_documents[i], rhos) + utilities[t - 1]) / t
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
                psis[i] = candidate_utils[i] - self._lambda * big_deltas[i]

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
