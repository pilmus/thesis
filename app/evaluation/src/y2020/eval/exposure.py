import math


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
from app.evaluation.src.y2020.eval import metrics


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


#
# function: target_exposures
#
# given a user model and its parameters, compute the target exposures for
# documents.
#
#
def target(qrels, umType, p, u, complete):
    """
    :param qrels: dictionary mapping docids to a relevance value. all docids should belong to a single qid
    :param umType: user model, can be rbp or gerr. GERR is used in the overview paper
    :param p: patience, default = 0.5
    :param u: utility, default = 0.5
    :param complete: complete relevant judgments known? used for re-ranking
    :return:
    """
    #
    # compute [ [relevanceLevel, count], ...] vector
    # example: [[0, 4],[1,3]] --> there were 7 items, 4 with relevance level 0 and 3 with relevance level 1
    #
    relevanceLevelAccumulators = {}
    relevanceLevels = []
    for did, rel in qrels.items():
        if rel in relevanceLevelAccumulators:
            relevanceLevelAccumulators[rel] += 1
        else:
            relevanceLevelAccumulators[rel] = 1
    for k, v in relevanceLevelAccumulators.items():
        relevanceLevels.append([k, v])
    relevanceLevels.sort(reverse=True)  # reverse == true to sort in desc order, so higher rels first
    #
    # compute { relevanceLevel : exposure }
    #
    b = 0  # numDominating
    targetExposurePerRelevanceLevel = {}
    for i in range(len(relevanceLevels)):
        g = relevanceLevels[i][0] # rellevel
        m = relevanceLevels[i][1] # num of docs of this level
        if (umType == "rbp"):
            te_g = (p ** (b) - p ** (b + m)) / (m * (1.0 - p))
        elif (umType == "gerr"):
            pp = p * (1.0 - u)
            if (g > 0):
                te_g = (pp ** (b) - pp ** (b + m)) / (m * (1.0 - pp))
            else:
                te_g = ((1.0 - u) ** (b) * (p ** (b) - p ** (b + m))) / (m * (1.0 - p))
        else:
            raise Exception("Invalid umType specified.")
        targetExposurePerRelevanceLevel[g] = te_g
        b = b + m
    #
    # create { did : exposure }
    #
    target = {}
    for did, rel in qrels.items():
        target[did] = targetExposurePerRelevanceLevel[rel]  # <-- target exposure level fully dependent on relevance
    #
    # compute the metric structure to maintain bounds, defaults, etc
    #
    n = len(qrels) if (complete) else math.inf  # for re-ranking == len(qrels)
    disparity = metrics.Disparity(target, umType, p, u, relevanceLevels, n)
    relevance = metrics.Relevance(target, umType, p, u, relevanceLevels, n)
    difference = metrics.Difference(target, umType, p, u, relevanceLevels, n)
    return target, disparity, relevance, difference
