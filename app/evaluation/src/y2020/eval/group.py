from app.evaluation.src.y2020.eval.metrics import GroupDisparity, GroupRelevance, GroupDifference


def exposure(exposures, did2gids, qrels, complete):
    """

    :param exposures: The exposure on the docs for a single query
    :param did2gids: The docid to groupid mapping for a single query
    :param qrels: The true relevance of the documents
    :param complete: reranking (true) or retrieval (false)
    :return:
    """
    groupExposures = {}
    for did,gids in did2gids.items():
        if (complete) or ((did in qrels) and (qrels[did] > 0)):
            for gid in gids:
                if not(gid in groupExposures):
                    groupExposures[gid] = 0.0
                if did in exposures:
                    groupExposures[gid] += exposures[did]
    return groupExposures

def metrics(target, umType, p, u, n, r):
    """

    :param target: target exposure aggregated by group
    :param umType:
    :param p:
    :param u:
    :param n: size of the ranking
    :param r: Total items with relevance == 1.
    :return:
    """
    k = len(target)
    disparity = GroupDisparity(target, umType, p, u, r, k)
    relevance = GroupRelevance(target, umType, p, u, r, k)
    difference = GroupDifference(target, umType, p, u, r, k)
    return disparity, relevance, difference
