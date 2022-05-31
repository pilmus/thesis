

#
# read_qrels
#
# QID GID DID REL
#
from app.evaluation.src.y2020.eval.permutation import Permutation


def read_qrels(fn, binarize, complete):
    qrels = {}
    did2gids = {}
    #
    # read qrels
    #
    with open(fn, 'r') as fp:
        for line in fp:
            fields = line.strip().split()
            if (len(fields) == 3):
                qid = fields[0]
                itr = "-1"  # no group == -1
                did = fields[1]
                rel = fields[2]
            else:
                qid = fields[0]
                itr = fields[1]
                did = fields[2]
                rel = fields[3]
            gids = []
            gids = map(lambda x: int(x), itr.split("|"))  # splits group ids by "|", so if >1 groups should be div by |

            rel = int(rel)
            if (rel > 0) and binarize:
                rel = 1
            if complete or (rel > 0):
                if not (qid in qrels):  # if qid not yet in the qrel dictionary, then...
                    qrels[qid] = {}
                    did2gids[qid] = {}
                qrels[qid][did] = rel
                if not (did in did2gids[qid]):
                    did2gids[qid][did] = []
                for gid in gids:
                    if not (gid in did2gids[qid][did]):
                        did2gids[qid][did].append(gid)

    return qrels, did2gids


def read_topfile(fn):
    #
    # get ranked lists
    #
    with open(fn, "r") as fp:
        sample_ids = set([])
        rls = {}  # qid x iteration -> permutation
        for line in fp:
            fields = line.strip().split()
            qid, itr, did, rank, score = fields[:5] # itr here is the iteration of the specific query
            sample_ids.add(itr)
            score = float(score)
            rank = int(rank)
            if not (qid in rls):
                rls[qid] = {}
            if not (itr in rls[qid]):
                rls[qid][itr] = Permutation()
            rls[qid][itr].add(rank, did)
    return rls
