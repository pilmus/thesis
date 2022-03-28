

#
# 
#
import math
import os

from app.evaluation.src.y2020.eval import cli, data, exposure, group


def main():
    parameters = cli.parseArguments()

    umType = parameters["umType"]
    umPatience = parameters["umPatience"]
    umUtility = parameters["umUtility"]
    binarize = parameters["binarize"]
    groupEvaluation = parameters["groupEvaluation"]
    complete = parameters["complete"]
    normalize = parameters["normalize"]

    square = parameters["sq"]

    destination = parameters["destination"]
    non_verbose = parameters["non_verbose"]

    relfn = parameters["relfn"]  # qrel file name
    topfn = parameters["topfn"]  # the run file?

    expeval(relfn, topfn, destination, binarize, complete, groupEvaluation, normalize, square, umPatience, umType,
                umUtility, non_verbose)


def expeval(qrels, runfile, outfile, binarize=False, complete=False, groupEvaluation=False, normalize=True, square=True, umPatience=0.5, umType='gerr',
                umUtility=0.5, non_verbose=True):
    if outfile and os.path.exists(outfile):
        os.remove(outfile)
    #
    # get target exposures
    #
    qrels, did2gids = data.read_qrels(qrels, binarize, complete)
    targExp = {}
    disparity = {}
    relevance = {}
    difference = {}
    for qid, qrels_qid in qrels.items():
        targ, disp, rel, diff = exposure.target(qrels_qid, umType,
                                                umPatience, umUtility,
                                                complete)
        targExp[qid] = targ
        disparity[qid] = disp
        relevance[qid] = rel
        difference[qid] = diff
    #
    # aggregate exposures if group evaluation and replace queries missing groups
    # with nulls
    #
    if groupEvaluation:
        for qid in targExp.keys():
            if qid in did2gids:
                t = targExp[qid]
                targ = group.exposure(t, did2gids[qid], qrels[qid], complete)  # og qrels used for targ compute
                n = len(t) if complete else math.inf
                r = sum(1 for v in qrels[qid].values() if v > 0)
                disp, rel, diff = group.metrics(targ, umType, umPatience,
                                                umUtility, n, r)
                targExp[qid] = targ
                disparity[qid] = disp
                relevance[qid] = rel
                difference[qid] = diff  # overwrite individual diff with group diff
            else:
                targExp[qid] = None
                disparity[qid] = None
                relevance[qid] = None
                difference[qid] = None
    #
    # get expected exposures for the run
    #
    permutations = data.read_topfile(runfile)
    runExp = {}
    for qid, permutations_qid in permutations.items():
        if (qid in qrels):
            runExp[qid] = exposure.expected(permutations_qid, qrels[qid], umType,
                                            umPatience, umUtility)
    #
    # aggregate exposures if group evaluation and replace queries missing groups
    # with nulls
    #
    # at the end of the loop below, runExp contains the actual group exposures for each query
    #
    if groupEvaluation:
        for qid in runExp.keys():  # for each query...
            if (
                    qid in did2gids):  # if query has documents that belong to a group... which all queries do, b/c if no group it's set to -1
                rexp = runExp[qid]
                runExp[qid] = group.exposure(rexp, did2gids[qid], qrels[qid], complete)
            else:
                runExp[qid] = None
    #
    # compute and print per-query metrics
    #
    for qid in targExp.keys():
        #
        # skip queries with null targets.  this happens if there is an
        # upstream problem (e.g. no relevant documents or no groups)
        #
        if (targExp[qid] == None):
            continue
        if (not (qid in runExp)) or (len(runExp[qid]) == 0):
            #
            # defaults for queries in relfn and not in topfn
            #
            disparity[qid].value = disparity[qid].upperBound
            relevance[qid].value = relevance[qid].lowerBound
            difference[qid].value = relevance[qid].upperBound
        else:
            #
            # compute the metrics for queries with results
            #
            r = runExp[qid]
            disparity[qid].compute(r)
            relevance[qid].compute(r)
            difference[qid].compute(r, square)
        #
        # output
        #
        if not non_verbose:
            print("\t".join([disparity[qid].name, qid, disparity[qid].string(normalize)]))
            print("\t".join([relevance[qid].name, qid, relevance[qid].string(normalize)]))
            print("\t".join([difference[qid].name, qid, difference[qid].string(normalize)]))
        if outfile:
            with open(outfile, "a") as df:
                df.write("\t".join([disparity[qid].name, qid, disparity[qid].string(normalize)]) + "\n")
                df.write("\t".join([relevance[qid].name, qid, relevance[qid].string(normalize)]) + "\n")
                df.write("\t".join([difference[qid].name, qid, difference[qid].string(normalize)]) + "\n")


if __name__ == '__main__':
    main()
