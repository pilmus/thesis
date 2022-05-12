import argparse
import json
import csv
import os.path


def main():
    parser = argparse.ArgumentParser(description='convert fair ranking qrel json to trec qrels')
    parser.add_argument('-R', dest='relfn')  # relfn == rel file name, NOT rel FUNCTION
    parser.add_argument('-G', dest='groupfn')  # groupfn == group file name
    parser.add_argument('-c', dest='complete', default=False, action='store_true')
    parser.add_argument('-D', dest='destination')
    parser.add_argument('-NV', dest='not_verbose', default=False, action='store_true')

    args = parser.parse_args()

    relfn = args.relfn
    groupfn = args.groupfn
    complete = args.complete
    destination = args.destination
    not_verbose = args.not_verbose
    json_to_group_qrels(relfn, groupfn, complete, destination, not_verbose)


def json_to_group_qrels(ground_truth_relevance, grouping_file, destination, complete, not_verbose):
    did2gids = {}
    with open(grouping_file, "r") as fp:
        for row in csv.reader(fp, delimiter=','):
            did2gids[row[0]] = "|".join(row[1:])
    if destination:
        if os.path.exists(destination):
            os.remove(destination)
    with open(ground_truth_relevance, "r") as fp:
        for line in fp:
            data = json.loads(line.strip())
            qid = "%d" % data["qid"]
            for d in data["documents"]:
                did = d["doc_id"]
                gids = ""
                if did in did2gids:
                    gids = did2gids[did]
                rel = d["relevance"]
                if complete or (rel > 0):
                    rel = "%d" % rel
                    if destination:
                        with open(destination, 'a') as df:
                            df.write("\t".join([qid, gids, did, rel]) + "\n")
                    if not not_verbose:
                        print("\t".join([qid, gids, did, rel]))  # each qid/did combination gets its own line?


def json_to_base_qrels(gt, destination):
    with open(gt, 'r') as fp:
        for line in fp:
            data = json.loads(line.strip())
            qid = "%d" % data["qid"]
            for d in data["documents"]:
                did = d["doc_id"]
                rel = "%d" % d["relevance"]
                with open(destination, 'a') as f:
                    f.write("\t".join([qid, "0", did, rel, "\n"]))


if __name__ == '__main__':
    main()
