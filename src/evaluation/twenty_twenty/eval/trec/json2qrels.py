#!/usr/bin/env python3

import argparse
import json
import csv
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='convert fair ranking qrel json to trec qrels')
    parser.add_argument('-R', dest='relfn')  # relfn == rel file name, NOT rel FUNCTION
    parser.add_argument('-G', dest='groupfn')  # groupfn == group file name
    parser.add_argument('-c', dest='complete', default=False, action='store_true')

    args = parser.parse_args()

    relfn = args.relfn
    groupfn = args.groupfn
    complete = args.complete
    did2gids = {}
    with open(groupfn, "r") as fp:
        for row in csv.reader(fp, delimiter=','):
            did2gids[row[0]] = "|".join(row[1:])

    with open(relfn, "r") as fp:
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
                    print("\t".join([qid, gids, did, rel])) # each qid/did combination gets its own line?


if __name__ == '__main__':
    main()
