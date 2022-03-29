import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description='convert fair ranking runfile json to trec runfile')
    parser.add_argument('-I', dest='runfile', required=True)
    parser.add_argument('-r', dest='run_id')
    parser.add_argument('-D', dest='destination')
    parser.add_argument('-NV', dest='non_verbose', default=False, action='store_true')

    args = parser.parse_args()
    runfile = args.runfile
    destination = args.destination
    non_verbose = args.non_verbose
    args_runid = args.run_id

    json2runfile(runfile, destination, non_verbose, args_runid)


def json2runfile(runfile, destination, non_verbose, args_runid=None):
    if not (args_runid != None):
        run_id = os.path.basename(runfile)
    else:
        run_id = args_runid
    qid_cnts = {}
    if destination:
        df = open(destination, "w")
    with open(runfile, "r") as fp:
        for line in fp:
            data = json.loads(line.strip())
            qid = "%d" % data["qid"]
            if not (qid in qid_cnts):
                qid_cnts[qid] = 0
            else:
                qid_cnts[qid] = qid_cnts[qid] + 1
            qid_idx = qid_cnts[qid]
            rank = 1
            for did in data["ranking"]:
                row = []
                row.append(qid)
                row.append("Q%d" % qid_idx)
                row.append(did)
                row.append("%d" % rank)
                row.append("%f" % (1.0 / rank))
                row.append(run_id)
                if not non_verbose:
                    print("\t".join(row))
                if destination:
                    df.write("\t".join(row) + "\n")
                rank = rank + 1


if __name__ == '__main__':
    main()
