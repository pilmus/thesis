#!/usr/bin/env python

import sys
import json
import argparse

def read_queries(fn):
    queries = {}
    with open(fn,"r") as fp:
        for line in fp:
            line = line.strip()
            try:
                data = json.loads(line)
                qid = data["qid"]
                documents = map(lambda x: x["doc_id"], data["documents"])
                queries[qid]=set(documents)
            except json.JSONDecodeError:
                print("illegal json at line %d", line_number)
    return queries

def read_query_sequence(fn):
    query_sequence={}
    with open(fn,"r") as fp:
        for line in fp:
            qno,qid = line.strip().split(",")
            query_sequence[qno]=int(qid)
    return query_sequence

def main():
    parser = argparse.ArgumentParser(description='validate trec fair ranking run.')
    parser.add_argument('--queries',help='fair ranking query file')
    parser.add_argument('--query_sequence_file',help='fair ranking query sequences file')
    parser.add_argument('--run_file',help='fair ranking run file')

    args = parser.parse_args()

    queries = read_queries(args.queries)
    query_sequence = read_query_sequence(args.query_sequence_file)
    query_sequence_seen = set([])

    line_number = 1
    with open(args.run_file,"r") as fp:
        for line in fp:
            line = line.strip()
            try:
                data = json.loads(line)
                #
                # 1. check fields in json object
                #
                for field in ["q_num","qid","ranking"]:
                    if not(field in data):
                        print("missing %s in line %d"%(field,line_number))
                        sys.exit()

                #
                # 2. validate query number
                #
                q_num = data["q_num"]
                if not(q_num in query_sequence):
                    print("%s not found in sequence file (line %d)"%(q_num,line_number))
                    sys.exit()

                #
                # 3. validate qid
                #
                qid = data["qid"]
                if not(qid in queries):
                    print("%s not found in query file (line %d)"%(qid,line_number))
                    sys.exit()

                #
                # 4. validate qid matches query number
                #
                if (qid != query_sequence[q_num]):
                    print("%s is not the correct qid for sequence number %s (line %d)"%(qid,q_num,line_number))
                    print("\tshould be %s"%(query_sequence[q_num])
                    sys.exit()

                #
                # 5. check for duplicate docids
                #
                ranking = data["ranking"]
                ranking_set = set(ranking)
                if (len(ranking) != len(ranking_set)):
                    print("duplicate document ids (line %d)"%(line_number))
                    sys.exit()

                #
                # 6. check for extra documents
                #
                if (len(ranking_set-queries[qid])>0):
                    print("extra document ids (line %d)"%(line_number))
                    sys.exit()

                #
                # 7. check for missing documents
                #
                if (len(queries[qid]-ranking_set)>0):
                    print("missing document ids (line %d)"%(line_number))
                    sys.exit()

                #
                # 8. check for duplicate query number
                #
                if (q_num in query_sequence_seen):
                    print("duplicate q_num (line %d)"%(line_number))
                    sys.exit()
                query_sequence_seen.add(q_num)

            except ValueError:
                print("illegal json at line %d"%line_number
                sys.exit()

            line_number = line_number + 1

        #
        # 9. check missing query numbers
        #
        if (len(query_sequence_seen)!=len(query_sequence.keys())):
            print("missing query numbers"
            sys.exit()
        

if __name__== "__main__":
  main()
