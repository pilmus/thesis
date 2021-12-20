#!/usr/bin/env python

# TREC Fair Ranking Track
# Query sequence generator

import argparse
import csv
import numpy as np
import json


def load_query_distribution(filename):

    """
        Reads a query distribution from a tsv file

        Input: a filepath containing query distribution in the form:
            query-id \t frequency

        Output: a list of tuples of the form:
            [(query_id, query_freq), ...]
    """

    query_distribution = []
    with open(filename, 'r') as f_in:
        for line in f_in:
            q = json.loads(line)
            query_distribution.append((q['qid'], q['frequency']))
    return query_distribution


def generate_sequence(seq_len, query_distribution):
    
    """
        Generates a list of query IDs sampled from a given distribution 
        with replacement.

        Input: 
            seq_len: length of the sequence to be generated
            query_distribution: a list of tuples of the form:
                [(query_id, query_freq), ...]
        Output:
            a list of query IDs
    """

    np.random.seed()

    #normailze the frequencies to form a distribution
    query_ids, distribution = zip(*query_distribution)
    distribution /= sum(np.array(distribution))

    return np.random.choice(query_ids, size=seq_len,
        replace=True, p=distribution)



if __name__ == '__main__':

    """
        This script generates a list of query IDs sampled from a given distribution 
        with replacement.

        Example usage:
            python query-sequence-generator.py 10 TREC-fair-training-sample.json

            python query-sequence-generator.py 10 TREC-fair-training-sample.json > training-sequence.tsv
        
        Args:
            seq-len: length of the sequence to be generated.

            distribution filename: tsv file containing the query distribution of the format:
                query-id \t frequency

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('seq_len', metavar='seq-len', type=int, 
        help='Length of the query sequence')
    parser.add_argument('query_distribution_file', 
        metavar='query-distribution-file', type=str, 
        help='File with the query distribution')
    args = parser.parse_args()

    query_sequence = generate_sequence(args.seq_len, 
        load_query_distribution(args.query_distribution_file))

    print('\n'.join(["%d,%s" % (i, qid) 
        for i, qid in enumerate(query_sequence)]))

