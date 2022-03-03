import csv
import json
from collections import Counter
import numpy as np
import networkx as nx
import load_data


authors_file = "data/corpus-subset-for-queries.authors.csv"
papers_file = "data/corpus-subset-for-queries.papers.csv"
authors_papers_file = "data/corpus-subset-for-queries.paper_authors.csv"
corpus_data_file = "data/corpus-subset-for-queries.jsonl"
training_data_file = 'data/TREC-Fair-Ranking-training-sample.json'

authors = load_data.load_authors(authors_file)
papers = load_data.load_papers(papers_file)
papers = load_data.load_papers_authors(authors_papers_file, papers)
papers = load_data.load_corpus(corpus_data_file, papers)
training_data = load_data.load_training_data(training_data_file)


G_papers = nx.Graph()
G_authors = nx.Graph()
for key in papers.keys():
    for key2 in papers[key]['outCitations']:
        G_papers.add_edge(key, key2) 
    for key2 in papers[key]['inCitations']:
        G_papers.add_edge(key2, key) 
    if 'authors_order' not in papers[key]:
        papers[key]['authors_avg_citations'] = papers[key]['n_citations']
    else:
        for author1 in papers[key]['authors_order']:
            for author2 in papers[key]['authors_order']:
                if author1[0] != author2[0] and author2[0] != '' and author1[0] != '':
                    G_authors.add_edge(author1[0], author2[0]) 
        #print (papers[key]['authors_order'])
        papers[key]['authors_avg_citations'] = np.mean([int(authors[a[0]]['num_citations']) for a in papers[key]['authors_order'] if (a[0] in authors)])
        papers[key]['authors_avg_h_index'] = np.mean([int(authors[a[0]]['h_index']) for a in papers[key]['authors_order'] if (a[0] in authors)])


import community as community_louvain
partion_papers = community_louvain.best_partition(G_papers)
partion_authors = community_louvain.best_partition(G_authors)
json.dump(partion_authors, open('partion_authors.json', 'w'))
"""
query_id = 0
queries_results_reach = []
queries_results_relevant = []
queries_results_nonrelevant = []
print ("comunities of authors for the query:", training_data[query_id]['query'])
for query in training_data:
    curr_partition_relevant = set()
    curr_partition_all = set()
    for d in query['documents']:
        if d['doc_id'] in papers:
            curr_partition = set()
            if 'authors_order' in papers[d['doc_id']]:
                for author in papers[d['doc_id']]['authors_order']:
                    if author[0] in authors:
                        #curr_partition.append(authors[author[0]]['h_class'])
                        if author[0] in partion_authors:
                            curr_partition.add(str(partion_authors[author[0]]))
                            #curr_partition.append(authors[author[0]]['h_class'])

                if d['relevance'] == 1 and (len(curr_partition)>0):
                    #queries_results.append(''.join(set(curr_partition)))
                    curr_partition_relevant.update(curr_partition)
                curr_partition_all.update(curr_partition)
                #print (list(curr_partition), d['relevance'])
    queries_results_reach += list(curr_partition_all)
    queries_results_relevant += list(curr_partition_relevant)
    queries_results_nonrelevant += list(curr_partition_all-curr_partition_relevant)"
"""
