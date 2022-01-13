import csv
import json


def load_authors(authors_file):
    reader = csv.DictReader(open(authors_file))
    authors = {}
    for author in reader:
        authors[author['corpus_author_id']] = dict(author)
    return authors

def load_papers(papers_file):
    reader = csv.DictReader(open(papers_file))
    papers = {}
    for paper in reader:
        papers[paper['paper_sha']] = dict(paper)
    return papers

def load_papers_authors(authors_papers_file, papers):
    reader = csv.DictReader(open(authors_papers_file))
    for author_paper in reader:
        if 'authors_order' not in papers[author_paper['paper_sha']]:
            papers[author_paper['paper_sha']]['authors_order'] = []
        papers[author_paper['paper_sha']]['authors_order'].append((author_paper['corpus_author_id'], author_paper['position']))
    return papers


def load_corpus(corpus_data_file, papers):
    with open(corpus_data_file) as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        papers[result['id']].update(result)
    return papers


def load_training_data(training_data_file):
    with open(training_data_file) as json_file:
        json_list = list(json_file)
    
    training_data = []
    for json_str in json_list:
        result = json.loads(json_str)
        training_data.append(result)
    return training_data


if __name__ == "__main__":
    authors_file = "data/corpus-subset-for-queries.authors.csv"
    papers_file = "data/corpus-subset-for-queries.papers.csv"
    authors_papers_file = "data/corpus-subset-for-queries.paper_authors.csv"
    corpus_data_file = "data/corpus-subset-for-queries.jsonl"
    training_data_file = 'data/TREC-Fair-Ranking-training-sample.json'

    authors = load_authors(authors_file)
    papers = load_papers(papers_file)
    papers = load_papers_authors(authors_papers_file, papers)
    papers = load_corpus(corpus_data_file, papers)
    training_data = load_training_data(training_data_file)


