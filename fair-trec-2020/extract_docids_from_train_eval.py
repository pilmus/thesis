import json

def extract_docids(filename):
    docids = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            for document in data['documents']:
                docids.append(document['doc_id'])
    return docids


trainfile = 'corpus2020/TREC-Fair-Ranking-training-sample.json'
evalfile = 'corpus2020/TREC-Fair-Ranking-eval-sample.json'
docidsfile = 'corpus2020/TREC-Fair-Ranking-docids.txt'



traindocids = extract_docids(trainfile)
evaldocids = extract_docids(evalfile)

print(f"There are {len(traindocids)} training documents.")
print(f"There are {len(set(traindocids))} UNIQUE training documents.")

print(f"There are {len(evaldocids)} evaluation documents.")
print(f"There are {len(set(evaldocids))} UNIQUE evaluation documents.")

alldocids = traindocids+evaldocids

print(f"There are {len(alldocids)} documents.")
print(f"There are {len(set(alldocids))} UNIQUE documents.")

with open(docidsfile, 'w') as f:
    f.write('\n'.join(set(alldocids)))