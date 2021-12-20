import gzip
import sys

from tqdm import tqdm
import jsonlines

datadir = 'corpus2019'

reldocids = 'fair-TREC-docids.txt'
with open(reldocids, 'r') as f:
    reldocids = set(f.read().splitlines())


firstdata = datadir + '/s2-corpus-00.gz'
with gzip.open(str(firstdata)) as f:
    reader = jsonlines.Reader(f)
    docs = []
    for doc in tqdm(reader.iter(type=dict), total=100000):
        if doc['id'] in reldocids:
            print(f"Doc {doc['id']} found.")
            docs.append(doc)


    #     docids.append(doc['id'])
    #     i += 1
    #     if i % 1 == 0:
    #         break
    #         print(i)
    # print(i)

# putfile = 'tempname.jsonl'

# with open(putfile,'w') as f:
  #  writer = jsonlines.Writer(f)
   # writer.write(docs)
