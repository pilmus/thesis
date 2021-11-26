# 1. loop through docs from training file
# 2. extract files from corpus that match train/eva
# 3. write them to a new file
import glob
import gzip
import os
import sys

import jsonlines as jsonlines

from fairtrec_util import sample_to_df

eval_file = 'resources/fair-TREC-evaluation-sample.json'
train_file = 'resources/fair-TREC-training-sample.json'
docid_source_files = [eval_file, train_file]

corpdir = 'resources/corpus2019'
subcorpdir = 'resources/subcorpora2019'
if not os.path.exists(subcorpdir):
    os.mkdir(subcorpdir)
eva_subcorp = subcorpdir + '/eva_subcorp.json'
tra_subcorp = subcorpdir + '/tra_subcorp.json'


# def extract_subcorpora(docid_sources):
#     dfs = []
#     for docid_source in docid_sources:
#         dfs.append(sample_to_df(docid_source))

eva_df = sample_to_df(eval_file)
tra_df = sample_to_df(train_file)




corpusfiles = glob.glob(corpdir + "/*.gz")
for corpusfile in corpusfiles:
    with gzip.open(corpusfile) as zipfile:
        reader = jsonlines.Reader(zipfile)
        print(reader.read()['id'] in eva_df.documents)
        print(reader.read()['id'] in tra_df.documents)
        sys.exit()

    # docs = []
#     for doc in tqdm(reader.iter(type=dict), total=100000):
#         if doc['id'] in reldocids:
#             print(f"Doc {doc['id']} found.")
#             docs.append(doc)
#
