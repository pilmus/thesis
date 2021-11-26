# 1. loop through docs from training file
# 2. extract files from corpus that match train/eva
# 3. write them to a new file
import glob
import os

import jsonlines as jsonlines
from tqdm import tqdm

from fairtrec_util import sample_to_df

eval_file = 'resources/fair-TREC-evaluation-sample.json'
train_file = 'resources/fair-TREC-training-sample.json'
docid_source_files = [eval_file, train_file]

corpdir = 'resources/corpus2019'
subcorpdir = 'resources/subcorpora2019'
if not os.path.exists(subcorpdir):
    os.mkdir(subcorpdir)
eva_subcorp = subcorpdir + '/eva_subcorp.jsonl'
tra_subcorp = subcorpdir + '/tra_subcorp.jsonl'

# def extract_subcorpora(docid_sources):
#     dfs = []
#     for docid_source in docid_sources:
#         dfs.append(sample_to_df(docid_source))

eva_df = sample_to_df(eval_file)
eva_docs = set(eva_df.documents.to_list())
tra_df = sample_to_df(train_file)
tra_docs = set(tra_df.documents.to_list())

DEFAULT_LAST_NUM = '00'

with jsonlines.open(eva_subcorp, mode="r") as eva_reader, jsonlines.open(tra_subcorp, mode="r") as tra_reader:
    try:
        *_, last_eva = eva_reader
        last_eva_corpfile = last_eva['corpus_file']
        last_eva_num = last_eva_corpfile[-2:]
    except ValueError:
        last_eva_num = DEFAULT_LAST_NUM

    try:
        *_, last_tra = tra_reader
        last_tra_corpfile = last_tra['corpus_file']
        last_tra_num = last_tra_corpfile[-2:]
    except ValueError:
        last_tra_num = DEFAULT_LAST_NUM

    last_num = min(last_eva_num, last_tra_num)
    last_ten = last_num[0]
    last_one = last_num[1]
#
# with gzip.open('resources/corpus2019/s2-corpus-00.gz') as zip:
#     reader = jsonlines.Reader(zip)
#     print(len([line for line in reader]))
#
# sys.exit()

with jsonlines.open(eva_subcorp, mode="a") as eva_writer, jsonlines.open(tra_subcorp, mode="a") as tra_writer:
    corpusfiles = glob.glob(corpdir + f"/*[{last_ten}-9][{last_one}-9]")

    for corpusfile in corpusfiles:
        print(f"Extracting subcorpora from {corpusfile}...")
        # corpusfile = 'resources/corpus2019/sample-S2-records'
        with jsonlines.open(corpusfile) as reader:
            eva_lines = []
            tra_lines = []
            for line in tqdm(reader, total=1000000):
            # for line in tqdm(reader, total=102):
                if line['id'] in eva_docs:
                    line['corpus_file'] = os.path.basename(corpusfile)
                    eva_lines.append(line)
                if line['id'] in tra_docs:
                    line['corpus_file'] = os.path.basename(corpusfile)
                    tra_lines.append(line)
            print(f"Found {len(eva_lines)} eva docs...")
            print(f"Found {len(tra_lines)} tra docs...")
            print("Saving extracted documents...")
            eva_writer.write_all(eva_lines)
            tra_writer.write_all(tra_lines)
        # break



    # docs = []
#     for doc in tqdm(reader.iter(type=dict), total=100000):
#         if doc['id'] in reldocids:
#             print(f"Doc {doc['id']} found.")
#             docs.append(doc)
#
