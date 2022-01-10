import glob
import shutil
import time

import pyterrier as pt
import jsonlines
import os

from generators import doc_generator, condensed_doc_generator

if not pt.started():
  pt.init()



SAMPLE_JSONL = './sample_corp.jsonl'
SAMPLE_TREC = './sample_corp.TREC'
SAMPLE_JSONL_CONDENSED = './sample_corp.condensed.jsonl'


jsonl_index = './sample_jsonl_index'
trec_index = './sample_trec_index'
jsonl_cond_index = './sample_jsonlc_index'





if os.path.exists(jsonl_index):
  shutil.rmtree(jsonl_index)

if os.path.exists(trec_index):
  shutil.rmtree(trec_index)

if os.path.exists(jsonl_cond_index):
  shutil.rmtree(jsonl_cond_index)

t1 = time.time()
# indexer = pt.IterDictIndexer(jsonl_index, blocks=True, meta={'docno':40,'text':4096}, meta_tags={'text':'other'},fields=['title', 'paperabstract', 'other'])
indexer = pt.IterDictIndexer(jsonl_index, blocks=True, meta={'docno':40,'text':4096}, meta_tags={'text':'other'})
indexer.setProperty("termpipelines", "")
# indexer.setProperty("FieldTags.process","title,paperabstract,other")
indexer.index(doc_generator(SAMPLE_JSONL),fields=['title', 'paperabstract', 'other'])

t2 = time.time()
print("--- %s seconds ---" % (t2 - t1))

indexer = pt.TRECCollectionIndexer(trec_index, blocks=True, meta={'docno': 40, 'text': 4096},
                                   meta_tags={'text': 'OTHER'})
indexer.setProperty("termpipelines", "")
indexer.setProperty("FieldTags.process","TITLE,PAPERABSTRACT,OTHER")
indexer.index([SAMPLE_TREC])

t3 = time.time()
print("--- %s seconds ---" % (t3 - t2))

indexer = pt.IterDictIndexer(jsonl_cond_index, blocks=True, meta={'docno':40,'text':4096}, meta_tags={'text':'other'})
indexer.setProperty("termpipelines", "")
indexer.setProperty("FieldTags.process","title,paperabstract,other")
indexer.index(condensed_doc_generator(SAMPLE_JSONL_CONDENSED),fields=['title', 'paperabstract', 'other'])

t4 = time.time()
print("--- %s seconds ---" % (t4 - t3))


# nothreads
# processing document 0
# --- 17.188092470169067 seconds ---
# --- 18.73360776901245 seconds ---
# --- 10.?