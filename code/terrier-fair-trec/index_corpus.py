import glob
import shutil

import pyterrier as pt
import jsonlines
import os

if not pt.started():
  pt.init()


# dict_keys(['entities', 'journalVolume', 'journalPages', 'pmid', 'year', 'outCitations', 's2Url', 's2PdfUrl', 'id', 'authors', 'journalName', 'paperAbstract', 'inCitations', 'pdfUrls', 'title', 'doi', 'sources', 'doiUrl', 'venue'])
def doc_generator():
  for filename in glob.glob("/mnt/g/thesis/corpus2019unzips/s2-corpus-[0-9][0-9]"):
    with jsonlines.open(filename) as reader:
      for i, doc in enumerate(reader.iter(type=dict, skip_invalid=True)):
        # if i == 1000:
        #   break
        if i % 100000 == 0:
          print(f'processing document {i}')
        entities = ' '.join(doc.get('entities'))
        journal_volume = doc.get('journalVolume')
        journal_pages = doc.get('journalPages')
        pmid = doc.get('pmid')
        year = str(doc.get('year'))
        outcitations = ' '.join(doc.get('outCitations'))
        incitations = ' '.join(doc.get('inCitations'))
        s2_url = doc.get('s2Url')
        s2_pdf_url = doc.get('s2PdfUrl')
        journal_name = doc.get('journalName')
        pdf_urls = ' '.join(doc.get('pdfUrls'))
        doi = doc.get('doi')
        sources = ' '.join(doc.get('sources'))
        doi_url = doc.get('doiUrl')
        venue = doc.get('venue')


        author_names = []
        author_ids = []
        for obj in doc.get('authors'):
          author_ids.extend(obj.get('ids'))
          author_names.append(obj.get('name'))
        author_ids = ' '.join(author_ids)
        author_names = ' '.join(author_names)


        other = ' '.join(
          [entities, journal_volume, journal_pages, pmid, year, outcitations, incitations, s2_url, s2_pdf_url,
           journal_name, pdf_urls, doi, doi_url, sources, venue, author_ids, author_names])

        # yield {
        #   'docno': doc.get('id'),
        #   'title': 'horse'
        #   }

        d = {
          "docno": doc.get('id'),
          "title": doc.get('title'),
          "paper_abstract": doc.get("paperAbstract"),
          "other": other
          } # field names have to be lower case?
        # print(d)

        yield d
#
if os.path.exists('./iter_index'):
  shutil.rmtree('./iter_index')


iter_indexer = pt.IterDictIndexer("./iter_index")
doc_iter = doc_generator()
indexref = iter_indexer.index(doc_iter, fields=['title'])