import glob

from jsonlines import jsonlines


def dir_generator(dir, generator):
    for file in glob.glob(dir + "/*.jsonl"):
        for doc in generator(file):
            yield doc


def doc_generator(file, printerval=100000):
    with jsonlines.open(file) as reader:
        print(f"Processing {file}")
        for i, doc in enumerate(reader.iter(type=dict, skip_invalid=True)):
            if printerval:
                if i % printerval == 0:
                    print(f"Processing line {i}")
            id_, other, paper_abstract, title = parse_doc_dict(doc)

            d = {
                "docno": id_,
                "title": title,
                "paperabstract": paper_abstract,
                "other": other
                }

            yield d


def condensed_doc_generator(file, printerval = 100000):
    with jsonlines.open(file) as reader:
      print(f"Processing {file}")
      for i, doc in enumerate(reader.iter(type=dict, skip_invalid=True)):
        if printerval:
          if i % printerval == 0:
            print(f"Processing line {i}")
          yield doc


def parse_doc_dict(doc):
    id_ = doc.get('id')
    title = doc.get('title')
    paper_abstract = doc.get("paperAbstract")
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
    return id_, other, paper_abstract, title


if __name__ == '__main__':
    for doc in condensed_doc_generator('sample_corp.condensed.jsonl'):
        print(doc)
