from jsonlines import jsonlines

CORPUS_DIR = "/mnt/g/thesis/corpus2019unzips"


def doc_generator(file):
    with jsonlines.open(file) as reader:
        for i, doc in enumerate(reader.iter(type=dict, skip_invalid=True)):
            if i % 100000 == 0:
                print(f'processing document {i}')

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

            text = ' '.join([id_, title, paper_abstract, other])

            d = {
                "docno": id_,
                "title": title,
                "paperabstract": paper_abstract,
                "other": other,
                "text": text
                }

            d = f"""<DOC>
<DOCNO>{id_}</DOCNO>
<TITLE>{title}</TITLE>
<PAPERABSTRACT>{paper_abstract}</PAPERABSTRACT>
<OTHER>{other}</OTHER>
</DOC>
"""

            yield d


gen = doc_generator('./sample_corp.jsonl')


with open('./sample_corp.TREC', 'a') as fp:
    for doc in gen:
        fp.write(doc)

