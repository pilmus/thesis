def doc_generator(reader, reldocids=[]):
    for doc in reader.iter(type=dict, skip_invalid=True):
        if not reldocids or doc['id'] in reldocids:
            author_names = []
            author_ids = []
            for obj in doc.get('authors'):
                author_ids.extend(obj.get('ids'))
                author_names.append(obj.get('name'))

            yield {
                "_index": 'semanticscholar',
                "_type": "document",
                "_id": doc.get('id'),
                "title": doc.get('title'),
                "paperAbstract": doc.get("paperAbstract"),
                "entities": doc.get("entities"),
                "author_names": author_names,
                "author_ids": author_ids,
                "inCitations": len(doc.get("inCitations")),
                "outCitations": len(doc.get("outCitations")),
                "year": doc.get("year"),
                "venue": doc.get('venue'),
                "journalName": doc.get('journalName'),
                "journalVolume": doc.get('journalVolume'),
                "sources": doc.get('sources'),
                "doi": doc.get('doi')
                }