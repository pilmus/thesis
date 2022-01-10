import json
import time

from jsonlines import jsonlines

from generators import doc_generator, parse_doc_dict

gen = doc_generator('/mnt/g/thesis/corpus2019unzips/s2-corpus-00', printerval=100000)

t1 = time.time()

with jsonlines.open('/mnt/g/thesis/corpus2019unzips/s2-corpus-00') as reader:
    with jsonlines.open('/mnt/g/thesis/condensed/s2-corpus-00.jsonl', 'w') as writer:
        for i, doc in enumerate(reader.iter(type=dict, skip_invalid=True)):
            if i % 100000 == 0:
                print(f"Processing line {i}")
            id_, other, paper_abstract, title = parse_doc_dict(doc)

            d = {
                "docno": id_,
                "title": title,
                "paperabstract": paper_abstract,
                "other": other
                }

            writer.write(d)

#
# lines = []
# for item_ in gen:
#     lines.append(item_)
# # lines = list(gen)
#
# with jsonlines.open('/mnt/g/thesis/condensed/s2-corpus-00.tempname.condensed.jsonl', 'w') as writer:
#     writer.write_all(lines)

t2 = time.time()

print("--- %s seconds ---" % (t2 - t1))