import pandas as pd
import numpy as np
from jsonlines import jsonlines

subcorp_eva = 'resources/subcorpora2019/eva_subcorp.jsonl'
subcorp_tra = 'resources/subcorpora2019/tra_subcorp.jsonl'

# dfeva = pd.read_json(subcorp_eva, lines=True)
# dftra = pd.read_json(subcorp_tra, lines=True)

docs = {}

with jsonlines.open(subcorp_eva, 'r') as reader:
    for line in reader:
        docs[line['id']] = line

with jsonlines.open(subcorp_tra, 'r') as reader:
    for line in reader:
        if not line['id'] in docs:
            docs[line['id']] = line

with jsonlines.open('resources/teeeemmp.jsonl','w') as writer:
    writer.write_all(list(docs.values()))