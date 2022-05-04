import argparse
import json
from enum import IntEnum

import numpy as np
import pandas as pd






def singleton_author_mapping(full_mapping, noauth_strategy):

    mapping = load_mapping(full_mapping)

    mapping = {k:'|'.join(v) for k,v in mapping.items()}

    groupdf = pd.DataFrame({'doc_id':list(mapping.keys()),'group':list(mapping.values())})

    if noauth_strategy == 'docid':
        groupdf.group = groupdf.apply(lambda row: (row.group or row.doc_id), axis=1)
    elif noauth_strategy == 'anon':
        groupdf.group = groupdf.apply(lambda row: (row.group or "anonymous"), axis=1)
    else:
        raise ValueError(f"Unsupported noauth_strategy: {noauth_strategy}.")

    return groupdf





def load_mapping(mapping):
    with open(mapping, 'r') as fp:
        doc_to_auth = json.load(fp)
    return doc_to_auth


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', '--outfile', dest='outfile')
    args = parser.parse_args()

    mapping = singleton_author_mapping()

if __name__ == '__main__':
    main()