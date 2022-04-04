import argparse
from enum import IntEnum

import numpy as np
import pandas as pd


class MappingMode(IntEnum):
    BASIC = 1
    MIXED_GROUP = 2
    NOMIXED = 3
    MIX_UP = 4
    MIX_DOWN = 5

    @classmethod
    def has_value(cls, value):
        return value in set(item.value for item in MappingMode)

class Grouping(IntEnum):
    HLevel = 1
    Level = 2

    @classmethod
    def has_value(cls, value):
        return value in set(item.value for item in Grouping)


def group_mapping(grouping, mode):
    if grouping == 'hlevel':
        keys = ["H","L","Mixed",None]
    elif grouping == "level":
        keys = ["Advanced","Developing","Mixed",None]
    else:
        raise ValueError(f"Illegal grouping: {grouping}.")


    if mode == 'basic':
        values = ["2","1","1|2",""]
        # return {'H': '2', 'L': '1', 'Mixed': '1|2', None: ''}
    elif mode == 'mixed_group':
        values = ["2", "1", "3", ""]
        # return {'H': '2', 'L': '1', 'Mixed': '3', None: ''}
    elif mode == 'nomixed':
        values = ["2", "1", "", ""]
        # return {'H': '2', 'Developing': '1', 'Mixed': '', None: ''}
    elif mode == 'mix_up':
        values = ["2", "1", "2", ""]
        # return {'H': '2', 'Developing': '1', 'Mixed': '2', None: ''}
    elif mode == 'mix_down':
        values = ["2", "1", "1", ""]
        # return {'H': '2', 'Developing': '1', 'Mixed': '1', None: ''}
    else:
        raise ValueError(f"Illegal mode: {mode}.")

    mapping = {k:v for k,v in zip(keys, values)}
    print(mapping)
    return mapping





def annotations_to_groups(training_sample, eval_sample, annotations, grouping, mode):
    adf = pd.read_csv(annotations)
    adf = adf.rename({'id':'doc_id'},axis=1)

    tdf = load_sample(training_sample)
    edf = load_sample(eval_sample)

    sdf = pd.concat([tdf,edf])

    mdf = pd.merge(sdf,adf[['doc_id','DocHLevel']],how='left')
    mdf = mdf.replace({np.nan: None})

    mapping = group_mapping(grouping, mode)


    mdf['group'] = mdf.DocHLevel.apply(lambda row: mapping[row])

    return mdf[['doc_id','group']]



def load_sample(training_sample):
    tdf = pd.read_json(training_sample, lines=True)
    tdf = tdf.explode('documents')
    tdf[['doc_id', 'relevance']] = tdf.documents.apply(pd.Series)
    return tdf



