import json
import os
import sys

import jsonlines


def read_json(file):
    with open(file) as f:
        dat = json.load(f)
    return dat

def write_json(content, outfile):
    with open(outfile, 'w') as fp:
        json.dump(content, fp)

def read_jsonlines(file, handler=lambda obj: obj):
    dat = []
    with jsonlines.open(file) as reader:
        for obj in reader:
            dat.append(handler(obj))
    return dat

def write_jsonlines(array, file):
    with jsonlines.open(file, mode='w') as writer:
        writer.write_all(array)




def invert_key_to_list_mapping(mapping):
    inverted = {}
    for k, valuelist in mapping.items():
        for value in valuelist:
            if value not in inverted:
                inverted[value] = []
            inverted[value] = inverted[value] + [k]

    return inverted


def block_print():
    """Disable printing."""
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """Enable printing."""
    sys.stdout = sys.__stdout__


def valid_file_with_none(filename):
    return os.path.exists(filename or '') and os.path.isfile(filename or '')