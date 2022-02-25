import pandas as pd

"""
Andres Ferraro in correspondence said the only training data he used for deltr was documents that have an H-class.
So to make a training sequence, we simply take each query for which all the docs have an H-class and put them in 
a sequence.
"""

train_file = 'resources/training/2020/DELTR-training-sample.json'
sequence_file = 'resources/training/2020/DELTR-sequence.tsv'

train = pd.read_json(train_file, lines=True)
train = train['qid']
train.to_csv(sequence_file, header=False)
