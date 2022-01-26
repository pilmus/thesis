import pandas as pd

train_file = 'resources/training/2020/DELTR-training-sample.json'
sequence_file = 'resources/training/2020/DELTR-sequence.tsv'

train = pd.read_json(train_file, lines=True)
train = train['qid']
train.to_csv(sequence_file, header=False)
