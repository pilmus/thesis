import os

import pandas as pd
from jsonlines import jsonlines

def __dummy(df):
    df
    return df

group_file = 'resources/evaluation/2020/fair-TREC-sample-author-groups.csv'
annot_file = 'resources/evaluation/2020/merged-annotations.json'


group_df = pd.read_csv(group_file)
annot_df = pd.read_json(annot_file, lines=True)
annot_df = annot_df.explode('authors')


merged_df = pd.merge(annot_df, group_df, how = 'left', on = ['author_id'])
merged_df = merged_df.fillna(-1)

merged_df.groupby('id').apply(__dummy)


print(group_df.head())
print(annot_df.head())

#
#
# with jsonlines.open(annot_file) as reader:
#     for line in reader:
#         print(line['id'])