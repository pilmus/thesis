from fairtrec_util import sample_to_df
import pandas as pd



eval_file = 'resources/fair-TREC-evaluation-sample.json'
train_file = 'resources/fair-TREC-training-sample.json'

eva_df = sample_to_df(eval_file)
tra_df = sample_to_df(train_file)
print(len(tra_df))
all_df = pd.concat([eva_df, tra_df])
all_df = all_df.drop_duplicates()

print(all_df.describe())
