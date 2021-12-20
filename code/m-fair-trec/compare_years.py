from fairtrec_util import sample_to_df

set2019 = set(sample_to_df('fair-trec-2019/resources/fair-TREC-training-sample.json').documents.tolist() +
              sample_to_df('fair-trec-2019/resources/fair-TREC-evaluation-sample.json').documents.tolist())

set2020 = set(sample_to_df('fair-trec-2020/resources/TREC-Fair-Ranking-eval-sample.json').documents.tolist() +
              sample_to_df('fair-trec-2020/resources/TREC-Fair-Ranking-training-sample.json').documents.tolist())

print(f"There are {len(set2019)} unique documents in the 2019 files.")
print(f"There are {len(set2020)} unique documents in the 2020 files.")
print(f"There are {len(set2020.intersection(set2019))} common files in 2019 and 2020.")
