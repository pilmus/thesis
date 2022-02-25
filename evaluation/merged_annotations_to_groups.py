import argparse

import pandas as pd

def merge_author_groups(df):
    levels = set(df.level.to_list())
    if 'Advanced' in levels and 'Developing' in levels:
        df['merged_level'] = 'Mixed'
    elif 'Advanced' in levels:
        df['merged_level'] = 'Advanced'
    elif 'Developing' in levels:
        df['merged_level'] = 'Developing'
    elif None in levels:
        df['merged_level'] = None
    else:
        print(df.id)
        print("Then what even is left??")
    return df

def group_mapping(mode):
    if mode == 'basic':
     return {'Advanced': '2', 'Developing': '1', 'Mixed': '1|2', None: ''}
    elif mode == 'mixed_group':
        return {'Advanced': '2', 'Developing': '1', 'Mixed': '3', None: ''}
    elif mode =='nomixed':
        return {'Advanced': '2', 'Developing': '1', 'Mixed': '', None: ''}
    elif mode == 'mix_up':
        return {'Advanced': '2', 'Developing': '1', 'Mixed': '2', None: ''}
    elif mode =='mix_down':
        return {'Advanced': '2', 'Developing': '1', 'Mixed': '1', None: ''}
    else:
        raise ValueError(f"Illegal mode: {mode}.")


def main():
    parser = argparse.ArgumentParser(description='create different group mappings')
    parser.add_argument('-m', dest='mode', default='basic')
    args = parser.parse_args()

    mode = args.mode

    merged_file = 'resources/evaluation/2020/merged-annotations.json'
    eval_file = 'resources/evaluation/2020/TREC-Fair-Ranking-eval-sample.json'

    merged = pd.read_json(merged_file, lines=True)
    eval = pd.read_json(eval_file, lines=True)

    eval = eval.explode('documents')
    eval[['doc_id', 'relevance']] = eval.documents.apply(pd.Series)


    merged = merged.explode('authors')
    merged = merged.reset_index(drop=True)
    merged['level'] = merged.authors.apply(lambda row: row.get('level'))



    merged  = merged.groupby('id').apply(merge_author_groups)

    di = group_mapping(mode)

    merged['group'] = merged.merged_level.map(di)
    merged = merged[['id', 'group']]
    merged = merged.drop_duplicates()

    docs_groups = pd.merge(eval, merged, how='left', left_on='doc_id', right_on='id')

    docs_groups = docs_groups[['doc_id','group']]
    docs_groups = docs_groups.drop_duplicates()
    outfile = f'resources/evaluation/2020/merged-annotations-groups-{mode}.csv'
    docs_groups.to_csv(outfile, index=False)


if __name__ == '__main__':
    main()